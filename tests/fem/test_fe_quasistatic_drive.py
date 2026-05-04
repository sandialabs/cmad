"""Tests for ``fe_quasistatic_drive``.

Covers four contracts: a single-step CLOSED_FORM drive matches
``fe_newton_solve`` directly; a multi-step COUPLED + Elastic drive
populates ``U_history`` and ``xi_history_by_block`` to length
``len(t_schedule)`` and the final-step solution matches the same
direct call; a J2-plasticity yield-crossing schedule reproduces the
recorded ``xi`` when the time loop is restarted from
``state.xi_at(prev_step, block)`` (catching axis-transposition bugs
in the gather / scatter path); and a mixed-mode 2-block problem
keeps every block's xi-history list shape-aligned across steps,
with the CLOSED_FORM block's entries echoing the init-tile
unchanged.

The unit-cube uniaxial-stretch fixture pins one component on each
``-`` face (xmin → u_x=0, ymin → u_y=0, zmin → u_z=0) and prescribes
``u_x(t) = slope * t`` on the +x face via a callable DBC. ``y=1``
and ``z=1`` faces are unconstrained, so the cube freely contracts
in y / z under the imposed stretch (uniaxial-stress conditions).
"""
import unittest
from typing import cast

import numpy as np
from jax.tree_util import tree_map
from numpy.typing import NDArray

from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.driver import fe_quasistatic_drive
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import Mesh, StructuredHexMesh
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.model import Model
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.parameters.parameters import Parameters
from cmad.typing import PyTreeDict
from tests.support.test_problems import J2AnalyticalProblem

_KAPPA = 100.0
_MU = 50.0


def _make_elastic_model() -> Elastic:
    values = cast(PyTreeDict, {"elastic": {"kappa": _KAPPA, "mu": _MU}})
    active = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Elastic(
        Parameters(values, active, transforms),
        def_type=DefType.FULL_3D,
    )


def _make_J2_model() -> SmallElasticPlastic:
    return SmallElasticPlastic(
        J2AnalyticalProblem().J2_parameters,
        def_type=DefType.FULL_3D,
    )


def _uniaxial_dbcs(slope: float) -> list[DirichletBC]:
    """Symmetry pins on -x / -y / -z faces and a t-ramped u_x on +x.

    The +x DBC's value callable returns ``(N_set, 1)`` — one
    component per node on the +x face — with ``slope * t`` as the
    constant value across the face.
    """
    def u_x_at_t(
            coords: NDArray[np.floating], t: float,
    ) -> NDArray[np.floating]:
        return np.full((coords.shape[0], 1), slope * t)

    return [
        DirichletBC(
            sideset_names=["xmin_sides"], field_name="u",
            dofs=(0,), values=None,
        ),
        DirichletBC(
            sideset_names=["ymin_sides"], field_name="u",
            dofs=(1,), values=None,
        ),
        DirichletBC(
            sideset_names=["zmin_sides"], field_name="u",
            dofs=(2,), values=None,
        ),
        DirichletBC(
            sideset_names=["xmax_sides"], field_name="u",
            dofs=(0,), values=u_x_at_t,
        ),
    ]


def _build_uniaxial_fe_problem(
        mesh: Mesh,
        models_by_block: dict[str, Model],
        modes_by_block: dict[str, GlobalResidualMode],
        slope: float,
) -> FEProblem:
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    dof_map = build_dof_map(
        mesh, [layout], _uniaxial_dbcs(slope),
        components_by_field={"u": 3},
    )
    gr = SmallDispEquilibrium(ndims=3)
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=gr,
        models_by_block=models_by_block,
        modes_by_block=modes_by_block,
    )


def _split_into_two_blocks(mesh: Mesh) -> Mesh:
    n_elems = mesh.connectivity.shape[0]
    half = n_elems // 2
    return Mesh(
        nodes=mesh.nodes,
        connectivity=mesh.connectivity,
        element_family=mesh.element_family,
        element_blocks={
            "left": np.arange(0, half, dtype=np.intp),
            "right": np.arange(half, n_elems, dtype=np.intp),
        },
        node_sets=mesh.node_sets,
        side_sets=mesh.side_sets,
    )


class TestDriveSingleStepClosedForm(unittest.TestCase):
    """A single-step CLOSED_FORM drive produces the same U as a
    direct ``fe_newton_solve`` call at the step's t. Regression-
    protects the pure CLOSED_FORM driver path that all existing
    callers depend on."""

    def test_matches_direct_newton_solve(self) -> None:
        mesh = StructuredHexMesh(
            lengths=(1.0, 1.0, 1.0), divisions=(1, 1, 1),
        )
        models_by_block: dict[str, Model] = {"all": _make_elastic_model()}
        fe_problem = _build_uniaxial_fe_problem(
            mesh,
            models_by_block,
            {"all": GlobalResidualMode.CLOSED_FORM},
            slope=5e-4,
        )
        t_schedule = [0.0, 1.0]

        state = fe_quasistatic_drive(fe_problem, t_schedule)

        n_dofs = fe_problem.dof_map.num_total_dofs
        U_prev_zeros = np.zeros(n_dofs, dtype=np.float64)
        U_direct, _, _, _ = fe_newton_solve(
            fe_problem, U_prev=U_prev_zeros, t=1.0,
        )
        self.assertEqual(len(state.U_history), 2)
        np.testing.assert_allclose(
            state.U_at(1), U_direct, rtol=0.0, atol=1e-12,
        )


class TestDriveMultiStepCoupledElastic(unittest.TestCase):
    """A 3-entry t_schedule on COUPLED + Elastic populates U /
    xi history to length 3, and the final-step solution matches a
    direct ``fe_newton_solve`` call with the recorded prev-step
    state. Validates the multi-step loop machinery (state.append,
    state.xi_at, xi_prev_by_block plumbing) without requiring
    plasticity."""

    def test_multi_step_history_and_final_match(self) -> None:
        mesh = StructuredHexMesh(
            lengths=(1.0, 1.0, 1.0), divisions=(1, 1, 1),
        )
        models_by_block: dict[str, Model] = {"all": _make_elastic_model()}
        fe_problem = _build_uniaxial_fe_problem(
            mesh,
            models_by_block,
            {"all": GlobalResidualMode.COUPLED},
            slope=5e-4,
        )
        t_schedule = [0.0, 0.5, 1.0]

        state = fe_quasistatic_drive(fe_problem, t_schedule)

        self.assertEqual(len(state.U_history), 3)
        self.assertEqual(len(state.xi_history_by_block["all"]), 3)
        self.assertEqual(state.xi_at(1, "all").shape, (1, 8, 6))

        U_direct, xi_direct, _, _ = fe_newton_solve(
            fe_problem,
            U_prev=state.U_at(1),
            t=1.0,
            xi_prev_by_block={"all": state.xi_at(1, "all")},
        )
        np.testing.assert_allclose(
            state.U_at(2), U_direct, rtol=0.0, atol=1e-12,
        )
        np.testing.assert_allclose(
            state.xi_at(2, "all"), xi_direct["all"],
            rtol=0.0, atol=1e-12,
        )


class TestDriveXiRestartConsistency(unittest.TestCase):
    """A J2-plasticity yield-crossing schedule's recorded xi at
    step n-1, fed directly back to ``fe_newton_solve`` at step n,
    reproduces the driver's recorded xi at step n. Catches axis-
    transposition bugs in the gather / scatter path: if ``xi_prev``
    is laid out wrong, the local Newton at each IP receives the
    wrong previous-step state and the converged xi diverges from
    the driver's record."""

    def test_xi_restart_matches_driver_record(self) -> None:
        # J2 yield strain = Y/E = 200/200e3 = 1e-3.
        # slope = 1.5e-3 puts step 1 at 1.5*ε_y (plastic) and
        # step 2 at 3*ε_y (more plastic).
        mesh = StructuredHexMesh(
            lengths=(1.0, 1.0, 1.0), divisions=(1, 1, 1),
        )
        models_by_block: dict[str, Model] = {"all": _make_J2_model()}
        fe_problem = _build_uniaxial_fe_problem(
            mesh,
            models_by_block,
            {"all": GlobalResidualMode.COUPLED},
            slope=1.5e-3,
        )
        t_schedule = [0.0, 1.0, 2.0]
        state = fe_quasistatic_drive(fe_problem, t_schedule)

        # Sanity: at least one IP plastic (alpha > 0) at step 2.
        # SmallElasticPlastic FULL_3D xi = [vec_cauchy(6), alpha(1)],
        # ravel-flat → alpha at index 6.
        alpha_step2 = state.xi_at(2, "all")[..., 6]
        self.assertTrue(
            bool(np.any(alpha_step2 > 0.0)),
            "expected at least one IP in plastic regime at step 2",
        )

        # Restart from recorded xi_1; verify identical xi_2.
        _, xi_direct, _, _ = fe_newton_solve(
            fe_problem,
            U_prev=state.U_at(1),
            t=2.0,
            xi_prev_by_block={"all": state.xi_at(1, "all")},
        )
        np.testing.assert_allclose(
            state.xi_at(2, "all"), xi_direct["all"],
            rtol=0.0, atol=1e-12,
        )


class TestDriveMixedModeFEState(unittest.TestCase):
    """A 2-block mesh (left CLOSED_FORM, right COUPLED) keeps both
    blocks' xi-history lists shape-aligned with the U history. The
    CLOSED_FORM block's per-step xi entries echo the init-tile from
    ``FEState.from_problem`` unchanged across the time loop."""

    def test_mixed_mode_xi_history_alignment(self) -> None:
        base_mesh = StructuredHexMesh(
            lengths=(2.0, 1.0, 1.0), divisions=(2, 1, 1),
        )
        mesh = _split_into_two_blocks(base_mesh)
        models_by_block: dict[str, Model] = {
            "left": _make_elastic_model(),
            "right": _make_elastic_model(),
        }
        fe_problem = _build_uniaxial_fe_problem(
            mesh,
            models_by_block,
            {
                "left": GlobalResidualMode.CLOSED_FORM,
                "right": GlobalResidualMode.COUPLED,
            },
            slope=5e-4,
        )
        t_schedule = [0.0, 0.5, 1.0]
        state = fe_quasistatic_drive(fe_problem, t_schedule)

        self.assertEqual(len(state.U_history), 3)
        self.assertEqual(len(state.xi_history_by_block["left"]), 3)
        self.assertEqual(len(state.xi_history_by_block["right"]), 3)
        # CLOSED_FORM block: xi entries echo the init-tile across
        # all steps. Init-tile from ``FEState.from_problem`` is
        # ``np.tile(init_xi_flat, (n_elems, n_ips, 1))`` and Elastic
        # FULL_3D's ``_init_xi`` is ``[zeros(6)]``, so every entry
        # is a zero array of shape ``(1, 8, 6)``.
        for step in range(3):
            np.testing.assert_array_equal(
                state.xi_at(step, "left"),
                np.zeros((1, 8, 6), dtype=np.float64),
            )


if __name__ == "__main__":
    unittest.main()
