"""Assembly-layer COUPLED-mode tests.

Validates that ``assemble_element_block`` dispatches on per-block mode
and produces the right ``xi_solved_per_block`` shape; that COUPLED on
a closed-form-equivalent Elastic model produces the same global K and
R as the CLOSED_FORM path; that mixed-mode FE problems return
``xi_solved_by_block`` with keys exactly equal to the COUPLED-block
subset; and that a missing xi_prev entry for a COUPLED block raises
``ValueError``.

The per-element COUPLED kernel itself is covered in
``tests/fem/test_per_element_coupled.py`` — these tests focus on the
assemble-layer dispatch and per-block scatter / gather contracts.
"""
import unittest
from typing import cast

import numpy as np
from jax.tree_util import tree_map
from numpy.typing import NDArray

from cmad.fem.assembly import assemble_element_block, assemble_global
from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import Mesh, StructuredHexMesh
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.typing import PyTreeDict

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


def _build_fe_problem(
        mesh: Mesh,
        modes_by_block: dict[str, GlobalResidualMode],
) -> FEProblem:
    """Build a SmallDispEquilibrium FEProblem on ``mesh`` with the
    given per-block mode dict. Each block gets its own Elastic FULL_3D
    model with shared (kappa, mu); the standard six-face DBC pattern
    pins the boundary so the dof_map has prescribed indices, but the
    BCs are not exercised by the assembly tests themselves."""
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    bc = DirichletBC(
        sideset_names=[
            "xmin_sides", "xmax_sides",
            "ymin_sides", "ymax_sides",
            "zmin_sides", "zmax_sides",
        ],
        field_name="u",
        dofs=(0, 1, 2),
        values=None,
    )
    dof_map = build_dof_map(
        mesh, [layout], [bc], components_by_field={"u": 3},
    )
    gr = SmallDispEquilibrium(ndims=3)
    models_by_block: dict[str, Model] = {
        b: _make_elastic_model() for b in mesh.element_blocks
    }
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=gr,
        models_by_block=models_by_block,
        modes_by_block=modes_by_block,
    )


def _split_into_two_blocks(mesh: Mesh) -> Mesh:
    """Re-partition ``mesh.element_blocks`` into two halves named
    ``"left"`` / ``"right"``. Same nodes / connectivity / sidesets;
    only the block partition changes."""
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


class TestAssembleElementBlockCoupledShape(unittest.TestCase):
    """``assemble_element_block`` returns the new 4-tuple; for a
    COUPLED block the fourth slot is ``xi_solved_per_block`` shaped
    ``(n_elems_block, n_ips, total_xi_dofs)``, and for a CLOSED_FORM
    block it is ``None``."""

    def test_coupled_returns_xi_solved_per_block(self) -> None:
        mesh = StructuredHexMesh(
            lengths=(1.0, 1.0, 1.0), divisions=(1, 1, 1),
        )
        fe_problem = _build_fe_problem(
            mesh, {"all": GlobalResidualMode.COUPLED},
        )
        n_dofs = fe_problem.dof_map.num_total_dofs
        U = np.zeros(n_dofs, dtype=np.float64)
        # Elastic FULL_3D: total_xi_dofs = 6 (vec_cauchy only),
        # 8 IPs from the default degree-2 hex Gauss-Legendre rule.
        xi_prev = np.zeros((1, 8, 6), dtype=np.float64)

        R_global = np.zeros(n_dofs, dtype=np.float64)
        rows, cols, vals, xi_solved = assemble_element_block(
            R_global, fe_problem, "all", U, U, t=0.0,
            xi_prev_per_block=xi_prev,
        )
        assert xi_solved is not None
        self.assertEqual(xi_solved.shape, (1, 8, 6))
        self.assertEqual(rows.ndim, 1)
        self.assertEqual(cols.ndim, 1)
        self.assertEqual(vals.ndim, 1)

    def test_closed_form_returns_xi_solved_None(self) -> None:
        mesh = StructuredHexMesh(
            lengths=(1.0, 1.0, 1.0), divisions=(1, 1, 1),
        )
        fe_problem = _build_fe_problem(
            mesh, {"all": GlobalResidualMode.CLOSED_FORM},
        )
        n_dofs = fe_problem.dof_map.num_total_dofs
        U = np.zeros(n_dofs, dtype=np.float64)
        R_global = np.zeros(n_dofs, dtype=np.float64)

        _, _, _, xi_solved = assemble_element_block(
            R_global, fe_problem, "all", U, U, t=0.0,
        )
        self.assertIsNone(xi_solved)


class TestAssembleGlobalCoupledClosedFormEquivalence(unittest.TestCase):
    """Elastic FULL_3D in COUPLED produces the same global K, R as
    the CLOSED_FORM path on the same problem. Elastic's local Newton
    converges in one iteration on a U-linear residual; the IFT-
    corrected total dR/dU reduces to the direct CLOSED_FORM
    derivative, so block-by-block parity at the kernel level (covered
    in test_per_element_coupled.py) carries through to the
    assemble_global layer."""

    def test_K_R_match_closed_form(self) -> None:
        mesh = StructuredHexMesh(
            lengths=(1.0, 1.0, 1.0), divisions=(2, 2, 2),
        )
        fe_closed = _build_fe_problem(
            mesh, {"all": GlobalResidualMode.CLOSED_FORM},
        )
        fe_coupled = _build_fe_problem(
            mesh, {"all": GlobalResidualMode.COUPLED},
        )
        n_dofs = fe_closed.dof_map.num_total_dofs
        # Some non-trivial U so internal stress is non-zero.
        rng = np.random.default_rng(seed=42)
        U = 1e-3 * rng.standard_normal(n_dofs)

        K_closed, R_closed, xi_closed = assemble_global(
            fe_closed, U, U, t=0.0,
        )
        n_elems = mesh.connectivity.shape[0]
        xi_prev_by_block: dict[str, NDArray[np.floating]] = {
            "all": np.zeros((n_elems, 8, 6)),
        }
        K_coupled, R_coupled, xi_coupled = assemble_global(
            fe_coupled, U, U, t=0.0,
            xi_prev_by_block=xi_prev_by_block,
        )

        np.testing.assert_allclose(R_closed, R_coupled, atol=1e-10)
        np.testing.assert_allclose(
            K_closed.toarray(), K_coupled.toarray(), atol=1e-10,
        )
        self.assertEqual(xi_closed, {})
        self.assertEqual(set(xi_coupled.keys()), {"all"})


class TestAssembleGlobalCoupledMixedMode(unittest.TestCase):
    """``xi_solved_by_block.keys()`` equals the COUPLED-block subset.
    A mixed-mode 2-block problem (``left`` CLOSED_FORM, ``right``
    COUPLED) returns a dict with only the COUPLED block's entry,
    shaped to that block's element / IP / xi count."""

    def test_xi_solved_keys_match_coupled_subset(self) -> None:
        base_mesh = StructuredHexMesh(
            lengths=(2.0, 1.0, 1.0), divisions=(2, 1, 1),
        )
        mesh = _split_into_two_blocks(base_mesh)
        fe_problem = _build_fe_problem(
            mesh,
            {
                "left": GlobalResidualMode.CLOSED_FORM,
                "right": GlobalResidualMode.COUPLED,
            },
        )
        n_dofs = fe_problem.dof_map.num_total_dofs
        U = np.zeros(n_dofs, dtype=np.float64)
        # ``right`` has 1 element after the split.
        xi_prev_by_block: dict[str, NDArray[np.floating]] = {
            "right": np.zeros((1, 8, 6)),
        }

        _, _, xi_solved = assemble_global(
            fe_problem, U, U, t=0.0,
            xi_prev_by_block=xi_prev_by_block,
        )
        self.assertEqual(set(xi_solved.keys()), {"right"})
        self.assertEqual(xi_solved["right"].shape, (1, 8, 6))


class TestAssembleGlobalCoupledMissingXiPrev(unittest.TestCase):
    """``assemble_element_block`` raises ``ValueError`` when a
    COUPLED block's ``xi_prev_per_block`` is None — covering both
    the all-COUPLED problem with no dict, the empty-dict case, and a
    mixed-mode problem missing the COUPLED block's key."""

    def test_none_xi_prev_when_coupled(self) -> None:
        mesh = StructuredHexMesh(
            lengths=(1.0, 1.0, 1.0), divisions=(1, 1, 1),
        )
        fe_problem = _build_fe_problem(
            mesh, {"all": GlobalResidualMode.COUPLED},
        )
        n_dofs = fe_problem.dof_map.num_total_dofs
        U = np.zeros(n_dofs, dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            assemble_global(fe_problem, U, U, t=0.0)
        self.assertIn("'all'", str(ctx.exception))

    def test_empty_dict_when_coupled(self) -> None:
        mesh = StructuredHexMesh(
            lengths=(1.0, 1.0, 1.0), divisions=(1, 1, 1),
        )
        fe_problem = _build_fe_problem(
            mesh, {"all": GlobalResidualMode.COUPLED},
        )
        n_dofs = fe_problem.dof_map.num_total_dofs
        U = np.zeros(n_dofs, dtype=np.float64)
        with self.assertRaises(ValueError):
            assemble_global(
                fe_problem, U, U, t=0.0,
                xi_prev_by_block={},
            )

    def test_missing_key_in_mixed_mode(self) -> None:
        base_mesh = StructuredHexMesh(
            lengths=(2.0, 1.0, 1.0), divisions=(2, 1, 1),
        )
        mesh = _split_into_two_blocks(base_mesh)
        fe_problem = _build_fe_problem(
            mesh,
            {
                "left": GlobalResidualMode.CLOSED_FORM,
                "right": GlobalResidualMode.COUPLED,
            },
        )
        n_dofs = fe_problem.dof_map.num_total_dofs
        U = np.zeros(n_dofs, dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            assemble_global(
                fe_problem, U, U, t=0.0,
                xi_prev_by_block={},
            )
        self.assertIn("'right'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
