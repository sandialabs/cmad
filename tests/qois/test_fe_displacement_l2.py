"""Unit tests for ``FEDisplacementL2``.

Exercises the closure on hand-constructed ``U`` vectors — not FE
solutions — to verify the closure's volume integration of
:math:`|u|^2` matches the analytical answer to floating-point
tolerance. The closure does not depend on equilibrium; the QoI's
integrand is purely a function of the basis-coefficient values, so
prescribed test fields are sufficient.
"""
import unittest

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.io.registry import resolve_qoi
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.parameters.parameters import Parameters
from cmad.qois.fe_displacement_l2 import FEDisplacementL2


def _elastic_parameters(
        kappa: float = 100.0, mu: float = 50.0,
) -> Parameters:
    values = {"elastic": {"kappa": kappa, "mu": mu}}
    active = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Parameters(values, active, transforms)


def _build_unit_cube_problem() -> FEProblem:
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    dof_map = build_dof_map(
        mesh, [layout], [], components_by_field={"u": 3},
    )
    gr = SmallDispEquilibrium(ndims=3)
    model = Elastic(_elastic_parameters(), def_type=DefType.FULL_3D)
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.CLOSED_FORM},
    )


def _U_diagonal_ramp(
        mesh: StructuredHexMesh,
        u_xmax: float,
        u_ymax: float,
        u_zmax: float,
) -> np.ndarray:
    """Hand-constructed test field — not an FE solution.

    ``u(x, y, z) = (u_xmax · x, u_ymax · y, u_zmax · z)``: each
    component is a linear ramp in its own coordinate, peaking at
    ``u_*max`` on the corresponding ``x_* = 1`` face of the unit
    cube. Used purely to feed the QoI closure a known ``U``
    vector with a closed-form ``∫ |u|² dV``.
    """
    n_nodes = mesh.nodes.shape[0]
    U = np.zeros(n_nodes * 3)
    U[0::3] = u_xmax * mesh.nodes[:, 0]
    U[1::3] = u_ymax * mesh.nodes[:, 1]
    U[2::3] = u_zmax * mesh.nodes[:, 2]
    return U


class TestRegistry(unittest.TestCase):
    def test_resolve_returns_class(self) -> None:
        cls = resolve_qoi("fe_displacement_l2")
        self.assertIs(cls, FEDisplacementL2)
        self.assertEqual(cls.problem_type, "fe")


class TestClosureAnalytical(unittest.TestCase):
    def setUp(self) -> None:
        self.fe_problem = _build_unit_cube_problem()
        self.qoi = FEDisplacementL2(self.fe_problem, [0.0, 1.0])
        self.closure = self.qoi.step_contribution(
            params_by_block_from_models(self.fe_problem),
        )

    def test_zero_U_gives_zero_J(self) -> None:
        n_dofs = self.fe_problem.dof_map.num_total_dofs
        U_zero = jnp.zeros(n_dofs)
        J = float(self.closure(
            U_zero, U_zero, {}, {},
            jnp.asarray(1.0), jnp.asarray(0.0),
        ))
        self.assertEqual(J, 0.0)

    def test_diagonal_ramp_matches_analytical(self) -> None:
        # u(x,y,z) = (u_xmax·x, u_ymax·y, u_zmax·z) on [0,1]³:
        #   |u|² = u_xmax²·x² + u_ymax²·y² + u_zmax²·z²
        # Each component integrates over the unit cube to 1/3:
        #   ∫₀¹∫₀¹∫₀¹ x² dx dy dz = 1/3
        #   (and y² and z² by symmetry)
        # so   ∫_Ω |u|² dV = (u_xmax² + u_ymax² + u_zmax²) / 3.
        # Unit cube + T = 1 + dt = 1 + |Ω| = 1:
        #   J = (1/(T·|Ω|)) · dt · ∫|u|² dV
        #     = (u_xmax² + u_ymax² + u_zmax²) / 3
        u_xmax, u_ymax, u_zmax = 0.01, 0.02, -0.005
        U = jnp.asarray(_U_diagonal_ramp(
            self.fe_problem.mesh, u_xmax, u_ymax, u_zmax,
        ))
        U_prev = jnp.zeros_like(U)
        expected_J = (u_xmax ** 2 + u_ymax ** 2 + u_zmax ** 2) / 3.0
        J = float(self.closure(
            U, U_prev, {}, {},
            jnp.asarray(1.0), jnp.asarray(0.0),
        ))
        self.assertAlmostEqual(J, expected_J, places=12)

    def test_dt_scales_J_linearly(self) -> None:
        # Same field; doubling dt doubles the per-step contribution.
        # Verifies the Δt factor sits where the formula puts it.
        u_xmax, u_ymax, u_zmax = 0.01, 0.02, -0.005
        U = jnp.asarray(_U_diagonal_ramp(
            self.fe_problem.mesh, u_xmax, u_ymax, u_zmax,
        ))
        U_prev = jnp.zeros_like(U)
        J_dt_half = float(self.closure(
            U, U_prev, {}, {},
            jnp.asarray(0.5), jnp.asarray(0.0),
        ))
        J_dt_full = float(self.closure(
            U, U_prev, {}, {},
            jnp.asarray(1.0), jnp.asarray(0.0),
        ))
        self.assertAlmostEqual(J_dt_full, 2.0 * J_dt_half, places=12)


if __name__ == "__main__":
    unittest.main()
