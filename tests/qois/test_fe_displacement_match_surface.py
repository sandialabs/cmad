"""Surface (sideset) option of ``FEDisplacementMatch``.

Hand-constructed ``U`` vectors -- not FE solutions -- feed the closure a
known displacement on the x=1 face of a unit cube, where the surface
integral of ``|u - u_data|^2`` has a closed form. A linear field is Q1
exact on the flat face and the quadratic integrand is integrated exactly
by the degree-2 side quadrature, so the match is to floating-point
tolerance.

The same closed form is checked on a mixed u-p problem, where the
measurement covers ``u`` alone: it must reach that field's equations and
the pressure must stay out of the mismatch.
"""
import unittest

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.global_residuals.mechanics import Mechanics
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.global_fields import StepTime
from cmad.parameters.parameters import Parameters
from cmad.qois.fe_displacement_match import FEDisplacementMatch


def _elastic_parameters(kappa: float = 100.0, mu: float = 50.0) -> Parameters:
    values = {"elastic": {"kappa": kappa, "mu": mu}}
    active = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Parameters(values, active, transforms)


def _unit_cube_problem(n: int = 2):
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (n, n, n))
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    dof_map = build_dof_map(mesh, [layout], [], components_by_field={"u": 3})
    gr = Mechanics(ndims=3)
    model = Elastic(_elastic_parameters(), def_type=DefType.FULL_3D)
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.CLOSED_FORM},
    )


def _unit_cube_mixed_problem(n: int = 2):
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (n, n, n))
    dof_map = build_dof_map(
        mesh,
        # pressure first, so 'u' does not start at equation zero and a
        # measurement written to the head of the vector would be misplaced
        [GlobalFieldLayout(name="p", finite_element=Q1_HEX),
         GlobalFieldLayout(name="u", finite_element=Q1_HEX)],
        [], components_by_field={"u": 3, "p": 1},
    )
    model = Elastic(_elastic_parameters(), def_type=DefType.FULL_3D)
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=Mechanics(ndims=3, mixed=True),
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.CLOSED_FORM},
    )


def _U_diagonal_ramp(mesh, a: float, b: float, c: float) -> np.ndarray:
    """``u(x, y, z) = (a x, b y, c z)`` as a flat nodal vector."""
    n_nodes = mesh.nodes.shape[0]
    U = np.zeros(n_nodes * 3)
    U[0::3] = a * mesh.nodes[:, 0]
    U[1::3] = b * mesh.nodes[:, 1]
    U[2::3] = c * mesh.nodes[:, 2]
    return U


class TestSurfaceMatch(unittest.TestCase):
    def setUp(self) -> None:
        self.fe = _unit_cube_problem(2)
        self.n_dofs = self.fe.dof_map.num_total_dofs
        self.data_shape = (2, self.fe.mesh.nodes.shape[0], 3)

    def _eval(self, U, data) -> float:
        qoi = FEDisplacementMatch(
            self.fe, [0.0, 1.0], jnp.asarray(data), sideset="xmax_sides",
        )
        closure = qoi.step_contribution(
            params_by_block_from_models(self.fe), self.fe.kernel_arrays,
        )
        return float(closure(
            jnp.asarray(U), jnp.zeros(self.n_dofs), {}, {},
            StepTime(jnp.asarray(1.0), jnp.asarray(0.0)),
        ))

    def test_surface_integral_matches_analytical(self) -> None:
        # On the x=1 face, u = (a, b y, c z) so |u|^2 = a^2 + b^2 y^2 + c^2 z^2.
        # Over the unit face, int |u|^2 dA = a^2 + (b^2 + c^2)/3. With
        # weight=1, T=1, dt=1, and face area 1, J equals that integral.
        a, b, c = 0.01, 0.02, -0.005
        U = _U_diagonal_ramp(self.fe.mesh, a, b, c)
        data = np.zeros(self.data_shape)
        expected = a ** 2 + (b ** 2 + c ** 2) / 3.0
        self.assertAlmostEqual(self._eval(U, data), expected, places=12)

    def test_matching_field_gives_zero(self) -> None:
        U = _U_diagonal_ramp(self.fe.mesh, 0.01, 0.02, -0.005)
        data = np.broadcast_to(U.reshape(-1, 3), self.data_shape)
        self.assertAlmostEqual(self._eval(U, data), 0.0, places=12)

    def test_only_the_sideset_contributes(self) -> None:
        # Perturbing U away from the x=1 face leaves the surface J unchanged.
        a, b, c = 0.01, 0.02, -0.005
        U = _U_diagonal_ramp(self.fe.mesh, a, b, c)
        data = np.zeros(self.data_shape)
        baseline = self._eval(U, data)

        off_face = self.fe.mesh.nodes[:, 0] < 1.0
        U_perturbed = U.copy()
        U_perturbed.reshape(-1, 3)[off_face] += 1.0
        self.assertAlmostEqual(self._eval(U_perturbed, data), baseline,
                               places=12)


class TestSurfaceMatchMixed(unittest.TestCase):
    """A pressure field alongside 'u', which the measurement never covers."""

    def setUp(self) -> None:
        self.fe = _unit_cube_mixed_problem(2)
        self.n_dofs = self.fe.dof_map.num_total_dofs
        self.nodes = self.fe.mesh.nodes
        self.data_shape = (2, self.nodes.shape[0], 3)
        # field-major, and this problem lists the pressure first
        self.pressure = 7.0 + self.nodes[:, 0]

    def _eval(self, u, data) -> float:
        U = np.concatenate([self.pressure, u])
        self.assertEqual(U.size, self.n_dofs)
        qoi = FEDisplacementMatch(
            self.fe, [0.0, 1.0], jnp.asarray(data), sideset="xmax_sides",
        )
        closure = qoi.step_contribution(
            params_by_block_from_models(self.fe), self.fe.kernel_arrays,
        )
        return float(closure(
            jnp.asarray(U), jnp.zeros(self.n_dofs), {}, {},
            StepTime(jnp.asarray(1.0), jnp.asarray(0.0)),
        ))

    def test_matching_field_gives_zero(self) -> None:
        # nonzero data, so it scores the full integral if it is written
        # anywhere other than the displacement equations
        u = _U_diagonal_ramp(self.fe.mesh, 0.01, 0.02, -0.005)
        data = np.broadcast_to(u.reshape(-1, 3), self.data_shape)
        self.assertAlmostEqual(self._eval(u, data), 0.0, places=12)

    def test_surface_integral_matches_analytical(self) -> None:
        a, b, c = 0.01, 0.02, -0.005
        u = _U_diagonal_ramp(self.fe.mesh, a, b, c)
        expected = a ** 2 + (b ** 2 + c ** 2) / 3.0
        self.assertAlmostEqual(
            self._eval(u, np.zeros(self.data_shape)), expected, places=12)


if __name__ == "__main__":
    unittest.main()
