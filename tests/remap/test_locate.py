"""Tests for point location and FE-field sampling on a sideset.

Three things are checked. (1) The quad inverse-bilinear recovers a known
parametric coordinate: forward-mapping a chosen ``(xi, eta)`` through a
non-parallelogram (trapezoidal) facet gives a physical point, and the
locator must return the bilinear weights at that ``(xi, eta)`` -- this
exercises the general quadratic branch, which a structured (rectangular)
mesh never reaches. (2) Through the public locator on real meshes, a quad
sideset reproduces a bilinear field exactly (Q1) and a tri sideset
reproduces a linear field exactly (P1). (3) ``sample_fe_displacement_cloud``
assembles a per-step cloud of the sampled displacement.
"""
import unittest

import numpy as np
from jax.tree_util import tree_map

from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEState, build_fe_problem
from cmad.fem.finite_element import P1_TET, Q1_HEX
from cmad.fem.mesh import StructuredHexMesh, hex_to_tet_split
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.parameters.parameters import Parameters
from cmad.remap.locate import (
    _inverse_bilinear,
    locate_points_in_sideset,
    sample_fe_displacement_cloud,
)


def _elastic_parameters() -> Parameters:
    values = {"elastic": {"kappa": 100.0, "mu": 50.0}}
    active = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Parameters(values, active, transforms)


def _problem(mesh, finite_element):
    layout = GlobalFieldLayout(name="u", finite_element=finite_element)
    dof_map = build_dof_map(mesh, [layout], [], components_by_field={"u": 3})
    gr = SmallDispEquilibrium(ndims=3)
    model = Elastic(_elastic_parameters(), def_type=DefType.FULL_3D)
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.CLOSED_FORM},
    )


def _hex_problem(n: int = 2):
    return _problem(StructuredHexMesh((1.0, 1.0, 1.0), (n, n, n)), Q1_HEX)


def _tet_problem(n: int = 2):
    mesh = hex_to_tet_split(StructuredHexMesh((1.0, 1.0, 1.0), (n, n, n)))
    return _problem(mesh, P1_TET)


def _linear_field(coords: np.ndarray) -> np.ndarray:
    """A field linear in space (exactly reproduced by P1 and Q1)."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    return np.stack([
        0.10 + 0.20 * x + 0.30 * y - 0.15 * z,
        -0.05 + 0.25 * y + 0.10 * z,
        0.02 + 0.30 * z - 0.20 * x,
    ], axis=1)


def _bilinear_field(coords: np.ndarray) -> np.ndarray:
    """A field with ``y*z`` cross terms: bilinear on the x = 1 face.

    Reproduced exactly by Q1 (the cross term is in the bilinear span) but
    not by P1, so it distinguishes the quad path from the tri path.
    """
    y, z = coords[:, 1], coords[:, 2]
    return np.stack([
        0.10 + 0.20 * y + 0.30 * z + 0.40 * y * z,
        -0.05 + 0.15 * y - 0.10 * z + 0.50 * y * z,
        0.03 * z + 0.02 * y * z,
    ], axis=1)


def _face_points(n_points: int = 40, seed: int = 0) -> np.ndarray:
    """Scattered points on the x = 1 face."""
    rng = np.random.default_rng(seed)
    yz = rng.uniform(0.0, 1.0, (n_points, 2))
    return np.column_stack([np.ones(n_points), yz])


class TestInverseBilinear(unittest.TestCase):
    def test_recovers_known_parametric_coordinate(self) -> None:
        # A planar (x = 1) but non-parallelogram quad, so the bilinear map
        # is genuinely non-affine (C != 0) and the quadratic branch runs.
        quad = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.1],
            [1.0, 0.8, 1.0],
            [1.0, 0.2, 1.2],
        ])
        nodal = np.array([0.0, 1.0, 3.0, 2.0])
        for xi, eta in [(0.3, 0.4), (0.7, 0.2), (0.1, 0.9), (0.5, 0.5)]:
            shape = np.array([
                (1.0 - xi) * (1.0 - eta), xi * (1.0 - eta),
                xi * eta, (1.0 - xi) * eta,
            ])
            point = shape @ quad
            weights = _inverse_bilinear(point, quad)
            np.testing.assert_allclose(weights, shape, atol=1e-10)
            self.assertAlmostEqual(
                float(weights @ nodal), float(shape @ nodal), places=10,
            )


class TestLocateOnMesh(unittest.TestCase):
    def _interpolate(self, fe, points, field):
        nodes, weights = locate_points_in_sideset(
            fe.mesh, fe.dof_map, "u", "xmax_sides", points,
        )
        nodal = field(fe.mesh.nodes)
        return np.einsum("pn,pnc->pc", weights, nodal[nodes])

    def test_quad_reproduces_bilinear_field(self) -> None:
        fe = _hex_problem(2)
        points = _face_points(40)
        got = self._interpolate(fe, points, _bilinear_field)
        np.testing.assert_allclose(got, _bilinear_field(points), atol=1e-12)

    def test_tri_reproduces_linear_field(self) -> None:
        fe = _tet_problem(2)
        points = _face_points(40, seed=1)
        got = self._interpolate(fe, points, _linear_field)
        np.testing.assert_allclose(got, _linear_field(points), atol=1e-12)

    def test_quad_weights_are_a_partition_of_unity(self) -> None:
        fe = _hex_problem(3)
        _, weights = locate_points_in_sideset(
            fe.mesh, fe.dof_map, "u", "xmax_sides", _face_points(50, seed=2),
        )
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)

    def test_tri_weights_are_a_partition_of_unity(self) -> None:
        fe = _tet_problem(2)
        _, weights = locate_points_in_sideset(
            fe.mesh, fe.dof_map, "u", "xmax_sides", _face_points(50, seed=3),
        )
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)

    def test_unknown_sideset_raises(self) -> None:
        fe = _hex_problem(2)
        with self.assertRaises(KeyError):
            locate_points_in_sideset(
                fe.mesh, fe.dof_map, "u", "nope_sides", _face_points(5),
            )


class TestSampleDisplacementCloud(unittest.TestCase):
    def test_cloud_matches_field_per_step(self) -> None:
        fe = _hex_problem(2)
        points = _face_points(30, seed=4)
        nodal = _bilinear_field(fe.mesh.nodes).ravel()
        state = FEState(
            U_history=[np.zeros(fe.dof_map.num_total_dofs), nodal],
            xi_history_by_block={},
            t_history=[0.0, 1.0],
        )
        cloud = sample_fe_displacement_cloud(fe, state, points, "xmax_sides")

        self.assertEqual(cloud.fields["displacement"].shape, (2, 30, 3))
        np.testing.assert_allclose(cloud.coords, points)
        np.testing.assert_allclose(cloud.times, [0.0, 1.0])
        np.testing.assert_allclose(
            cloud.fields["displacement"][0], 0.0, atol=1e-14,
        )
        np.testing.assert_allclose(
            cloud.fields["displacement"][1], _bilinear_field(points),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
