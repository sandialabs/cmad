"""Tests for :func:`cmad.fem.surface_integration.build_surface_integration_groups`.

Builds the per-facet surface cache on the x=1 face of a unit cube and
checks the surface area element, the side shape values, the gather
indices, and the facet partitioning. Also builds it from
``(elem_id, local_side_id)`` pairs given directly, which is how a
measurement covering part of a face is integrated over.
"""
import unittest

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.fem.surface_integration import build_surface_integration_groups
from cmad.global_residuals.mechanics import Mechanics
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.parameters.parameters import Parameters


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


class TestSurfaceIntegrationGroups(unittest.TestCase):
    def setUp(self) -> None:
        self.fe = _unit_cube_problem(2)
        self.groups = build_surface_integration_groups(
            self.fe.mesh, self.fe.dof_map, "u", "xmax_sides",
            self.fe.side_quadrature,
        )

    def test_area_sums_to_face_area(self) -> None:
        area = sum(
            float(jnp.sum(g.dA * g.side_w[None, :])) for g in self.groups
        )
        self.assertAlmostEqual(area, 1.0, places=12)

    def test_side_shapes_are_partition_of_unity(self) -> None:
        for g in self.groups:
            row_sums = np.asarray(g.N_side.sum(axis=1))
            self.assertTrue(np.allclose(row_sums, 1.0, rtol=0, atol=1e-12))

    def test_gather_indices_lie_on_the_face(self) -> None:
        # field "u" has block_offset 0 and 3 components, so the global eq
        # of (node, component) is node*3 + component; every gathered node
        # must sit on the x=1 face.
        nodes = self.fe.mesh.nodes
        for g in self.groups:
            eq = np.asarray(g.eq)
            self.assertEqual(eq.shape[-1], 3)
            gathered = eq[..., 0] // 3
            self.assertTrue(np.allclose(nodes[gathered][..., 0], 1.0))

    def test_facet_count_matches_sideset(self) -> None:
        n_facets = sum(g.dA.shape[0] for g in self.groups)
        self.assertEqual(n_facets, 4)

    def test_unknown_sideset_raises(self) -> None:
        with self.assertRaises(KeyError):
            build_surface_integration_groups(
                self.fe.mesh, self.fe.dof_map, "u", "no_such_sideset",
                self.fe.side_quadrature,
            )


class TestGivenSidePairs(unittest.TestCase):
    """Pairs given directly, rather than resolved from a sideset name."""

    def setUp(self) -> None:
        self.fe = _unit_cube_problem(2)

    def _groups(self, sides):
        return build_surface_integration_groups(
            self.fe.mesh, self.fe.dof_map, "u", sides,
            self.fe.side_quadrature,
        )

    def test_the_sideset_pairs_reproduce_the_sideset(self) -> None:
        by_name = self._groups("xmax_sides")
        by_pairs = self._groups(self.fe.mesh.side_sets["xmax_sides"])
        self.assertEqual(len(by_pairs), len(by_name))
        for named, given in zip(by_name, by_pairs, strict=True):
            np.testing.assert_allclose(named.dA, given.dA, rtol=0, atol=0)
            np.testing.assert_array_equal(named.eq, given.eq)

    def test_pairs_across_two_faces_partition_by_local_side(self) -> None:
        # one facet from each of two faces, which no single sideset holds
        groups = self._groups(np.vstack([
            self.fe.mesh.side_sets["xmax_sides"][:1],
            self.fe.mesh.side_sets["ymax_sides"][:1],
        ]))
        self.assertEqual(len(groups), 2)
        area = sum(float(jnp.sum(g.dA * g.side_w[None, :])) for g in groups)
        self.assertAlmostEqual(area, 2 * 0.25, places=12)

    def test_wrong_shape_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"shape \(n, 2\)"):
            self._groups(np.array([0, 1, 2], dtype=np.intp))

    def test_no_sides_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no sides"):
            self._groups(np.empty((0, 2), dtype=np.intp))


if __name__ == "__main__":
    unittest.main()
