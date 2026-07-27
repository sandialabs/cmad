"""Tests for :func:`cmad.fem.dof.dof_physical_coords`.

Checks the reference coordinates and global equation numbers returned
for a field's basis coefficients on a sideset and over the whole field,
on a 2x2x2 unit cube.
"""
import unittest

import numpy as np

from cmad.fem.dof import GlobalFieldLayout, build_dof_map, dof_physical_coords
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh


class TestDofPhysicalCoords(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        self.dof_map = build_dof_map(
            self.mesh, [layout], [], components_by_field={"u": 3},
        )

    def test_sideset_coords_lie_on_face(self) -> None:
        coords, eq = dof_physical_coords(
            self.mesh, self.dof_map, "u", "xmax_sides",
        )
        # 3x3 vertices on the x=1 face of a 2x2x2 cube.
        self.assertEqual(coords.shape, (9, 3))
        self.assertEqual(eq.shape, (9, 3))
        self.assertTrue(np.allclose(coords[:, 0], 1.0))

    def test_eq_matches_formula(self) -> None:
        coords, eq = dof_physical_coords(
            self.mesh, self.dof_map, "u", "xmax_sides",
        )
        node = eq[:, 0] // 3
        for c in range(3):
            self.assertTrue(np.array_equal(eq[:, c], node * 3 + c))
        self.assertTrue(np.allclose(coords, self.mesh.nodes[node]))

    def test_whole_field_covers_all_nodes(self) -> None:
        coords, eq = dof_physical_coords(self.mesh, self.dof_map, "u")
        n_nodes = self.mesh.nodes.shape[0]
        self.assertEqual(coords.shape, (n_nodes, 3))
        self.assertEqual(eq.shape, (n_nodes, 3))

    def test_unknown_sideset_raises(self) -> None:
        with self.assertRaises(KeyError):
            dof_physical_coords(self.mesh, self.dof_map, "u", "nope")

    def test_unknown_field_raises(self) -> None:
        with self.assertRaises(ValueError):
            dof_physical_coords(self.mesh, self.dof_map, "T", "xmax_sides")


if __name__ == "__main__":
    unittest.main()
