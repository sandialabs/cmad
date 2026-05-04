"""Tests for :mod:`cmad.io.exodus`.

Stage A covers ``read_mesh`` against the meshio-generated fixture
``tests/io/fixtures/small_hex.exo`` (a 2x2x2 hex mesh on the unit cube
with one element block and two nodesets). The fixture has no sidesets
because meshio has no first-class sideset concept; sideset-reader
coverage rides on Stage B+ round-trip via cmad's own writer.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from cmad.fem.element_family import ElementFamily
from cmad.io.exodus import ExodusFormatError, read_mesh

_FIXTURE = Path(__file__).parent / "fixtures" / "small_hex.exo"


class TestReadMeshSmallHex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mesh = read_mesh(_FIXTURE)

    def test_node_count_and_coords(self):
        self.assertEqual(self.mesh.nodes.shape, (27, 3))
        # corner nodes of the unit cube
        np.testing.assert_allclose(self.mesh.nodes[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(self.mesh.nodes[-1], [1.0, 1.0, 1.0])

    def test_element_family_hex_linear(self):
        self.assertEqual(self.mesh.element_family, ElementFamily.HEX_LINEAR)

    def test_connectivity_shape(self):
        self.assertEqual(self.mesh.connectivity.shape, (8, 8))

    def test_first_element_node_ordering_matches_hex_linear(self):
        # cmad hex_linear: bottom CCW from (-,-,-), top CCW from (-,-,+).
        # Element 0 spans the (-,-,-) corner cell of the 2x2x2 mesh.
        expected = np.array([
            [0.0, 0.0, 0.0],   # 0: (-,-,-)
            [0.5, 0.0, 0.0],   # 1: (+,-,-)
            [0.5, 0.5, 0.0],   # 2: (+,+,-)
            [0.0, 0.5, 0.0],   # 3: (-,+,-)
            [0.0, 0.0, 0.5],   # 4: (-,-,+)
            [0.5, 0.0, 0.5],   # 5: (+,-,+)
            [0.5, 0.5, 0.5],   # 6: (+,+,+)
            [0.0, 0.5, 0.5],   # 7: (-,+,+)
        ])
        actual = self.mesh.nodes[self.mesh.connectivity[0]]
        np.testing.assert_allclose(actual, expected)

    def test_single_element_block_default_named(self):
        # meshio writes eb_prop1=[0] with no eb_names; reader defaults to "block_0"
        self.assertEqual(list(self.mesh.element_blocks), ["block_0"])
        np.testing.assert_array_equal(
            self.mesh.element_blocks["block_0"], np.arange(8)
        )

    def test_node_sets_decoded_and_zero_based(self):
        self.assertEqual(
            sorted(self.mesh.node_sets), ["xmax_nodes", "xmin_nodes"]
        )
        # x = 0 face: 9 nodes at indices 0..8 in cmad's (i, j, k) ordering
        np.testing.assert_array_equal(
            np.sort(self.mesh.node_sets["xmin_nodes"]), np.arange(9)
        )
        # x = 1 face: 9 nodes at indices 18..26
        np.testing.assert_array_equal(
            np.sort(self.mesh.node_sets["xmax_nodes"]), np.arange(18, 27)
        )

    def test_node_set_coordinates_lie_on_expected_face(self):
        xmin = self.mesh.nodes[self.mesh.node_sets["xmin_nodes"]]
        xmax = self.mesh.nodes[self.mesh.node_sets["xmax_nodes"]]
        np.testing.assert_allclose(xmin[:, 0], 0.0)
        np.testing.assert_allclose(xmax[:, 0], 1.0)

    def test_no_side_sets(self):
        self.assertEqual(self.mesh.side_sets, {})


class TestReadMeshErrors(unittest.TestCase):

    def test_missing_file_raises(self):
        with self.assertRaises((FileNotFoundError, OSError)):
            read_mesh(_FIXTURE.parent / "does_not_exist.exo")


if __name__ == "__main__":
    unittest.main()
