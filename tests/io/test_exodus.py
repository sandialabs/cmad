"""Tests for :mod:`cmad.io.exodus`.

External-fixture coverage uses ``tests/io/fixtures/small_hex.exo``
(a 2x2x2 hex mesh on the unit cube, one element block, two nodesets,
no sidesets — meshio has no first-class sideset concept). Internal
round-trip tests exercise the writer + reader together on
``StructuredHexMesh`` instances that include all six built-in sidesets.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import netCDF4
import numpy as np

from cmad.fem.element_family import ElementFamily
from cmad.fem.mesh import StructuredHexMesh, hex_to_tet_split
from cmad.io.exodus import ExodusFormatError, ExodusWriter, read_mesh

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
        # No eb_names in meshio output; reader defaults to
        # f"block_{eb_prop1[i]}" — the fixture has eb_prop1 = [1].
        self.assertEqual(list(self.mesh.element_blocks), ["block_1"])
        np.testing.assert_array_equal(
            self.mesh.element_blocks["block_1"], np.arange(8)
        )
        self.assertEqual(self.mesh.element_block_ids["block_1"], 1)

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


def _assert_meshes_equal(test: unittest.TestCase, expected, actual):
    np.testing.assert_allclose(actual.nodes, expected.nodes)
    np.testing.assert_array_equal(actual.connectivity, expected.connectivity)
    test.assertEqual(actual.element_family, expected.element_family)

    test.assertEqual(set(actual.element_blocks), set(expected.element_blocks))
    for name in expected.element_blocks:
        np.testing.assert_array_equal(
            actual.element_blocks[name], expected.element_blocks[name]
        )

    test.assertEqual(set(actual.node_sets), set(expected.node_sets))
    for name in expected.node_sets:
        np.testing.assert_array_equal(
            np.sort(actual.node_sets[name]),
            np.sort(expected.node_sets[name]),
        )

    test.assertEqual(set(actual.side_sets), set(expected.side_sets))
    for name in expected.side_sets:
        np.testing.assert_array_equal(
            actual.side_sets[name], expected.side_sets[name]
        )


class TestWriterReaderRoundTrip(unittest.TestCase):

    def test_structured_hex_mesh_with_sidesets(self):
        mesh = StructuredHexMesh(
            lengths=(1.0, 2.0, 3.0),
            divisions=(2, 3, 4),
            origin=(0.5, 1.0, 1.5),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rt_hex.exo"
            with ExodusWriter(path, mesh, title="hex round-trip"):
                pass
            rt = read_mesh(path)
        _assert_meshes_equal(self, mesh, rt)
        # Sanity: 6 named sidesets survived the round trip.
        self.assertEqual(
            sorted(rt.side_sets),
            sorted([
                "xmin_sides", "xmax_sides",
                "ymin_sides", "ymax_sides",
                "zmin_sides", "zmax_sides",
            ]),
        )

    def test_tet_mesh_round_trip(self):
        hex_mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        tet_mesh = hex_to_tet_split(hex_mesh)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rt_tet.exo"
            with ExodusWriter(path, tet_mesh):
                pass
            rt = read_mesh(path)
        _assert_meshes_equal(self, tet_mesh, rt)
        self.assertEqual(rt.element_family, ElementFamily.TET_LINEAR)


class TestSetIdPreservation(unittest.TestCase):

    def test_round_trip_preserves_non_sequential_side_set_ids(self):
        # Build a mesh with cmad's writer (sequential IDs by default),
        # then mutate the file: set ss_prop1 to non-sequential values and
        # blank ss_names so the reader exercises the f"sideset_{id}"
        # default-name path. Read with read_mesh, write again, read again,
        # and confirm IDs survived the second round-trip.
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        non_seq_ids = [3, 1, 7, 4, 5, 6]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "non_seq_in.exo"
            with ExodusWriter(path, mesh):
                pass
            with netCDF4.Dataset(str(path), "r+") as ds:
                ds["ss_prop1"][:] = np.array(non_seq_ids, dtype=np.int32)
                ds["ss_names"][:] = np.zeros(
                    ds["ss_names"].shape, dtype="S1"
                )
            in_mesh = read_mesh(path)
            expected = {f"sideset_{i}": i for i in non_seq_ids}
            self.assertEqual(in_mesh.side_set_ids, expected)

            out_path = Path(tmpdir) / "non_seq_rt.exo"
            with ExodusWriter(out_path, in_mesh):
                pass
            rt = read_mesh(out_path)
        self.assertEqual(rt.side_set_ids, expected)

    def test_in_house_mesh_writer_assigns_sequential_ids_when_ids_empty(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        self.assertEqual(mesh.element_block_ids, {})
        self.assertEqual(mesh.node_set_ids, {})
        self.assertEqual(mesh.side_set_ids, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "in_house.exo"
            with ExodusWriter(path, mesh):
                pass
            rt = read_mesh(path)
        self.assertEqual(
            rt.side_set_ids,
            {
                "xmin_sides": 1,
                "xmax_sides": 2,
                "ymin_sides": 3,
                "ymax_sides": 4,
                "zmin_sides": 5,
                "zmax_sides": 6,
            },
        )


if __name__ == "__main__":
    unittest.main()
