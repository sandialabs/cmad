"""Unit tests for `cmad.fem.mesh`.

Covers `Mesh` dataclass invariants, `StructuredHexMesh` builder, and
`hex_to_tet_split` helper. Mesh sizes kept small (2x2x2) so the tests
exercise corner / edge / face combinatorics without becoming sluggish.
"""
import unittest

import numpy as np

from cmad.fem.element_family import ElementFamily
from cmad.fem.finite_element import P1_TET, Q1_HEX, EntityType, FiniteElement
from cmad.fem.interpolants import hex_linear
from cmad.fem.mesh import (
    Mesh,
    StructuredHexMesh,
    hex_to_tet_split,
)


def _build_unit_tet_mesh() -> Mesh:
    """Single linear tet at the unit-tet vertices."""
    nodes = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.intp)
    return Mesh(
        nodes=nodes,
        connectivity=connectivity,
        element_family=ElementFamily.TET_LINEAR,
        element_blocks={"all": np.array([0], dtype=np.intp)},
        node_sets={},
        side_sets={},
    )


class TestMesh(unittest.TestCase):

    def test_dataclass_round_trip(self):
        # Minimal valid hex mesh: 1 hex on the unit cube.
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        self.assertEqual(mesh.nodes.shape, (8, 3))
        self.assertEqual(mesh.connectivity.shape, (1, 8))
        self.assertEqual(mesh.element_family, ElementFamily.HEX_LINEAR)

    def test_post_init_rejects_wrong_connectivity_width(self):
        # Hex family expects 8 columns; passing 4 should raise.
        nodes = np.zeros((4, 3))
        connectivity = np.zeros((1, 4), dtype=np.intp)
        with self.assertRaisesRegex(ValueError, "nodes per element"):
            Mesh(
                nodes=nodes,
                connectivity=connectivity,
                element_family=ElementFamily.HEX_LINEAR,
                element_blocks={"all": np.zeros(1, dtype=np.intp)},
                node_sets={},
                side_sets={},
            )

    def test_post_init_rejects_element_block_partition_holes(self):
        # 2-element mesh with element_blocks covering only element 0
        # (element 1 is unassigned) — should raise.
        mesh = StructuredHexMesh((2.0, 1.0, 1.0), (2, 1, 1))
        with self.assertRaisesRegex(ValueError, "strict partition"):
            Mesh(
                nodes=mesh.nodes,
                connectivity=mesh.connectivity,
                element_family=mesh.element_family,
                element_blocks={"left": np.array([0], dtype=np.intp)},
                node_sets=mesh.node_sets,
                side_sets=mesh.side_sets,
            )


class TestStructuredHexMesh(unittest.TestCase):

    def test_node_count(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 3, 4))
        self.assertEqual(mesh.nodes.shape, (3 * 4 * 5, 3))

    def test_element_count(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 3, 4))
        self.assertEqual(mesh.connectivity.shape, (2 * 3 * 4, 8))

    def test_hex_node_ordering(self):
        # First hex of a 2x2x2 mesh on the unit cube spans (0..0.5)^3.
        # Verify the 8 corners come back in the hex_linear ordering.
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        first_hex_coords = mesh.nodes[mesh.connectivity[0]]
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
        np.testing.assert_allclose(first_hex_coords, expected)

    def test_origin_offset(self):
        mesh_at_origin = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        mesh_shifted = StructuredHexMesh(
            (1.0, 1.0, 1.0), (1, 1, 1), origin=(2.0, 3.0, 4.0)
        )
        np.testing.assert_allclose(
            mesh_shifted.nodes - mesh_at_origin.nodes,
            np.broadcast_to([2.0, 3.0, 4.0], (8, 3)),
        )

    def test_element_blocks_default_all(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        self.assertEqual(set(mesh.element_blocks.keys()), {"all"})
        np.testing.assert_array_equal(
            mesh.element_blocks["all"], np.arange(8, dtype=np.intp)
        )

    def test_node_sets_six_named_correctly(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        expected_keys = {
            "xmin_nodes", "xmax_nodes",
            "ymin_nodes", "ymax_nodes",
            "zmin_nodes", "zmax_nodes",
        }
        self.assertEqual(set(mesh.node_sets.keys()), expected_keys)
        # 2x2x2 cube: each face has 3x3 = 9 nodes.
        for name in expected_keys:
            self.assertEqual(mesh.node_sets[name].shape[0], 9)

    def test_corner_node_belongs_to_three_face_sets(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        # Corner (0, 0, 0) is mesh node 0.
        np.testing.assert_allclose(mesh.nodes[0], [0.0, 0.0, 0.0])
        in_xmin = 0 in mesh.node_sets["xmin_nodes"]
        in_ymin = 0 in mesh.node_sets["ymin_nodes"]
        in_zmin = 0 in mesh.node_sets["zmin_nodes"]
        self.assertTrue(in_xmin and in_ymin and in_zmin)

    def test_side_sets_six_named_correctly(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        expected_keys = {
            "xmin_sides", "xmax_sides",
            "ymin_sides", "ymax_sides",
            "zmin_sides", "zmax_sides",
        }
        self.assertEqual(set(mesh.side_sets.keys()), expected_keys)
        # 2x2x2 cube: each face has 2x2 = 4 boundary hex faces.
        for name in expected_keys:
            self.assertEqual(mesh.side_sets[name].shape, (4, 2))

    def test_side_set_local_face_ids_match_exodus(self):
        # Per Exodus 0-based convention: 0=-z, 1=+z, 2=-y, 3=+x, 4=+y, 5=-x.
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        face_id_for = {
            "xmin_sides": 5, "xmax_sides": 3,
            "ymin_sides": 2, "ymax_sides": 4,
            "zmin_sides": 0, "zmax_sides": 1,
        }
        for name, expected_face_id in face_id_for.items():
            face_ids = mesh.side_sets[name][:, 1]
            np.testing.assert_array_equal(
                face_ids, np.full(face_ids.shape, expected_face_id)
            )


class TestHexToTetSplit(unittest.TestCase):

    def test_element_count_six_times_hex(self):
        hex_mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        tet_mesh = hex_to_tet_split(hex_mesh)
        self.assertEqual(
            tet_mesh.connectivity.shape,
            (6 * hex_mesh.connectivity.shape[0], 4),
        )

    def test_positive_volume_on_every_tet(self):
        hex_mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        tet_mesh = hex_to_tet_split(hex_mesh)
        coords = tet_mesh.nodes[tet_mesh.connectivity]   # (n_tet, 4, 3)
        edges = coords[:, 1:, :] - coords[:, 0:1, :]     # (n_tet, 3, 3)
        # Tet volume = (1/6) det(edges); positive when tet_linear-ordered
        dets = np.linalg.det(edges)
        self.assertTrue(np.all(dets > 0))

    def test_node_sets_carry_over_unchanged(self):
        hex_mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        tet_mesh = hex_to_tet_split(hex_mesh)
        self.assertEqual(
            set(tet_mesh.node_sets.keys()),
            set(hex_mesh.node_sets.keys()),
        )
        for name in hex_mesh.node_sets:
            np.testing.assert_array_equal(
                tet_mesh.node_sets[name], hex_mesh.node_sets[name]
            )

    def test_side_sets_double_in_size(self):
        hex_mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        tet_mesh = hex_to_tet_split(hex_mesh)
        for name in hex_mesh.side_sets:
            self.assertEqual(
                tet_mesh.side_sets[name].shape[0],
                2 * hex_mesh.side_sets[name].shape[0],
            )

    def test_element_blocks_remap_to_six_per_hex(self):
        hex_mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        tet_mesh = hex_to_tet_split(hex_mesh)
        # "all" element block on hex side has 8 entries; should map to
        # 48 tet entries covering [0, 48).
        np.testing.assert_array_equal(
            np.sort(tet_mesh.element_blocks["all"]),
            np.arange(48, dtype=np.intp),
        )

    def test_element_family_flips_to_tet_linear(self):
        hex_mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        tet_mesh = hex_to_tet_split(hex_mesh)
        self.assertEqual(tet_mesh.element_family, ElementFamily.TET_LINEAR)

    def test_rejects_non_hex_input(self):
        hex_mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        tet_mesh = hex_to_tet_split(hex_mesh)
        with self.assertRaisesRegex(ValueError, "HEX_LINEAR"):
            hex_to_tet_split(tet_mesh)


class TestEdgeEnumeration(unittest.TestCase):

    def test_single_hex_has_12_edges(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        self.assertEqual(mesh.edges.shape, (12, 2))
        self.assertEqual(mesh.element_edges.shape, (1, 12))

    def test_single_tet_has_6_edges(self):
        mesh = _build_unit_tet_mesh()
        self.assertEqual(mesh.edges.shape, (6, 2))
        self.assertEqual(mesh.element_edges.shape, (1, 6))

    def test_2x2x2_hex_has_54_unique_edges(self):
        # Structured (nx,ny,nz)=(2,2,2): nx*(ny+1)*(nz+1) +
        # (nx+1)*ny*(nz+1) + (nx+1)*(ny+1)*nz = 18+18+18 = 54.
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        self.assertEqual(mesh.edges.shape, (54, 2))
        self.assertEqual(mesh.element_edges.shape, (8, 12))

    def test_adjacent_hexes_share_face_edges(self):
        # 2x1x1: two hexes share a 4-vertex face → 4 shared edges.
        # Total: 12 + 12 - 4 = 20.
        mesh = StructuredHexMesh((2.0, 1.0, 1.0), (2, 1, 1))
        self.assertEqual(mesh.edges.shape, (20, 2))

    def test_edges_are_sorted_vertex_pairs(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        self.assertTrue((mesh.edges[:, 0] < mesh.edges[:, 1]).all())

    def test_element_edges_indices_in_range(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        self.assertGreaterEqual(int(mesh.element_edges.min()), 0)
        self.assertLess(
            int(mesh.element_edges.max()), mesh.edges.shape[0],
        )

    def test_element_edge_round_trip(self):
        # Hex local edge 0 is between hex-local nodes 0 and 1.
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        elem_edge_idx = int(mesh.element_edges[0, 0])
        edge_verts = mesh.edges[elem_edge_idx]
        expected = np.sort([
            mesh.connectivity[0, 0], mesh.connectivity[0, 1],
        ])
        np.testing.assert_array_equal(edge_verts, expected)


class TestFaceEnumeration(unittest.TestCase):

    def test_single_hex_has_6_faces(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        self.assertEqual(mesh.faces.shape, (6, 4))
        self.assertEqual(mesh.element_faces.shape, (1, 6))

    def test_single_tet_has_4_faces(self):
        mesh = _build_unit_tet_mesh()
        self.assertEqual(mesh.faces.shape, (4, 3))
        self.assertEqual(mesh.element_faces.shape, (1, 4))

    def test_2x2x2_hex_has_36_unique_faces(self):
        # Structured (2,2,2): (nx+1)*ny*nz + nx*(ny+1)*nz +
        # nx*ny*(nz+1) = 12+12+12 = 36.
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        self.assertEqual(mesh.faces.shape, (36, 4))
        self.assertEqual(mesh.element_faces.shape, (8, 6))

    def test_faces_are_sorted_vertex_tuples(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        diffs = np.diff(mesh.faces, axis=1)
        self.assertTrue((diffs > 0).all())

    def test_element_faces_indices_in_range(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        self.assertGreaterEqual(int(mesh.element_faces.min()), 0)
        self.assertLess(
            int(mesh.element_faces.max()), mesh.faces.shape[0],
        )


class TestEntityCount(unittest.TestCase):

    def test_2x2x2_hex_entity_counts(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        self.assertEqual(mesh.entity_count(EntityType.VERTEX), 27)
        self.assertEqual(mesh.entity_count(EntityType.EDGE), 54)
        self.assertEqual(mesh.entity_count(EntityType.FACE), 36)
        self.assertEqual(mesh.entity_count(EntityType.CELL), 8)

    def test_single_tet_entity_counts(self):
        mesh = _build_unit_tet_mesh()
        self.assertEqual(mesh.entity_count(EntityType.VERTEX), 4)
        self.assertEqual(mesh.entity_count(EntityType.EDGE), 6)
        self.assertEqual(mesh.entity_count(EntityType.FACE), 4)
        self.assertEqual(mesh.entity_count(EntityType.CELL), 1)


class TestGeometricFiniteElement(unittest.TestCase):

    def test_hex_default_is_q1_hex(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        self.assertIs(mesh.geometric_finite_element, Q1_HEX)

    def test_tet_default_is_p1_tet(self):
        hex_mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        tet_mesh = hex_to_tet_split(hex_mesh)
        self.assertIs(tet_mesh.geometric_finite_element, P1_TET)

    def test_explicit_geometric_fe_preserved(self):
        custom = FiniteElement(
            name="my_q1_hex",
            element_family=ElementFamily.HEX_LINEAR,
            dofs_per_entity={EntityType.VERTEX: 1},
            interpolant_fn=hex_linear,
        )
        base = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        mesh = Mesh(
            nodes=base.nodes,
            connectivity=base.connectivity,
            element_family=base.element_family,
            element_blocks=base.element_blocks,
            node_sets=base.node_sets,
            side_sets=base.side_sets,
            geometric_finite_element=custom,
        )
        self.assertIs(mesh.geometric_finite_element, custom)


if __name__ == "__main__":
    unittest.main()
