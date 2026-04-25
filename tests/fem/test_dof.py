"""Unit tests for `cmad.fem.dof`.

Covers `GlobalFieldLayout` validation, `GlobalDofMap` formula round-
trips and counts, multi-field layouts, the basis_fn_to_vertex identity
and explicit paths, time-dependent value evaluation, and `build_dof_map`
error paths.
"""
import unittest

import numpy as np

from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.mesh import StructuredHexMesh


def _unit_cube_2x2x2():
    """Shared fixture: 27-node 2x2x2 hex mesh on the unit cube."""
    return StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))


class TestGlobalFieldLayoutValidation(unittest.TestCase):

    def test_rejects_zero_basis_fns(self):
        with self.assertRaisesRegex(ValueError, "num_basis_fns"):
            GlobalFieldLayout(name="u", num_basis_fns=0,
                              num_dofs_per_basis_fn=3)

    def test_rejects_zero_ndofs_per_basis_fn(self):
        with self.assertRaisesRegex(ValueError, "num_dofs_per_basis_fn"):
            GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=0)

    def test_rejects_basis_fn_to_vertex_length_mismatch(self):
        with self.assertRaisesRegex(
            ValueError, "basis_fn_to_vertex length"
        ):
            GlobalFieldLayout(
                name="u", num_basis_fns=4, num_dofs_per_basis_fn=1,
                basis_fn_to_vertex=np.array([0, 1, 2], dtype=np.intp),
            )

    def test_rejects_basis_fn_to_vertex_duplicates(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            GlobalFieldLayout(
                name="u", num_basis_fns=3, num_dofs_per_basis_fn=1,
                basis_fn_to_vertex=np.array([0, 1, 1], dtype=np.intp),
            )


class TestGlobalDofMapFormula(unittest.TestCase):

    def test_eq_index_single_field(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        dm = build_dof_map(mesh, [u], [])
        # Field 0, basis_fn 5, component 2: 0 + 5*3 + 2 = 17
        self.assertEqual(dm.eq_index(0, 5, 2), 17)
        # Last basis_fn, last component: 0 + 26*3 + 2 = 80
        self.assertEqual(dm.eq_index(0, 26, 2), 80)

    def test_eq_index_multi_field(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        p = GlobalFieldLayout(name="p", num_basis_fns=27,
                              num_dofs_per_basis_fn=1)
        dm = build_dof_map(mesh, [u, p], [])
        # Field 0 (u) occupies [0, 81); field 1 (p) occupies [81, 108).
        self.assertEqual(dm.eq_index(0, 0, 0), 0)
        self.assertEqual(dm.eq_index(1, 0, 0), 81)
        self.assertEqual(dm.eq_index(1, 26, 0), 107)

    def test_total_dof_count(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        p = GlobalFieldLayout(name="p", num_basis_fns=27,
                              num_dofs_per_basis_fn=1)
        dm = build_dof_map(mesh, [u, p], [])
        self.assertEqual(dm.num_total_dofs, 81 + 27)

    def test_block_offsets_cumulative(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        p = GlobalFieldLayout(name="p", num_basis_fns=27,
                              num_dofs_per_basis_fn=1)
        T = GlobalFieldLayout(name="T", num_basis_fns=27,
                              num_dofs_per_basis_fn=1)
        dm = build_dof_map(mesh, [u, p, T], [])
        np.testing.assert_array_equal(
            dm.block_offsets, np.array([0, 81, 108, 135], dtype=np.intp)
        )


class TestGlobalDofMapPartition(unittest.TestCase):

    def test_free_plus_prescribed_equals_total(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        # Disjoint per-face dof clamp — no double-prescription.
        bcs = [
            DirichletBC(nodeset_name="xmin_nodes", field_name="u",
                        dofs=[0]),
            DirichletBC(nodeset_name="ymin_nodes", field_name="u",
                        dofs=[1]),
            DirichletBC(nodeset_name="zmin_nodes", field_name="u",
                        dofs=[2]),
        ]
        dm = build_dof_map(mesh, [u], bcs)
        self.assertEqual(
            dm.num_free_dofs + dm.num_prescribed_dofs, dm.num_total_dofs
        )

    def test_field_with_no_bcs_has_zero_prescribed(self):
        # Multi-field; BC on u only. p contributes no prescribed entries.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        p = GlobalFieldLayout(name="p", num_basis_fns=27,
                              num_dofs_per_basis_fn=1)
        bcs = [
            DirichletBC(nodeset_name="xmin_nodes", field_name="u",
                        dofs=[0]),
        ]
        dm = build_dof_map(mesh, [u, p], bcs)
        # All prescribed eqs lie inside u's range [0, 81).
        self.assertTrue(np.all(dm.prescribed_indices < 81))


class TestBasisFnToVertex(unittest.TestCase):

    def test_identity_mapping_single_field_clamp(self):
        # No basis_fn_to_vertex (None). 27-node mesh, 27-basis-fn field.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        bcs = [
            DirichletBC(nodeset_name="xmin_nodes", field_name="u",
                        dofs=[0]),
        ]
        dm = build_dof_map(mesh, [u], bcs)
        # xmin_nodes has 9 nodes, dofs=[0] -> 9 prescribed eqs.
        # Each at eq = node_idx * 3 + 0 (block_offset = 0).
        expected = mesh.node_sets["xmin_nodes"] * 3
        np.testing.assert_array_equal(
            np.sort(dm.prescribed_indices), np.sort(expected)
        )

    def test_identity_requires_num_basis_fns_eq_n_nodes(self):
        # 27-node mesh but layout claims 26 basis fns: identity invalid.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=26,
                              num_dofs_per_basis_fn=3)
        with self.assertRaisesRegex(
            ValueError, "identity basis_fn_to_vertex requires"
        ):
            build_dof_map(mesh, [u], [])

    def test_explicit_mapping_subset_of_vertices(self):
        # Synthetic Taylor-Hood-shaped fixture: a "p" field with 8
        # basis fns living only at the corner vertices of a 2x2x2 hex
        # mesh (corner indices 0, 2, 6, 8, 18, 20, 24, 26 in node-major
        # ordering). Verify a BC on a corner-only nodeset resolves
        # through the explicit mapping.
        mesh = _unit_cube_2x2x2()
        # Explicitly compute the 8 mesh-corner indices on the 27-node
        # cube with x slowest, z fastest.
        corners = np.array([
            0,  2,  6,  8,
            18, 20, 24, 26,
        ], dtype=np.intp)
        # Sanity: those mesh nodes are at the 8 unit-cube corners.
        for c, expected_xyz in zip(corners, [
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ], strict=True):
            np.testing.assert_allclose(mesh.nodes[c], expected_xyz)
        # Field "p" with 8 basis fns mapped to those 8 corner vertices.
        p = GlobalFieldLayout(
            name="p", num_basis_fns=8, num_dofs_per_basis_fn=1,
            basis_fn_to_vertex=corners,
        )
        # Custom corner-only nodeset for the BC: only the 4 corners on
        # x=0 (mesh nodes 0, 2, 6, 8 — basis fns 0, 1, 2, 3 of p).
        # Mutate node_sets to add a corner-only set; mesh is frozen but
        # the dict isn't deep-frozen, so we wrap the existing dict.
        node_sets_aug = dict(mesh.node_sets)
        node_sets_aug["xmin_corners"] = np.array([0, 2, 6, 8],
                                                dtype=np.intp)
        from cmad.fem.mesh import Mesh
        mesh_aug = Mesh(
            nodes=mesh.nodes, connectivity=mesh.connectivity,
            element_family=mesh.element_family,
            element_blocks=mesh.element_blocks,
            node_sets=node_sets_aug, side_sets=mesh.side_sets,
        )
        bcs = [
            DirichletBC(nodeset_name="xmin_corners", field_name="p",
                        dofs=[0]),
        ]
        dm = build_dof_map(mesh_aug, [p], bcs)
        # 4 prescribed eqs, at basis-fn indices 0, 1, 2, 3
        # (block_offset = 0, ndofs = 1, dof = 0).
        np.testing.assert_array_equal(
            np.sort(dm.prescribed_indices),
            np.array([0, 1, 2, 3], dtype=np.intp),
        )


class TestEvaluatePrescribedValues(unittest.TestCase):

    def _setup_clamped_xmin(self, values):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        bcs = [
            DirichletBC(nodeset_name="xmin_nodes", field_name="u",
                        dofs=[0, 1, 2], values=values),
        ]
        return build_dof_map(mesh, [u], bcs)

    def test_homogeneous_zeros(self):
        dm = self._setup_clamped_xmin(values=None)
        np.testing.assert_array_equal(
            dm.evaluate_prescribed_values(),
            np.zeros(9 * 3, dtype=np.float64),
        )

    def test_constant_sequence_broadcasts(self):
        dm = self._setup_clamped_xmin(values=[0.5, 1.0, 1.5])
        vals = dm.evaluate_prescribed_values()
        # Reshape back to (9 nodes, 3 components) — every row should be
        # [0.5, 1.0, 1.5].
        np.testing.assert_allclose(
            vals.reshape(9, 3),
            np.broadcast_to([0.5, 1.0, 1.5], (9, 3)),
        )

    def test_callable_spatially_varying(self):
        # u_x = 0.5 * x + y on xmin face. Other components zero.
        def vals_fn(coords, t):
            x, y = coords[:, 0], coords[:, 1]
            n = coords.shape[0]
            out = np.zeros((n, 3))
            out[:, 0] = 0.5 * x + y
            return out
        dm = self._setup_clamped_xmin(values=vals_fn)
        vals = dm.evaluate_prescribed_values(t=0.0)
        # xmin face has x=0, so u_x = 0 + y = y on that face.
        # Per-node values aren't sorted; verify via reshape + element-wise.
        coords_xmin = _unit_cube_2x2x2().nodes[
            _unit_cube_2x2x2().node_sets["xmin_nodes"]
        ]
        expected = np.zeros((9, 3))
        expected[:, 0] = 0.5 * coords_xmin[:, 0] + coords_xmin[:, 1]
        np.testing.assert_allclose(vals.reshape(9, 3), expected)

    def test_callable_time_dependent(self):
        # u_x = t * x on xmax face (where x = 1 always).
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        def vals_fn(coords, t):
            n = coords.shape[0]
            out = np.zeros((n, 1))
            out[:, 0] = t * coords[:, 0]
            return out
        bcs = [
            DirichletBC(nodeset_name="xmax_nodes", field_name="u",
                        dofs=[0], values=vals_fn),
        ]
        dm = build_dof_map(mesh, [u], bcs)
        v0 = dm.evaluate_prescribed_values(t=0.0)
        v3 = dm.evaluate_prescribed_values(t=3.0)
        np.testing.assert_allclose(v0, np.zeros_like(v0))
        np.testing.assert_allclose(v3, np.full_like(v3, 3.0))


class TestBuildDofMapErrors(unittest.TestCase):

    def test_double_prescription_raises(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        bcs = [
            DirichletBC(nodeset_name="xmin_nodes", field_name="u",
                        dofs=[0]),
            DirichletBC(nodeset_name="xmin_nodes", field_name="u",
                        dofs=[0]),
        ]
        with self.assertRaisesRegex(ValueError, "double-prescribed"):
            build_dof_map(mesh, [u], bcs)

    def test_unknown_field_name_raises(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        bcs = [
            DirichletBC(nodeset_name="xmin_nodes", field_name="temperature",
                        dofs=[0]),
        ]
        with self.assertRaisesRegex(ValueError, "unknown field"):
            build_dof_map(mesh, [u], bcs)

    def test_unknown_nodeset_name_raises(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", num_basis_fns=27,
                              num_dofs_per_basis_fn=3)
        bcs = [
            DirichletBC(nodeset_name="not_a_real_nodeset", field_name="u",
                        dofs=[0]),
        ]
        with self.assertRaisesRegex(KeyError, "unknown nodeset"):
            build_dof_map(mesh, [u], bcs)

    def test_dof_index_out_of_range_raises(self):
        mesh = _unit_cube_2x2x2()
        # Scalar field; only dof 0 is valid.
        T = GlobalFieldLayout(name="T", num_basis_fns=27,
                              num_dofs_per_basis_fn=1)
        bcs = [
            DirichletBC(nodeset_name="xmin_nodes", field_name="T",
                        dofs=[2]),
        ]
        with self.assertRaisesRegex(ValueError, "dof 2 outside"):
            build_dof_map(mesh, [T], bcs)

    def test_duplicate_field_names_raises(self):
        mesh = _unit_cube_2x2x2()
        u1 = GlobalFieldLayout(name="u", num_basis_fns=27,
                               num_dofs_per_basis_fn=3)
        u2 = GlobalFieldLayout(name="u", num_basis_fns=27,
                               num_dofs_per_basis_fn=1)
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            build_dof_map(mesh, [u1, u2], [])


if __name__ == "__main__":
    unittest.main()
