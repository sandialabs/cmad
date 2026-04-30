"""Unit tests for `cmad.fem.dof`.

Covers `GlobalFieldLayout` validation, `GlobalDofMap` formula round-
trips and counts, multi-field layouts, FE-family / mesh-family
agreement, vertex-anchored BC resolution, time-dependent value
evaluation, and `build_dof_map` error paths.
"""
import unittest

import numpy as np

from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.element_family import ElementFamily
from cmad.fem.finite_element import (
    P1_TET,
    Q1_HEX,
    EntityType,
    FiniteElement,
)
from cmad.fem.interpolants import hex_linear
from cmad.fem.mesh import StructuredHexMesh


def _unit_cube_2x2x2():
    """Shared fixture: 27-node 2x2x2 hex mesh on the unit cube."""
    return StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))


class TestGlobalFieldLayoutValidation(unittest.TestCase):

    def test_rejects_zero_ndofs_per_basis_fn(self):
        with self.assertRaisesRegex(ValueError, "num_dofs_per_basis_fn"):
            GlobalFieldLayout(
                name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=0,
            )


class TestGlobalDofMapFormula(unittest.TestCase):

    def test_eq_index_single_field(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
        dm = build_dof_map(mesh, [u], [])
        # Field 0, basis_fn 5, component 2: 0 + 5*3 + 2 = 17
        self.assertEqual(dm.eq_index(0, 5, 2), 17)
        # Last basis_fn, last component: 0 + 26*3 + 2 = 80
        self.assertEqual(dm.eq_index(0, 26, 2), 80)

    def test_eq_index_multi_field(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
        p = GlobalFieldLayout(
            name="p", finite_element=Q1_HEX, num_dofs_per_basis_fn=1,
        )
        dm = build_dof_map(mesh, [u, p], [])
        # Field 0 (u) occupies [0, 81); field 1 (p) occupies [81, 108).
        self.assertEqual(dm.eq_index(0, 0, 0), 0)
        self.assertEqual(dm.eq_index(1, 0, 0), 81)
        self.assertEqual(dm.eq_index(1, 26, 0), 107)

    def test_total_dof_count(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
        p = GlobalFieldLayout(
            name="p", finite_element=Q1_HEX, num_dofs_per_basis_fn=1,
        )
        dm = build_dof_map(mesh, [u, p], [])
        self.assertEqual(dm.num_total_dofs, 81 + 27)

    def test_block_offsets_cumulative(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
        p = GlobalFieldLayout(
            name="p", finite_element=Q1_HEX, num_dofs_per_basis_fn=1,
        )
        T = GlobalFieldLayout(
            name="T", finite_element=Q1_HEX, num_dofs_per_basis_fn=1,
        )
        dm = build_dof_map(mesh, [u, p, T], [])
        np.testing.assert_array_equal(
            dm.block_offsets, np.array([0, 81, 108, 135], dtype=np.intp)
        )


class TestGlobalDofMapPartition(unittest.TestCase):

    def test_free_plus_prescribed_equals_total(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
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
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
        p = GlobalFieldLayout(
            name="p", finite_element=Q1_HEX, num_dofs_per_basis_fn=1,
        )
        bcs = [
            DirichletBC(nodeset_name="xmin_nodes", field_name="u",
                        dofs=[0]),
        ]
        dm = build_dof_map(mesh, [u, p], bcs)
        # All prescribed eqs lie inside u's range [0, 81).
        self.assertTrue(np.all(dm.prescribed_indices < 81))


class TestVertexBcResolution(unittest.TestCase):

    def test_xmin_clamp_resolves_to_correct_eqs(self):
        # Q1_HEX has dofs_per_entity[VERTEX]=1: identity basis_fn[v]=v.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
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


class TestEvaluatePrescribedValues(unittest.TestCase):

    def _setup_clamped_xmin(self, values):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
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
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
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
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
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
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
        bcs = [
            DirichletBC(nodeset_name="xmin_nodes", field_name="temperature",
                        dofs=[0]),
        ]
        with self.assertRaisesRegex(ValueError, "unknown field"):
            build_dof_map(mesh, [u], bcs)

    def test_unknown_nodeset_name_raises(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
        bcs = [
            DirichletBC(nodeset_name="not_a_real_nodeset", field_name="u",
                        dofs=[0]),
        ]
        with self.assertRaisesRegex(KeyError, "unknown nodeset"):
            build_dof_map(mesh, [u], bcs)

    def test_dof_index_out_of_range_raises(self):
        mesh = _unit_cube_2x2x2()
        # Scalar field; only dof 0 is valid.
        T = GlobalFieldLayout(
            name="T", finite_element=Q1_HEX, num_dofs_per_basis_fn=1,
        )
        bcs = [
            DirichletBC(nodeset_name="xmin_nodes", field_name="T",
                        dofs=[2]),
        ]
        with self.assertRaisesRegex(ValueError, "dof 2 outside"):
            build_dof_map(mesh, [T], bcs)

    def test_duplicate_field_names_raises(self):
        mesh = _unit_cube_2x2x2()
        u1 = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=3,
        )
        u2 = GlobalFieldLayout(
            name="u", finite_element=Q1_HEX, num_dofs_per_basis_fn=1,
        )
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            build_dof_map(mesh, [u1, u2], [])


class TestFieldFamilyMatch(unittest.TestCase):

    def test_fe_family_must_match_mesh_family(self):
        # Hex mesh paired with a tet FE -> ValueError.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(
            name="u", finite_element=P1_TET, num_dofs_per_basis_fn=3,
        )
        with self.assertRaisesRegex(
            ValueError, "does not match mesh element_family"
        ):
            build_dof_map(mesh, [u], [])


class TestBcOnFieldWithoutVertexDofs(unittest.TestCase):

    def test_nodeset_bc_on_cell_only_field_raises(self):
        # Synthetic DG-style FE: all DOFs on the CELL entity, none on
        # vertices. Nodeset-keyed BCs cannot resolve through a non-
        # vertex DOF placement; build_dof_map must raise.
        cell_only_hex = FiniteElement(
            name="DG-cell-only-hex",
            element_family=ElementFamily.HEX_LINEAR,
            dofs_per_entity={EntityType.CELL: 8},
            interpolant_fn=hex_linear,
        )
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(
            name="u",
            finite_element=cell_only_hex,
            num_dofs_per_basis_fn=3,
        )
        bcs = [
            DirichletBC(nodeset_name="xmin_nodes", field_name="u",
                        dofs=[0]),
        ]
        with self.assertRaisesRegex(ValueError, "no VERTEX DOFs"):
            build_dof_map(mesh, [u], bcs)


if __name__ == "__main__":
    unittest.main()
