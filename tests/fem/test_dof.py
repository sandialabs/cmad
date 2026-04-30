"""Unit tests for `cmad.fem.dof`.

Covers `GlobalFieldLayout` validation, `GlobalDofMap` formula round-
trips and counts, multi-field layouts, FE-family / mesh-family
agreement, sideset-keyed BC resolution (single + multi-sideset, with
intra-BC dedup), time-dependent value evaluation, cross-BC value
consistency check, and `build_dof_map` error paths.
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

ALL_SIDES = [
    "xmin_sides", "xmax_sides",
    "ymin_sides", "ymax_sides",
    "zmin_sides", "zmax_sides",
]


def _unit_cube_2x2x2():
    """Shared fixture: 27-node 2x2x2 hex mesh on the unit cube."""
    return StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))


class TestComponentsByFieldValidation(unittest.TestCase):

    def test_rejects_zero_component_count(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            build_dof_map(mesh, [u], [], components_by_field={"u": 0})

    def test_rejects_mismatched_keys(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        with self.assertRaisesRegex(ValueError, "must match field-layout"):
            build_dof_map(mesh, [u], [], components_by_field={"v": 3})


class TestGlobalDofMapFormula(unittest.TestCase):

    def test_eq_index_single_field(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        dm = build_dof_map(mesh, [u], [], components_by_field={"u": 3})
        # Field 0, basis_fn 5, component 2: 0 + 5*3 + 2 = 17
        self.assertEqual(dm.eq_index(0, 5, 2), 17)
        # Last basis_fn, last component: 0 + 26*3 + 2 = 80
        self.assertEqual(dm.eq_index(0, 26, 2), 80)

    def test_eq_index_multi_field(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        p = GlobalFieldLayout(name="p", finite_element=Q1_HEX)
        dm = build_dof_map(
            mesh, [u, p], [], components_by_field={"u": 3, "p": 1},
        )
        # Field 0 (u) occupies [0, 81); field 1 (p) occupies [81, 108).
        self.assertEqual(dm.eq_index(0, 0, 0), 0)
        self.assertEqual(dm.eq_index(1, 0, 0), 81)
        self.assertEqual(dm.eq_index(1, 26, 0), 107)

    def test_total_dof_count(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        p = GlobalFieldLayout(name="p", finite_element=Q1_HEX)
        dm = build_dof_map(
            mesh, [u, p], [], components_by_field={"u": 3, "p": 1},
        )
        self.assertEqual(dm.num_total_dofs, 81 + 27)

    def test_block_offsets_cumulative(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        p = GlobalFieldLayout(name="p", finite_element=Q1_HEX)
        T = GlobalFieldLayout(name="T", finite_element=Q1_HEX)
        dm = build_dof_map(
            mesh, [u, p, T], [],
            components_by_field={"u": 3, "p": 1, "T": 1},
        )
        np.testing.assert_array_equal(
            dm.block_offsets, np.array([0, 81, 108, 135], dtype=np.intp)
        )


class TestGlobalDofMapPartition(unittest.TestCase):

    def test_free_plus_prescribed_equals_total(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        # Disjoint per-face dof clamp — no double-prescription.
        bcs = [
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[0]),
            DirichletBC(sideset_names=["ymin_sides"], field_name="u",
                        dofs=[1]),
            DirichletBC(sideset_names=["zmin_sides"], field_name="u",
                        dofs=[2]),
        ]
        dm = build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})
        self.assertEqual(
            dm.num_free_dofs + dm.num_prescribed_dofs, dm.num_total_dofs
        )

    def test_field_with_no_bcs_has_zero_prescribed(self):
        # Multi-field; BC on u only. p contributes no prescribed entries.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        p = GlobalFieldLayout(name="p", finite_element=Q1_HEX)
        bcs = [
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[0]),
        ]
        dm = build_dof_map(
            mesh, [u, p], bcs, components_by_field={"u": 3, "p": 1},
        )
        # All prescribed eqs lie inside u's range [0, 81).
        self.assertTrue(np.all(dm.prescribed_indices < 81))


class TestVertexBcResolution(unittest.TestCase):

    def test_xmin_clamp_resolves_to_correct_eqs(self):
        # Q1_HEX has dofs_per_entity[VERTEX]=1: identity basis_fn[v]=v.
        # xmin_sides walked + deduped recovers the 9 vertices on the
        # x=0 face — same set as node_sets["xmin_nodes"].
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        bcs = [
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[0]),
        ]
        dm = build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})
        # 9 vertices on xmin face, dof 0 only -> 9 prescribed eqs at v*3+0.
        expected = np.sort(mesh.node_sets["xmin_nodes"]) * 3
        np.testing.assert_array_equal(dm.prescribed_indices, expected)

    def test_multi_sideset_clamp_dedups_intra_bc(self):
        # One BC clamping all 6 faces. Boundary vertices on shared
        # edges/corners should appear once, not multiple times.
        # For 2x2x2 mesh: 27 total nodes, 1 interior, 26 boundary.
        # 3 components → 78 prescribed eqs.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        bcs = [
            DirichletBC(sideset_names=ALL_SIDES, field_name="u",
                        dofs=[0, 1, 2]),
        ]
        dm = build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})
        self.assertEqual(dm.num_prescribed_dofs, 26 * 3)
        # Interior vertex is index 13 (center of 3x3x3 grid). Its eqs
        # 39, 40, 41 should NOT be in prescribed_indices.
        for eq in (39, 40, 41):
            self.assertNotIn(eq, dm.prescribed_indices)


class TestEvaluatePrescribedValues(unittest.TestCase):

    def _setup_clamped_xmin(self, values):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        bcs = [
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[0, 1, 2], values=values),
        ]
        return build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})

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
        # Use the BC's deduplicated coord order for the reference; the
        # returned values flat ordering matches prescribed_indices,
        # which for a single BC clamping a single face is exactly
        # (sorted vertex)-major (dof)-minor.
        coords_xmin = dm.resolved_bcs[0].set_coords
        expected = np.zeros((9, 3))
        expected[:, 0] = 0.5 * coords_xmin[:, 0] + coords_xmin[:, 1]
        np.testing.assert_allclose(vals.reshape(9, 3), expected)

    def test_callable_time_dependent(self):
        # u_x = t * x on xmax face (where x = 1 always).
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        def vals_fn(coords, t):
            n = coords.shape[0]
            out = np.zeros((n, 1))
            out[:, 0] = t * coords[:, 0]
            return out
        bcs = [
            DirichletBC(sideset_names=["xmax_sides"], field_name="u",
                        dofs=[0], values=vals_fn),
        ]
        dm = build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})
        v0 = dm.evaluate_prescribed_values(t=0.0)
        v3 = dm.evaluate_prescribed_values(t=3.0)
        np.testing.assert_allclose(v0, np.zeros_like(v0))
        np.testing.assert_allclose(v3, np.full_like(v3, 3.0))


class TestCrossBcConsistency(unittest.TestCase):

    def test_overlapping_consistent_bcs_silent(self):
        # Two BCs on the same sideset and same component, both
        # homogeneous. Cross-BC overlap on every (vertex, dof=0) eq;
        # values agree (zero); evaluate succeeds silently.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        bcs = [
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[0]),
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[0]),
        ]
        dm = build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})
        # Deduplicated: should have exactly 9 prescribed eqs, not 18.
        self.assertEqual(dm.num_prescribed_dofs, 9)
        np.testing.assert_array_equal(
            dm.evaluate_prescribed_values(),
            np.zeros(9, dtype=np.float64),
        )

    def test_overlapping_inconsistent_bcs_raise_at_evaluate(self):
        # Two BCs on the same sideset and component with different
        # constant values. Build succeeds (overlap detection is
        # structural); evaluate raises with a diagnostic.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        bcs = [
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[0], values=[0.0]),
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[0], values=[1.0]),
        ]
        dm = build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})
        with self.assertRaisesRegex(
            ValueError, "inconsistent prescribed values"
        ):
            dm.evaluate_prescribed_values()

    def test_partial_component_overlap_consistent(self):
        # BC1 clamps dofs [0, 1] on xmin; BC2 clamps dofs [1, 2] on
        # xmin. Overlap only on dof=1 with consistent values; BC1's
        # dof=0 and BC2's dof=2 are uniquely prescribed.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        bcs = [
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[0, 1], values=[0.5, 1.0]),
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[1, 2], values=[1.0, 2.0]),
        ]
        dm = build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})
        # Deduped: 9 vertices x 3 components = 27 prescribed eqs.
        self.assertEqual(dm.num_prescribed_dofs, 27)
        vals = dm.evaluate_prescribed_values()
        # Per-vertex pattern is [0.5, 1.0, 2.0] for components 0/1/2.
        np.testing.assert_allclose(
            vals.reshape(9, 3),
            np.broadcast_to([0.5, 1.0, 2.0], (9, 3)),
        )

    def test_partial_component_overlap_inconsistent_raises(self):
        # Same as above but BC1 and BC2 disagree on dof=1.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        bcs = [
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[0, 1], values=[0.5, 1.0]),
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[1, 2], values=[5.0, 2.0]),
        ]
        dm = build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})
        with self.assertRaisesRegex(
            ValueError, "inconsistent prescribed values"
        ):
            dm.evaluate_prescribed_values()


class TestBuildDofMapErrors(unittest.TestCase):

    def test_unknown_field_name_raises(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        bcs = [
            DirichletBC(sideset_names=["xmin_sides"], field_name="temperature",
                        dofs=[0]),
        ]
        with self.assertRaisesRegex(ValueError, "unknown field"):
            build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})

    def test_unknown_sideset_name_raises(self):
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        bcs = [
            DirichletBC(sideset_names=["not_a_real_sideset"], field_name="u",
                        dofs=[0]),
        ]
        with self.assertRaisesRegex(KeyError, "unknown sideset"):
            build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})

    def test_dof_index_out_of_range_raises(self):
        mesh = _unit_cube_2x2x2()
        # Scalar field; only dof 0 is valid.
        T = GlobalFieldLayout(name="T", finite_element=Q1_HEX)
        bcs = [
            DirichletBC(sideset_names=["xmin_sides"], field_name="T",
                        dofs=[2]),
        ]
        with self.assertRaisesRegex(ValueError, "dof 2 outside"):
            build_dof_map(mesh, [T], bcs, components_by_field={"T": 1})

    def test_duplicate_field_names_raises(self):
        mesh = _unit_cube_2x2x2()
        u1 = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        u2 = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            build_dof_map(
                mesh, [u1, u2], [], components_by_field={"u": 3},
            )


class TestFieldFamilyMatch(unittest.TestCase):

    def test_fe_family_must_match_mesh_family(self):
        # Hex mesh paired with a tet FE -> ValueError.
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=P1_TET)
        with self.assertRaisesRegex(
            ValueError, "does not match mesh element_family"
        ):
            build_dof_map(mesh, [u], [], components_by_field={"u": 3})


class TestBcOnFieldWithoutVertexDofs(unittest.TestCase):

    def test_sideset_bc_on_cell_only_field_raises(self):
        # Synthetic DG-style FE: all DOFs on the CELL entity, none on
        # vertices. Sideset-keyed BCs cannot resolve through a non-
        # vertex DOF placement; build_dof_map must raise.
        cell_only_hex = FiniteElement(
            name="DG-cell-only-hex",
            element_family=ElementFamily.HEX_LINEAR,
            dofs_per_entity={EntityType.CELL: 8},
            interpolant_fn=hex_linear,
        )
        mesh = _unit_cube_2x2x2()
        u = GlobalFieldLayout(name="u", finite_element=cell_only_hex)
        bcs = [
            DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                        dofs=[0]),
        ]
        with self.assertRaisesRegex(ValueError, "no VERTEX DOFs"):
            build_dof_map(mesh, [u], bcs, components_by_field={"u": 3})


if __name__ == "__main__":
    unittest.main()
