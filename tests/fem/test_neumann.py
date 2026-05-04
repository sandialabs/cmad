"""Unit tests for `cmad.fem.neumann`.

Pins the resolver, the per-side surface evaluator, and the surface
scatter. Resolution validates field/sideset names and the VERTEX-only
FE constraint with exactly 1 DOF per vertex. The evaluator integrates
``-∫_side N · t̄ dA`` per element with degree-2 side quadrature and
the scatter distributes per-element side residuals into the global
``R`` via the field's eq formula. The threading tests cover the
``build_fe_problem`` ↔ ``FEProblem`` ↔ ``assemble_global`` path so
NBCs reach the global residual through the public builder.
"""
import unittest

import numpy as np
from jax.tree_util import tree_map

from cmad.fem.assembly import assemble_global
from cmad.fem.bcs import NeumannBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import build_fe_problem
from cmad.fem.finite_element import (
    P1_TET,
    Q1_HEX,
    EntityType,
    FiniteElement,
)
from cmad.fem.interpolants import hex_linear
from cmad.fem.mesh import Mesh, StructuredHexMesh
from cmad.fem.neumann import (
    assemble_side_neumann,
    resolve_neumann_bcs,
)
from cmad.fem.quadrature import quad_quadrature, tri_quadrature
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.parameters.parameters import Parameters

_SIDE_QUAD = {
    ElementFamily.HEX_LINEAR: quad_quadrature(degree=2),
    ElementFamily.TET_LINEAR: tri_quadrature(degree=2),
}


def _build_hex_mesh_and_dofmap(
        divisions=(1, 1, 1), lengths=(1.0, 1.0, 1.0),
):
    mesh = StructuredHexMesh(lengths, divisions)
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    dm = build_dof_map(
        mesh, [layout], [], components_by_field={"u": 3},
    )
    return mesh, dm


def _build_unit_tet_mesh_and_dofmap():
    """Single P1 tet with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1).

    Slant face (face 1) has vertices [1, 2, 3]; outward unit normal
    is (1, 1, 1) / sqrt(3); face area is sqrt(3) / 2.
    """
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.intp)
    element_blocks = {"all": np.array([0], dtype=np.intp)}
    side_sets = {
        "slant_sides": np.array([[0, 1]], dtype=np.intp),
    }
    mesh = Mesh(
        nodes=nodes,
        connectivity=connectivity,
        element_family=ElementFamily.TET_LINEAR,
        element_blocks=element_blocks,
        node_sets={},
        side_sets=side_sets,
    )
    layout = GlobalFieldLayout(name="u", finite_element=P1_TET)
    dm = build_dof_map(
        mesh, [layout], [], components_by_field={"u": 3},
    )
    return mesh, dm


class TestResolveNeumannBCs(unittest.TestCase):

    def test_unknown_field_raises(self):
        mesh, dm = _build_hex_mesh_and_dofmap()
        bc = NeumannBC(
            sideset_names=["xmax_sides"],
            field_name="not_a_field",
            values=[0.0, 0.0, 0.0],
        )
        with self.assertRaisesRegex(
            ValueError, "field_name='not_a_field'",
        ):
            resolve_neumann_bcs(mesh, dm, [bc])

    def test_unknown_sideset_raises(self):
        mesh, dm = _build_hex_mesh_and_dofmap()
        bc = NeumannBC(
            sideset_names=["bogus_sides"],
            field_name="u",
            values=[0.0, 0.0, 0.0],
        )
        with self.assertRaisesRegex(
            ValueError, "sideset_name='bogus_sides'",
        ):
            resolve_neumann_bcs(mesh, dm, [bc])

    def test_non_vertex_fe_raises(self):
        edge_fe = FiniteElement(
            name="P2_HEX_FAKE",
            element_family=ElementFamily.HEX_LINEAR,
            dofs_per_entity={
                EntityType.VERTEX: 1, EntityType.EDGE: 1,
            },
            interpolant_fn=hex_linear,
        )
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        layout = GlobalFieldLayout(name="u", finite_element=edge_fe)
        dm = build_dof_map(
            mesh, [layout], [], components_by_field={"u": 1},
        )
        bc = NeumannBC(
            sideset_names=["xmax_sides"],
            field_name="u",
            values=[0.0],
        )
        with self.assertRaisesRegex(
            NotImplementedError, "EDGE",
        ):
            resolve_neumann_bcs(mesh, dm, [bc])

    def test_sequence_values_length_mismatch_raises(self):
        mesh, dm = _build_hex_mesh_and_dofmap()
        bc = NeumannBC(
            sideset_names=["xmax_sides"],
            field_name="u",
            values=[0.0, 0.0],
        )
        with self.assertRaisesRegex(
            ValueError, "component count",
        ):
            resolve_neumann_bcs(mesh, dm, [bc])

    def test_grouping_single_sideset_one_group(self):
        mesh, dm = _build_hex_mesh_and_dofmap(divisions=(2, 1, 1))
        bc = NeumannBC(
            sideset_names=["xmin_sides"],
            field_name="u",
            values=[1.0, 0.0, 0.0],
        )
        [resolved] = resolve_neumann_bcs(mesh, dm, [bc])
        self.assertEqual(len(resolved.elem_ids_by_side), 1)
        # xmin_sides on hex -> local face 5 (per StructuredHexMesh).
        key = (ElementFamily.HEX_LINEAR, 5)
        self.assertIn(key, resolved.elem_ids_by_side)
        np.testing.assert_array_equal(
            resolved.elem_ids_by_side[key],
            np.array([0], dtype=np.intp),
        )

    def test_grouping_multi_sideset_two_groups(self):
        mesh, dm = _build_hex_mesh_and_dofmap(divisions=(2, 1, 1))
        bc = NeumannBC(
            sideset_names=["xmin_sides", "ymax_sides"],
            field_name="u",
            values=[1.0, 1.0, 1.0],
        )
        [resolved] = resolve_neumann_bcs(mesh, dm, [bc])
        # Two distinct local_side_ids: 5 (xmin) and 4 (ymax).
        self.assertEqual(len(resolved.elem_ids_by_side), 2)
        self.assertIn(
            (ElementFamily.HEX_LINEAR, 5),
            resolved.elem_ids_by_side,
        )
        self.assertIn(
            (ElementFamily.HEX_LINEAR, 4),
            resolved.elem_ids_by_side,
        )


class TestAssembleSideNeumann(unittest.TestCase):

    def test_empty_nbc_short_circuits(self):
        mesh, dm = _build_hex_mesh_and_dofmap()
        n_dofs = dm.num_total_dofs
        R = np.arange(n_dofs, dtype=np.float64)
        R_orig = R.copy()
        assemble_side_neumann(R, mesh, dm, [], _SIDE_QUAD, 0.0)
        np.testing.assert_array_equal(R, R_orig)

    def test_uniform_traction_zmax_unit_hex(self):
        # +z face of the unit cube has area 1; for uniform traction
        # (0, 0, p), each of the four +z corner nodes receives
        # R[node, 2] = -p / 4 (Q1 bilinear N integrates exactly under
        # degree-2 quadrature).
        mesh, dm = _build_hex_mesh_and_dofmap()
        p = 2.5
        bc = NeumannBC(
            sideset_names=["zmax_sides"],
            field_name="u",
            values=[0.0, 0.0, p],
        )
        resolved = resolve_neumann_bcs(mesh, dm, [bc])
        n_dofs = dm.num_total_dofs
        R = np.zeros(n_dofs, dtype=np.float64)
        assemble_side_neumann(R, mesh, dm, resolved, _SIDE_QUAD, 0.0)
        local_zmax = np.array([4, 5, 6, 7])
        global_zmax = mesh.connectivity[0, local_zmax]
        nonzero_eqs = set()
        for g in global_zmax:
            nonzero_eqs.add(int(g) * 3 + 2)
            np.testing.assert_allclose(
                R[g * 3 + 0], 0.0, atol=1e-12,
            )
            np.testing.assert_allclose(
                R[g * 3 + 1], 0.0, atol=1e-12,
            )
            np.testing.assert_allclose(
                R[g * 3 + 2], -p / 4.0, atol=1e-12,
            )
        for eq in range(n_dofs):
            if eq in nonzero_eqs:
                continue
            np.testing.assert_allclose(R[eq], 0.0, atol=1e-12)

    def test_slant_tet_uniform_traction(self):
        # Slant face of a unit tet has area sqrt(3)/2 and outward unit
        # normal (1, 1, 1)/sqrt(3). For uniform traction (1, 0, 0),
        # each face-resident vertex receives
        # R[node, 0] = -|t̄| · A_slant / 3 = -sqrt(3) / 6.
        mesh, dm = _build_unit_tet_mesh_and_dofmap()
        bc = NeumannBC(
            sideset_names=["slant_sides"],
            field_name="u",
            values=[1.0, 0.0, 0.0],
        )
        resolved = resolve_neumann_bcs(mesh, dm, [bc])
        n_dofs = dm.num_total_dofs
        R = np.zeros(n_dofs, dtype=np.float64)
        assemble_side_neumann(R, mesh, dm, resolved, _SIDE_QUAD, 0.0)
        expected = -np.sqrt(3.0) / 6.0
        for node in (1, 2, 3):
            np.testing.assert_allclose(
                R[node * 3 + 0], expected, atol=1e-12,
            )
            np.testing.assert_allclose(
                R[node * 3 + 1], 0.0, atol=1e-12,
            )
            np.testing.assert_allclose(
                R[node * 3 + 2], 0.0, atol=1e-12,
            )
        for k in range(3):
            np.testing.assert_allclose(
                R[0 * 3 + k], 0.0, atol=1e-12,
            )

    def test_multi_element_scatter_zmax(self):
        # 2x1x1 hex mesh: zmax_sides spans two elements with a shared
        # edge along x = 1, z = 1. Uniform z-traction p on zmax;
        # each element contributes -p/4 to its 4 +z corners; shared
        # corners accumulate -p/2.
        mesh, dm = _build_hex_mesh_and_dofmap(
            divisions=(2, 1, 1), lengths=(2.0, 1.0, 1.0),
        )
        p = 1.0
        bc = NeumannBC(
            sideset_names=["zmax_sides"],
            field_name="u",
            values=[0.0, 0.0, p],
        )
        resolved = resolve_neumann_bcs(mesh, dm, [bc])
        n_dofs = dm.num_total_dofs
        R = np.zeros(n_dofs, dtype=np.float64)
        assemble_side_neumann(R, mesh, dm, resolved, _SIDE_QUAD, 0.0)
        local_zmax = np.array([4, 5, 6, 7])
        e0 = mesh.connectivity[0, local_zmax]
        e1 = mesh.connectivity[1, local_zmax]
        shared = np.intersect1d(e0, e1)
        only_e0 = np.setdiff1d(e0, shared)
        only_e1 = np.setdiff1d(e1, shared)
        self.assertEqual(shared.size, 2)
        self.assertEqual(only_e0.size, 2)
        self.assertEqual(only_e1.size, 2)
        for g in only_e0:
            np.testing.assert_allclose(
                R[g * 3 + 2], -p / 4.0, atol=1e-12,
            )
        for g in only_e1:
            np.testing.assert_allclose(
                R[g * 3 + 2], -p / 4.0, atol=1e-12,
            )
        for g in shared:
            np.testing.assert_allclose(
                R[g * 3 + 2], -p / 2.0, atol=1e-12,
            )


def _make_elastic_parameters() -> Parameters:
    values: dict = {"elastic": {"kappa": 100.0, "mu": 50.0}}
    active_flags = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Parameters(values, active_flags, transforms)


def _build_unit_hex_fe_problem(
        neumann_bcs: tuple[NeumannBC, ...] = (),
):
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    dof_map = build_dof_map(
        mesh, [layout], [], components_by_field={"u": 3},
    )
    gr = SmallDispEquilibrium(ndims=3)
    elastic = Elastic(_make_elastic_parameters(), def_type=DefType.FULL_3D)
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=gr,
        models_by_block={"all": elastic},
        neumann_bcs=neumann_bcs,
    )


class TestNeumannBCThreading(unittest.TestCase):

    def test_build_fe_problem_threads_neumann_bcs(self):
        bc = NeumannBC(
            sideset_names=["zmax_sides"],
            field_name="u",
            values=[0.0, 0.0, 1.5],
        )
        fe_problem = _build_unit_hex_fe_problem(neumann_bcs=(bc,))
        self.assertEqual(len(fe_problem.resolved_neumann_bcs), 1)
        resolved = fe_problem.resolved_neumann_bcs[0]
        self.assertEqual(resolved.field_idx, 0)
        self.assertEqual(resolved.num_components, 3)
        self.assertIn(
            (ElementFamily.HEX_LINEAR, 1),
            resolved.elem_ids_by_side,
        )
        self.assertIn(
            ElementFamily.HEX_LINEAR, fe_problem.side_quadrature,
        )

    def test_assemble_global_includes_neumann_contribution(self):
        # At U=0 the linear-elastic volume contribution is zero, so any
        # nonzero R entry must come from the Neumann scatter inside
        # assemble_global. Pin against the unit-hex zmax expectation:
        # -p/4 at each of the four +z corner nodes' z-DOFs.
        p = 2.5
        bc = NeumannBC(
            sideset_names=["zmax_sides"],
            field_name="u",
            values=[0.0, 0.0, p],
        )
        fe_problem = _build_unit_hex_fe_problem(neumann_bcs=(bc,))
        n_dofs = fe_problem.dof_map.num_total_dofs
        U_zero = np.zeros(n_dofs, dtype=np.float64)
        _, R, _ = assemble_global(fe_problem, U_zero, U_zero, t=0.0)
        local_zmax = np.array([4, 5, 6, 7])
        global_zmax = fe_problem.mesh.connectivity[0, local_zmax]
        for g in global_zmax:
            np.testing.assert_allclose(
                R[g * 3 + 2], -p / 4.0, atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
