"""Tests for ``cmad.io.results``.

Covers component-name conventions, internal/Exodus storage permutation,
volume-weighted IP -> element reduction, and global-field volume-
integral averaging via the geometry cache.
"""
import unittest

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import build_fe_problem
from cmad.fem.finite_element import P1_TET, Q1_HEX
from cmad.fem.mesh import Mesh, StructuredHexMesh, hex_to_tet_split
from cmad.fem.precompute import precompute_block_geometry
from cmad.fem.quadrature import hex_quadrature, tet_quadrature
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.io.results import (
    FieldSpec,
    component_names,
    from_exodus_storage,
    ip_average_to_element,
    to_exodus_storage,
    volume_average_global_field,
)
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.var_types import VarType
from cmad.parameters.parameters import Parameters


def _hex_quadrature_by_family():
    return {ElementFamily.HEX_LINEAR: hex_quadrature(degree=2)}


def _tet_quadrature_by_family():
    return {ElementFamily.TET_LINEAR: tet_quadrature(degree=2)}


def _make_elastic_parameters(kappa=100.0, mu=50.0):
    values = {"elastic": {"kappa": kappa, "mu": mu}}
    active = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Parameters(values, active, transforms)


def _build_tiny_fe_problem(mesh):
    if mesh.element_family == ElementFamily.HEX_LINEAR:
        fe = Q1_HEX
    elif mesh.element_family == ElementFamily.TET_LINEAR:
        fe = P1_TET
    else:
        raise ValueError(f"unsupported family {mesh.element_family}")
    layout = GlobalFieldLayout(name="u", finite_element=fe)
    dof_map = build_dof_map(
        mesh, [layout], [], components_by_field={"u": 3},
    )
    gr = SmallDispEquilibrium(ndims=3)
    elastic = Elastic(
        _make_elastic_parameters(), def_type=DefType.FULL_3D,
    )
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={
            name: elastic for name in mesh.element_blocks
        },
    )


class TestComponentNames(unittest.TestCase):

    def test_scalar_returns_bare_name(self):
        spec = FieldSpec(name="temperature", var_type=VarType.SCALAR)
        self.assertEqual(component_names(spec, ndims=3), ("temperature",))

    def test_vector_xyz_suffixes(self):
        spec = FieldSpec(name="displacement", var_type=VarType.VECTOR)
        self.assertEqual(
            component_names(spec, ndims=3),
            ("displacement_x", "displacement_y", "displacement_z"),
        )

    def test_vector_2d(self):
        spec = FieldSpec(name="u", var_type=VarType.VECTOR)
        self.assertEqual(component_names(spec, ndims=2), ("u_x", "u_y"))

    def test_sym_tensor_exodus_order(self):
        spec = FieldSpec(name="cauchy", var_type=VarType.SYM_TENSOR)
        self.assertEqual(
            component_names(spec, ndims=3),
            (
                "cauchy_xx", "cauchy_yy", "cauchy_zz",
                "cauchy_xy", "cauchy_xz", "cauchy_yz",
            ),
        )

    def test_tensor_row_major_nine_components(self):
        spec = FieldSpec(name="F", var_type=VarType.TENSOR)
        self.assertEqual(
            component_names(spec, ndims=3),
            (
                "F_xx", "F_xy", "F_xz",
                "F_yx", "F_yy", "F_yz",
                "F_zx", "F_zy", "F_zz",
            ),
        )


class TestStoragePermutation(unittest.TestCase):

    def test_scalar_passthrough(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            to_exodus_storage(x, VarType.SCALAR), x,
        )

    def test_vector_passthrough(self):
        x = np.arange(12.0).reshape(4, 3)
        np.testing.assert_array_equal(
            to_exodus_storage(x, VarType.VECTOR), x,
        )

    def test_sym_tensor_internal_to_exodus_known_layout(self):
        # Internal: [xx, xy, xz, yy, yz, zz]
        internal = np.array([[10.0, 12.0, 13.0, 22.0, 23.0, 33.0]])
        # Exodus:   [xx, yy, zz, xy, xz, yz]
        expected = np.array([[10.0, 22.0, 33.0, 12.0, 13.0, 23.0]])
        np.testing.assert_array_equal(
            to_exodus_storage(internal, VarType.SYM_TENSOR),
            expected,
        )

    def test_sym_tensor_round_trip_returns_input(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5, 6))
        out = from_exodus_storage(
            to_exodus_storage(x, VarType.SYM_TENSOR), VarType.SYM_TENSOR,
        )
        np.testing.assert_allclose(out, x)

    def test_tensor_passthrough(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((3, 9))
        np.testing.assert_array_equal(
            to_exodus_storage(x, VarType.TENSOR), x,
        )

    def test_works_with_extra_leading_axes(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((4, 7, 6))  # (n_step, n_elem, 6)
        out = from_exodus_storage(
            to_exodus_storage(x, VarType.SYM_TENSOR), VarType.SYM_TENSOR,
        )
        np.testing.assert_allclose(out, x)


class TestIpAverageToElement(unittest.TestCase):

    def _hex_cache(self, dims=(1.0, 1.0, 1.0), divs=(2, 2, 2)):
        mesh = StructuredHexMesh(dims, divs)
        layouts = [GlobalFieldLayout(name="u", finite_element=Q1_HEX)]
        cache = precompute_block_geometry(
            mesh, _hex_quadrature_by_family(), layouts,
        )
        return mesh, cache

    def test_uniform_field_returns_input_value(self):
        _, cache = self._hex_cache()
        n_b, n_ip = cache["all"].per_elem.iso_jac_det.shape
        values = np.full((n_b, n_ip), 7.0)
        out = ip_average_to_element(values, cache, "all")
        np.testing.assert_allclose(out, np.full((n_b,), 7.0), rtol=1e-12)

    def test_polynomial_volume_average_unit_cube(self):
        # f(x,y,z) = x on [0,1]^3 single hex; average should be 0.5.
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        layouts = [GlobalFieldLayout(name="u", finite_element=Q1_HEX)]
        cache = precompute_block_geometry(
            mesh, _hex_quadrature_by_family(), layouts,
        )
        coords_x = np.asarray(cache["all"].per_elem.coords_ip[..., 0])
        out = ip_average_to_element(coords_x, cache, "all")
        np.testing.assert_allclose(out, np.array([0.5]), rtol=1e-12)

    def test_vector_components_independent(self):
        _, cache = self._hex_cache()
        n_b, n_ip = cache["all"].per_elem.iso_jac_det.shape
        values = np.zeros((n_b, n_ip, 3))
        values[..., 0] = 2.0
        values[..., 1] = -1.5
        values[..., 2] = 0.0
        out = ip_average_to_element(values, cache, "all")
        np.testing.assert_allclose(out[:, 0], 2.0, rtol=1e-12)
        np.testing.assert_allclose(out[:, 1], -1.5, rtol=1e-12)
        np.testing.assert_allclose(out[:, 2], 0.0, atol=1e-12)

    def test_block_dispatch_isolates_other_blocks(self):
        # 2x2x2 hex split into block_a (elems 0..3) + block_b (4..7).
        base = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        mesh = Mesh(
            nodes=base.nodes,
            connectivity=base.connectivity,
            element_family=base.element_family,
            element_blocks={
                "block_a": np.arange(0, 4, dtype=np.intp),
                "block_b": np.arange(4, 8, dtype=np.intp),
            },
            node_sets={},
            side_sets={},
        )
        layouts = [GlobalFieldLayout(name="u", finite_element=Q1_HEX)]
        cache = precompute_block_geometry(
            mesh, _hex_quadrature_by_family(), layouts,
        )
        n_ip = cache["block_a"].per_elem.iso_jac_det.shape[1]
        values_a = np.full((4, n_ip), 3.0)
        out_a = ip_average_to_element(values_a, cache, "block_a")
        np.testing.assert_allclose(out_a, np.full((4,), 3.0), rtol=1e-12)
        # Wrong-block call would shape-mismatch since block_b has 4 elements
        # too in this mesh; sanity-check independent dispatch by passing
        # block_b values of a different magnitude.
        values_b = np.full((4, n_ip), -2.0)
        out_b = ip_average_to_element(values_b, cache, "block_b")
        np.testing.assert_allclose(out_b, np.full((4,), -2.0), rtol=1e-12)


class TestVolumeAverageGlobalField(unittest.TestCase):

    def test_constant_global_field_returns_constant_per_element(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        fe_problem = _build_tiny_fe_problem(mesh)
        n_dofs = fe_problem.dof_map.num_total_dofs
        # Set every component of every node to 3.5 (vector field).
        U = np.full(n_dofs, 3.5)
        out = volume_average_global_field(U, fe_problem, "all", "u")
        n_b = mesh.element_blocks["all"].size
        np.testing.assert_allclose(
            out, np.full((n_b, 3), 3.5), rtol=1e-12,
        )

    def test_linear_global_field_volume_average_matches_centroid(self):
        # U[node, 0] = nodes[node, 0]; per-element volume avg of U_x
        # equals each element's centroid x-coordinate (linear field
        # through Q1 hex → exact at centroid).
        mesh = StructuredHexMesh((2.0, 2.0, 2.0), (2, 2, 2))
        fe_problem = _build_tiny_fe_problem(mesh)
        n_nodes = mesh.nodes.shape[0]
        U = np.zeros(n_nodes * 3)
        U[0::3] = mesh.nodes[:, 0]  # x-component carries node x

        out = volume_average_global_field(U, fe_problem, "all", "u")

        # Per-element centroid x = mean of element-node x.
        elem_indices = mesh.element_blocks["all"]
        connectivity = mesh.connectivity[elem_indices]
        centroid_x = mesh.nodes[connectivity, 0].mean(axis=1)
        np.testing.assert_allclose(out[:, 0], centroid_x, rtol=1e-12)
        np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(out[:, 2], 0.0, atol=1e-12)

    def test_unknown_field_name_raises(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        fe_problem = _build_tiny_fe_problem(mesh)
        U = np.zeros(fe_problem.dof_map.num_total_dofs)
        with self.assertRaises(ValueError):
            volume_average_global_field(U, fe_problem, "all", "nope")


if __name__ == "__main__":
    unittest.main()
