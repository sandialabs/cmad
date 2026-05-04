"""Tests for ``cmad.fem.precompute``.

Covers per-block reference-frame geometry caching: cache values match
direct ``iso_jac_at_ip`` computations at picked (block, elem, IP)
points, and integrating ``iso_jac_det * quad_w`` across all
(block, elem, IP) recovers the analytic mesh volume on hex and tet
meshes.
"""
import unittest

import jax.numpy as jnp
import numpy as np

from cmad.fem.assembly import iso_jac_at_ip
from cmad.fem.dof import GlobalFieldLayout
from cmad.fem.finite_element import P1_TET, Q1_HEX
from cmad.fem.mesh import StructuredHexMesh, hex_to_tet_split
from cmad.fem.precompute import precompute_block_geometry
from cmad.fem.quadrature import hex_quadrature, tet_quadrature


def _hex_quadrature_by_family():
    from cmad.fem.element_family import ElementFamily
    return {ElementFamily.HEX_LINEAR: hex_quadrature(degree=2)}


def _tet_quadrature_by_family():
    from cmad.fem.element_family import ElementFamily
    return {ElementFamily.TET_LINEAR: tet_quadrature(degree=2)}


def _total_volume(cache):
    total = 0.0
    for block_cache in cache.values():
        det = np.asarray(block_cache.per_elem.iso_jac_det)
        w = np.asarray(block_cache.shared.quad_w)
        total += float((det * w[None, :]).sum())
    return total


class TestCacheCorrectness(unittest.TestCase):

    def test_cache_matches_direct_iso_jac_at_ip(self):
        # 2x2x2 hex mesh, 8-IP rule. For several (elem, IP) pairs check
        # that the cache values match a direct iso_jac_at_ip + manual
        # lift computation.
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        layouts = [GlobalFieldLayout(name="u", finite_element=Q1_HEX)]
        quadrature_by_family = _hex_quadrature_by_family()

        cache = precompute_block_geometry(
            mesh, quadrature_by_family, layouts,
        )

        block_name = "all"
        block_cache = cache[block_name]

        elem_indices = mesh.element_blocks[block_name]
        connectivity_block = mesh.connectivity[elem_indices]
        quad_rule = quadrature_by_family[mesh.element_family]
        n_ip = quad_rule.xi.shape[0]
        n_elem = len(elem_indices)

        for elem in (0, n_elem // 2, n_elem - 1):
            X_elem = jnp.asarray(mesh.nodes[connectivity_block[elem]])
            for ip in range(n_ip):
                xi_ip = jnp.asarray(quad_rule.xi[ip])
                geom_shapes = (
                    mesh.geometric_finite_element.interpolant_fn(xi_ip)
                )
                _, iso_jac_det_ref, iso_jac_ref = iso_jac_at_ip(
                    geom_shapes.grad_N, X_elem,
                )

                np.testing.assert_allclose(
                    float(block_cache.per_elem.iso_jac_det[elem, ip]),
                    float(iso_jac_det_ref),
                    rtol=1e-12,
                )
                np.testing.assert_allclose(
                    np.asarray(block_cache.per_elem.coords_ip[elem, ip]),
                    np.asarray(geom_shapes.N @ X_elem),
                    rtol=1e-12,
                )

                field_shapes = (
                    layouts[0].finite_element.interpolant_fn(xi_ip)
                )
                grad_N_phys_ref = (
                    field_shapes.grad_N @ jnp.linalg.inv(iso_jac_ref)
                )
                np.testing.assert_allclose(
                    np.asarray(
                        block_cache.per_elem
                        .field_grad_N_phys_per_block[0][elem, ip]
                    ),
                    np.asarray(grad_N_phys_ref),
                    rtol=1e-12,
                )
                np.testing.assert_allclose(
                    np.asarray(
                        block_cache.shared.field_N_per_block[0][ip]
                    ),
                    np.asarray(field_shapes.N),
                    rtol=1e-12,
                )

    def test_cache_shapes(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        layouts = [GlobalFieldLayout(name="u", finite_element=Q1_HEX)]
        cache = precompute_block_geometry(
            mesh, _hex_quadrature_by_family(), layouts,
        )

        n_b = 8
        n_ip = 8
        n_dofs = 8
        block = cache["all"]
        self.assertEqual(block.per_elem.iso_jac_det.shape, (n_b, n_ip))
        self.assertEqual(block.per_elem.coords_ip.shape, (n_b, n_ip, 3))
        self.assertEqual(
            block.per_elem.field_grad_N_phys_per_block[0].shape,
            (n_b, n_ip, n_dofs, 3),
        )
        self.assertEqual(block.shared.quad_w.shape, (n_ip,))
        self.assertEqual(
            block.shared.field_N_per_block[0].shape, (n_ip, n_dofs),
        )


class TestVolumeSanity(unittest.TestCase):

    def test_unit_cube_hex(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        layouts = [GlobalFieldLayout(name="u", finite_element=Q1_HEX)]
        cache = precompute_block_geometry(
            mesh, _hex_quadrature_by_family(), layouts,
        )
        np.testing.assert_allclose(_total_volume(cache), 1.0, rtol=1e-12)

    def test_offset_box_hex(self):
        mesh = StructuredHexMesh(
            (2.0, 3.0, 5.0), (3, 2, 4), origin=(1.0, 1.0, 1.0),
        )
        layouts = [GlobalFieldLayout(name="u", finite_element=Q1_HEX)]
        cache = precompute_block_geometry(
            mesh, _hex_quadrature_by_family(), layouts,
        )
        np.testing.assert_allclose(_total_volume(cache), 30.0, rtol=1e-12)

    def test_unit_cube_tet_split(self):
        hex_mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        tet_mesh = hex_to_tet_split(hex_mesh)
        layouts = [GlobalFieldLayout(name="u", finite_element=P1_TET)]
        cache = precompute_block_geometry(
            tet_mesh, _tet_quadrature_by_family(), layouts,
        )
        np.testing.assert_allclose(_total_volume(cache), 1.0, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
