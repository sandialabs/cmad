"""Unit tests for interpolate_global_fields_at_ip."""
import unittest

import jax
import jax.numpy as jnp

from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.global_residuals.interpolation import interpolate_global_fields_at_ip


def _tet_barycenter_shapes() -> ShapeFunctionsAtIP:
    """Reference linear tet, shape functions + gradients at the barycenter.

    Nodes: (0,0,0), (1,0,0), (0,1,0), (0,0,1). Shape-function gradients
    are constant over the canonical reference element.
    """
    N = jnp.array([0.25, 0.25, 0.25, 0.25])
    grad_N = jnp.array([
        [-1., -1., -1.],
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.],
    ])
    return ShapeFunctionsAtIP(N=N, grad_N=grad_N)


class TestInterpolation(unittest.TestCase):
    def test_single_block_u_only_shape(self):
        U = [jnp.arange(12, dtype=float).reshape(4, 3)]
        shapes = _tet_barycenter_shapes()
        result = interpolate_global_fields_at_ip(U, [shapes], ["u"])

        self.assertEqual(result.fields["u"].shape, (3,))
        self.assertEqual(result.grad_fields["u"].shape, (3, 3))
        self.assertTrue(jnp.allclose(
            result.fields["u"], shapes.N @ U[0]))
        self.assertTrue(jnp.allclose(
            result.grad_fields["u"], U[0].T @ shapes.grad_N))

    def test_multi_block_u_p_iteration(self):
        U = [
            jnp.arange(12, dtype=float).reshape(4, 3),
            jnp.arange(4, dtype=float).reshape(4, 1),
        ]
        shapes = _tet_barycenter_shapes()
        result = interpolate_global_fields_at_ip(
            U, [shapes, shapes], ["u", "p"])

        self.assertEqual(result.fields["u"].shape, (3,))
        self.assertEqual(result.fields["p"].shape, (1,))
        self.assertEqual(result.grad_fields["u"].shape, (3, 3))
        self.assertEqual(result.grad_fields["p"].shape, (1, 3))

    def test_pytree_round_trip(self):
        U = [jnp.arange(12, dtype=float).reshape(4, 3)]
        shapes = _tet_barycenter_shapes()

        jitted = jax.jit(
            lambda U_in, shapes_in:
                interpolate_global_fields_at_ip(U_in, shapes_in, ["u"]),
        )
        result = jitted(U, [shapes])

        leaves, treedef = jax.tree_util.tree_flatten(result)
        self.assertEqual(len(leaves), 2)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertTrue(jnp.allclose(
            result.fields["u"], rebuilt.fields["u"]))
        self.assertTrue(jnp.allclose(
            result.grad_fields["u"], rebuilt.grad_fields["u"]))


if __name__ == "__main__":
    unittest.main()
