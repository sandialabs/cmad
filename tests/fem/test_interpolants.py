"""Unit tests for linear hex and tet interpolants.

Partition-of-unity (sum of shape functions == 1) and reference-gradient
sum-to-zero (column-sum of grad_N == 0) are swept over every integration
point of every supported quadrature rule for the corresponding element
family. A failure surfaces the (degree, ip) pair via ``subTest``.
"""
import unittest

import jax.numpy as jnp
import numpy as np

from cmad.fem.interpolants import hex_linear, tet_linear
from cmad.fem.quadrature import hex_quadrature, tet_quadrature


class TestHexLinear(unittest.TestCase):
    _degrees = tuple(range(1, 8))

    def test_partition_of_unity(self):
        for degree in self._degrees:
            rule = hex_quadrature(degree)
            for idx in range(rule.xi.shape[0]):
                shapes = hex_linear(jnp.asarray(rule.xi[idx]))
                with self.subTest(degree=degree, ip=idx):
                    self.assertAlmostEqual(
                        float(jnp.sum(shapes.N)), 1.0, places=12)

    def test_gradient_sum_to_zero(self):
        for degree in self._degrees:
            rule = hex_quadrature(degree)
            for idx in range(rule.xi.shape[0]):
                shapes = hex_linear(jnp.asarray(rule.xi[idx]))
                grad_sum = np.asarray(jnp.sum(shapes.grad_N, axis=0))
                with self.subTest(degree=degree, ip=idx):
                    self.assertTrue(np.allclose(
                        grad_sum, np.zeros(3), atol=1e-12))


class TestTetLinear(unittest.TestCase):
    _degrees = (1, 2, 3, 4, 5, 6)

    def test_partition_of_unity(self):
        for degree in self._degrees:
            rule = tet_quadrature(degree)
            for idx in range(rule.xi.shape[0]):
                shapes = tet_linear(jnp.asarray(rule.xi[idx]))
                with self.subTest(degree=degree, ip=idx):
                    self.assertAlmostEqual(
                        float(jnp.sum(shapes.N)), 1.0, places=12)

    def test_gradient_sum_to_zero(self):
        for degree in self._degrees:
            rule = tet_quadrature(degree)
            for idx in range(rule.xi.shape[0]):
                shapes = tet_linear(jnp.asarray(rule.xi[idx]))
                grad_sum = np.asarray(jnp.sum(shapes.grad_N, axis=0))
                with self.subTest(degree=degree, ip=idx):
                    self.assertTrue(np.allclose(
                        grad_sum, np.zeros(3), atol=1e-12))


if __name__ == "__main__":
    unittest.main()
