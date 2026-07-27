"""Tests for :mod:`cmad.remap.gmls`: GMLS reconstruction operators.

Two complementary properties give the coverage:

* Reproduction (exactness): an order ``m`` fit reproduces polynomials of
  degree ``<= m`` exactly, to roundoff, for the value and each first
  derivative. This checks the basis, the scaling, and the value and gradient
  extraction, but it is automatic for polynomials, so on its own it says
  nothing about other fields.

* Convergence (approximation quality): on a smooth field that is not a
  polynomial the error must drop at the GMLS rates under cloud refinement --
  value ``O(h^{m+1})`` and gradient ``O(h^m)``. This is what exercises the
  weight and the locality on realistic data, which a polynomial cannot.

Partition of unity and a constant field having zero gradient are sanity
checks on the same machinery.
"""
import unittest

import numpy as np

from cmad.remap.gmls import _monomial_exponents, build_gmls_operators

_TARGETS = np.array(
    [[0.50, 0.50], [0.40, 0.60], [0.60, 0.40], [0.45, 0.55], [0.55, 0.45]]
)


def _source_cloud(n: int = 16) -> np.ndarray:
    """A jittered n x n grid of 2D points in the unit square.

    The jitter uses ``np.random.default_rng(0)``, so the cloud is the same on
    every run (and the convergence rates below are reproducible).
    """
    rng = np.random.default_rng(0)
    axis = np.linspace(0.0, 1.0, n)
    grid_x, grid_y = np.meshgrid(axis, axis)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    points += rng.uniform(-0.3 / n, 0.3 / n, points.shape)
    return points


def _poly_and_grad(coords: np.ndarray, exponents: np.ndarray, coeffs):
    """Evaluate a 2D polynomial (given monomial coeffs) and its gradient."""
    x, y = coords[:, 0], coords[:, 1]
    value = np.zeros(coords.shape[0])
    grad_x = np.zeros(coords.shape[0])
    grad_y = np.zeros(coords.shape[0])
    for coeff, (ex, ey) in zip(coeffs, exponents, strict=True):
        value += coeff * x**ex * y**ey
        if ex > 0:
            grad_x += coeff * ex * x ** (ex - 1) * y**ey
        if ey > 0:
            grad_y += coeff * ey * x**ex * y ** (ey - 1)
    return value, grad_x, grad_y


def _smooth_field(coords: np.ndarray):
    """The field sin(pi x) sin(pi y) (not a polynomial) and its gradient."""
    x, y = coords[:, 0], coords[:, 1]
    value = np.sin(np.pi * x) * np.sin(np.pi * y)
    grad_x = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    grad_y = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    return value, grad_x, grad_y


class TestGmlsReproduction(unittest.TestCase):
    def test_reproduces_polynomials_orders_1_2_3(self):
        source = _source_cloud()
        rng = np.random.default_rng(1)
        for order in (1, 2, 3):
            with self.subTest(poly_order=order):
                exponents = _monomial_exponents(order, 2)
                coeffs = rng.standard_normal(exponents.shape[0])
                ops = build_gmls_operators(source, _TARGETS, poly_order=order)
                f_source, _, _ = _poly_and_grad(source, exponents, coeffs)
                f_t, gx_t, gy_t = _poly_and_grad(_TARGETS, exponents, coeffs)
                self.assertTrue(
                    np.allclose(ops.value @ f_source, f_t, rtol=0, atol=1e-11)
                )
                self.assertTrue(
                    np.allclose(ops.grad[0] @ f_source, gx_t, rtol=0,
                                atol=1e-10)
                )
                self.assertTrue(
                    np.allclose(ops.grad[1] @ f_source, gy_t, rtol=0,
                                atol=1e-10)
                )

    def test_partition_of_unity(self):
        source = _source_cloud()
        ops = build_gmls_operators(source, _TARGETS, poly_order=2)
        row_sums = np.asarray(ops.value.sum(axis=1)).ravel()
        self.assertTrue(np.allclose(row_sums, 1.0, rtol=0, atol=1e-11))

    def test_constant_field_zero_gradient(self):
        source = _source_cloud()
        ops = build_gmls_operators(source, _TARGETS, poly_order=2)
        ones = np.ones(source.shape[0])
        self.assertTrue(np.allclose(ops.value @ ones, 1.0, rtol=0, atol=1e-11))
        self.assertTrue(
            np.allclose(ops.grad[0] @ ones, 0.0, rtol=0, atol=1e-10)
        )
        self.assertTrue(
            np.allclose(ops.grad[1] @ ones, 0.0, rtol=0, atol=1e-10)
        )


class TestGmlsConvergence(unittest.TestCase):
    def test_rates_on_smooth_field_orders_1_2_3(self):
        f_t, fx_t, fy_t = _smooth_field(_TARGETS)
        for order in (1, 2, 3):
            with self.subTest(poly_order=order):
                value_errors = []
                grad_errors = []
                for n in (20, 40, 80):
                    source = _source_cloud(n)
                    ops = build_gmls_operators(
                        source, _TARGETS, poly_order=order
                    )
                    f_source, _, _ = _smooth_field(source)
                    value_errors.append(
                        np.max(np.abs(ops.value @ f_source - f_t))
                    )
                    grad_x = ops.grad[0] @ f_source
                    grad_y = ops.grad[1] @ f_source
                    grad_errors.append(
                        np.max(np.hypot(grad_x - fx_t, grad_y - fy_t))
                    )
                # Average rate per halving of the point spacing.
                value_rate = np.log2(value_errors[0] / value_errors[-1]) / 2.0
                grad_rate = np.log2(grad_errors[0] / grad_errors[-1]) / 2.0
                # Value beats h^order (the O(h^{m+1}) edge); gradient one less.
                self.assertGreater(value_rate, order, f"value {value_rate}")
                self.assertGreater(grad_rate, order - 1, f"grad {grad_rate}")


class TestGmlsStructure(unittest.TestCase):
    def test_operator_shapes(self):
        source = _source_cloud()
        ops = build_gmls_operators(source, _TARGETS, poly_order=2)
        self.assertEqual(ops.value.shape, (_TARGETS.shape[0], source.shape[0]))
        self.assertEqual(len(ops.grad), 2)
        self.assertEqual(ops.dim, 2)

    def test_too_few_sources_raises(self):
        source = np.array([[0.0, 0.0], [1.0, 0.0]])  # 2 < num_basis = 6
        with self.assertRaises(ValueError):
            build_gmls_operators(source, _TARGETS, poly_order=2)


if __name__ == "__main__":
    unittest.main()
