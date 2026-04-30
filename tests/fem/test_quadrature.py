"""Monomial-exactness tests for Gauss-Legendre quadrature rules.

Hex rules (``hex_quadrature(degree)``) and quad rules
(``quad_quadrature(degree)``) are Gauss-Legendre tensor products with
per-coordinate exactness: every monomial ``x^a y^b (z^c)`` with
``max(a, b, ...) <= degree`` is integrated exactly.

Tet rules (``tet_quadrature(degree)``) and tri rules
(``tri_quadrature(degree)``) are tabulated total-degree-exact rules:
every monomial ``x^a y^b (z^c)`` with ``a + b (+ c) <= degree`` is
integrated exactly on the unit simplex / unit triangle.

The analytical integrals are in closed form:
- Hex on [-1, 1]^3 / quad on [-1, 1]^2: the coordinatewise
  factorization ``integral x^n dx over [-1, 1] = 0 if n odd else
  2 / (n + 1)``.
- Tet on the unit simplex: the Dirichlet integral
  ``integral x^a y^b z^c dV = a! b! c! / (a + b + c + 3)!``.
- Tri on the unit triangle: the 2D Dirichlet integral
  ``integral x^a y^b dA = a! b! / (a + b + 2)!``.
"""
import math
import unittest

import numpy as np

from cmad.fem.quadrature import (
    hex_quadrature,
    quad_quadrature,
    tet_quadrature,
    tri_quadrature,
)


def _hex_mono_int(n: int) -> float:
    """Analytic integral ``int_{-1}^{1} x^n dx``."""
    return 0.0 if n % 2 == 1 else 2.0 / (n + 1)


def _tet_mono_int(a: int, b: int, c: int) -> float:
    """Analytic integral ``int_tet x^a y^b z^c dV`` on the unit simplex."""
    return (
        math.factorial(a) * math.factorial(b) * math.factorial(c)
        / math.factorial(a + b + c + 3)
    )


def _tri_mono_int(a: int, b: int) -> float:
    """Analytic integral ``int_T x^a y^b dA`` on the unit triangle."""
    return (
        math.factorial(a) * math.factorial(b)
        / math.factorial(a + b + 2)
    )


class TestHexQuadrature(unittest.TestCase):
    _degrees = tuple(range(1, 8))

    def test_monomial_exactness(self):
        for degree in self._degrees:
            rule = hex_quadrature(degree)
            for a in range(degree + 1):
                for b in range(degree + 1):
                    for c in range(degree + 1):
                        analytical = (
                            _hex_mono_int(a)
                            * _hex_mono_int(b)
                            * _hex_mono_int(c)
                        )
                        numerical = float(np.sum(
                            rule.w
                            * rule.xi[:, 0]**a
                            * rule.xi[:, 1]**b
                            * rule.xi[:, 2]**c
                        ))
                        with self.subTest(degree=degree, mono=(a, b, c)):
                            self.assertAlmostEqual(
                                numerical, analytical, places=11)


class TestQuadQuadrature(unittest.TestCase):
    _degrees = tuple(range(1, 8))

    def test_monomial_exactness(self):
        for degree in self._degrees:
            rule = quad_quadrature(degree)
            for a in range(degree + 1):
                for b in range(degree + 1):
                    analytical = _hex_mono_int(a) * _hex_mono_int(b)
                    numerical = float(np.sum(
                        rule.w
                        * rule.xi[:, 0]**a
                        * rule.xi[:, 1]**b
                    ))
                    with self.subTest(degree=degree, mono=(a, b)):
                        self.assertAlmostEqual(
                            numerical, analytical, places=11)


class TestTetQuadrature(unittest.TestCase):
    _degrees = (1, 2, 3, 4, 5, 6)

    def test_monomial_exactness(self):
        for degree in self._degrees:
            rule = tet_quadrature(degree)
            for a in range(degree + 1):
                for b in range(degree + 1 - a):
                    for c in range(degree + 1 - a - b):
                        analytical = _tet_mono_int(a, b, c)
                        numerical = float(np.sum(
                            rule.w
                            * rule.xi[:, 0]**a
                            * rule.xi[:, 1]**b
                            * rule.xi[:, 2]**c
                        ))
                        with self.subTest(degree=degree, mono=(a, b, c)):
                            self.assertAlmostEqual(
                                numerical, analytical, places=11)


class TestTriQuadrature(unittest.TestCase):
    _degrees = (1, 2, 3, 4, 5, 6, 10)

    def test_monomial_exactness(self):
        for degree in self._degrees:
            rule = tri_quadrature(degree)
            for a in range(degree + 1):
                for b in range(degree + 1 - a):
                    analytical = _tri_mono_int(a, b)
                    numerical = float(np.sum(
                        rule.w
                        * rule.xi[:, 0]**a
                        * rule.xi[:, 1]**b
                    ))
                    with self.subTest(degree=degree, mono=(a, b)):
                        self.assertAlmostEqual(
                            numerical, analytical, places=11)

    def test_unsupported_degree_raises(self):
        for degree in (-1, 0, 7, 8, 9, 11):
            with self.subTest(degree=degree), self.assertRaises(ValueError):
                tri_quadrature(degree)


if __name__ == "__main__":
    unittest.main()
