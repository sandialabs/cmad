"""Monomial-exactness tests for 3D Gauss-Legendre quadrature rules.

Hex rules (``hex_quadrature(degree)``) are Gauss-Legendre tensor products
with per-coordinate exactness: every monomial ``x^a y^b z^c`` with
``max(a,b,c) <= degree`` is integrated exactly.

Tet rules (``tet_quadrature(degree)``) are Keast tables with total-degree
exactness: every monomial ``x^a y^b z^c`` with ``a + b + c <= degree`` is
integrated exactly on the unit simplex.

The analytical integrals are in closed form:
- Hex on [-1, 1]^3: the coordinatewise factorization
  ``integral x^n dx over [-1, 1] = 0 if n odd else 2 / (n + 1)``.
- Tet on the unit simplex: the Dirichlet integral
  ``integral x^a y^b z^c dV = a! b! c! / (a + b + c + 3)!``.
"""
import math
import unittest

import numpy as np

from cmad.fem.quadrature import hex_quadrature, tet_quadrature


def _hex_mono_int(n: int) -> float:
    """Analytic integral ``int_{-1}^{1} x^n dx``."""
    return 0.0 if n % 2 == 1 else 2.0 / (n + 1)


def _tet_mono_int(a: int, b: int, c: int) -> float:
    """Analytic integral ``int_tet x^a y^b z^c dV`` on the unit simplex."""
    return (
        math.factorial(a) * math.factorial(b) * math.factorial(c)
        / math.factorial(a + b + c + 3)
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


if __name__ == "__main__":
    unittest.main()
