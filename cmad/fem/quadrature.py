"""Gauss-Legendre quadrature rules on reference finite elements.

Factories return a :class:`QuadratureRule` exact for polynomials up to
the requested degree. :func:`hex_quadrature` and
:func:`quad_quadrature` are Gauss-Legendre tensor products on
[-1, 1]^d (per-coordinate exactness). :func:`tet_quadrature` and
:func:`tri_quadrature` use tabulated rules on the unit simplex /
unit triangle (total-degree exactness).

Quadrature data is stored as numpy arrays (static configuration; not
traced). Callers pass IP coordinates through interpolants in
:mod:`cmad.fem.interpolants`, which lift to JAX via implicit
``jnp.asarray``.

Tet rules are P. Keast's tabulated quadratures from "Moderate-degree
tetrahedral quadrature formulas" (Computer Methods in Applied
Mechanics and Engineering, 55, 339-348, 1986), transcribed from the
add_fem branch. Degrees 3 and 4 each carry one negative weight at
the tet centroid — the rules still integrate polynomials exactly to
their stated degree, but negative weights can degrade mass-matrix
positivity and nonlinear-iteration stability depending on the
integrand. Use degrees 5 or 6 when positive-weight rules are
required.

Tri rules: degrees 1, 2, 4, 5, 6, 10 are Dunavant 1985 ("High degree
efficient symmetrical Gaussian quadrature rules for the triangle",
IJNME 21(6), 1129-1148), transcribed from the add_fem branch — all
positive weights, cyclically symmetric. Degree 3 is the Hammer-Stroud
4-point rule (Hammer, Marlowe, Stroud, "Numerical integration over
simplexes and cones", Math. Tables Aids Comp. 10, 130-137, 1956),
which has one negative weight at the centroid; use degree 4 (same
point count, all-positive weights) when positive-weight rules are
required.
"""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre


@dataclass(frozen=True)
class QuadratureRule:
    """A quadrature rule on a reference finite element.

    ``xi`` has shape ``(npts, ref_dim)`` — reference-element
    integration-point coordinates, where ``ref_dim`` is the topological
    dimension of the reference element (1 for line, 2 for tri/quad,
    3 for tet/hex). ``w`` has shape ``(npts,)`` — corresponding
    weights. Both are numpy arrays; static configuration (not traced).
    """
    xi: NDArray[np.floating]
    w: NDArray[np.floating]


def gauss_legendre_1d(
        n_points: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """1D Gauss-Legendre rule on [-1, 1] with ``n_points`` points.

    Returns ``(xi_1d, w_1d)`` flat numpy arrays of length ``n_points``.
    Exact for polynomials of degree ``<= 2 * n_points - 1``; weights
    sum to 2 (interval length). Building block for tensor-product
    factories on Cartesian-product reference elements.
    """
    xi_1d, w_1d = roots_legendre(n_points)
    return np.asarray(xi_1d), np.asarray(w_1d)


def hex_quadrature(degree: int) -> QuadratureRule:
    """Gauss-Legendre tensor-product rule on the reference hex [-1,1]^3.

    Exact for polynomials in (x,y,z) with each coordinate exponent
    <= ``degree`` (per-direction exactness, not total-degree).
    Internally chooses ``n = ceil((degree + 1) / 2)`` points per
    direction, for a total of ``n ** 3`` integration points.

    Raises :class:`ValueError` if ``degree < 1``.
    """
    if degree < 1:
        raise ValueError(
            f"hex_quadrature requires degree >= 1; got degree={degree}"
        )
    n = int(np.ceil((degree + 1) / 2))
    xi_1d, w_1d = gauss_legendre_1d(n)
    xi_mesh = np.stack(
        np.meshgrid(xi_1d, xi_1d, xi_1d, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    w_mesh = (
        w_1d[:, None, None] * w_1d[None, :, None] * w_1d[None, None, :]
    ).reshape(-1)
    return QuadratureRule(xi=xi_mesh, w=w_mesh)


def quad_quadrature(degree: int) -> QuadratureRule:
    """Gauss-Legendre tensor-product rule on the reference quad [-1,1]^2.

    Exact for polynomials in (x,y) with each coordinate exponent
    <= ``degree`` (per-direction exactness, not total-degree).
    Internally chooses ``n = ceil((degree + 1) / 2)`` points per
    direction, for a total of ``n ** 2`` integration points. Weights
    sum to 4 (area of [-1,1]^2).

    Raises :class:`ValueError` if ``degree < 1``.
    """
    if degree < 1:
        raise ValueError(
            f"quad_quadrature requires degree >= 1; got degree={degree}"
        )
    n = int(np.ceil((degree + 1) / 2))
    xi_1d, w_1d = gauss_legendre_1d(n)
    xi_mesh = np.stack(
        np.meshgrid(xi_1d, xi_1d, indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    w_mesh = (w_1d[:, None] * w_1d[None, :]).reshape(-1)
    return QuadratureRule(xi=xi_mesh, w=w_mesh)


# ---- Keast tet tables (degrees 1..6) ---------------------------------------
#
# Transcribed verbatim from add_fem's
# cmad/fem_utils/quadrature/quadrature_rule.py. Degrees 3 and 4 each
# carry one negative weight at the tet centroid; see module docstring
# for caveats on application.

_TET_XI_1 = np.array([[0.25, 0.25, 0.25]])
_TET_W_1 = np.array([1.0 / 6.0])

_TET_XI_2 = np.array([
    [0.138196601125011, 0.138196601125011, 0.138196601125011],
    [0.585410196624969, 0.138196601125011, 0.138196601125011],
    [0.138196601125011, 0.585410196624969, 0.138196601125011],
    [0.138196601125011, 0.138196601125011, 0.585410196624969],
])
_TET_W_2 = np.array([0.25 / 6.0, 0.25 / 6.0, 0.25 / 6.0, 0.25 / 6.0])

_TET_XI_3 = np.array([
    [1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0],
    [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0],
    [1.0 / 6.0, 1.0 / 2.0, 1.0 / 6.0],
    [1.0 / 2.0, 1.0 / 6.0, 1.0 / 6.0],
])
_TET_W_3 = np.array([
    -2.0 / 15.0,
    3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0,
])

_TET_XI_4 = np.array([
    [0.25, 0.25, 0.25],
    [0.785714285714286, 0.0714285714285714, 0.0714285714285714],
    [0.0714285714285714, 0.0714285714285714, 0.0714285714285714],
    [0.0714285714285714, 0.0714285714285714, 0.785714285714286],
    [0.0714285714285714, 0.785714285714286, 0.0714285714285714],
    [0.100596423833201, 0.399403576166799, 0.399403576166799],
    [0.399403576166799, 0.100596423833201, 0.399403576166799],
    [0.399403576166799, 0.399403576166799, 0.100596423833201],
    [0.399403576166799, 0.100596423833201, 0.100596423833201],
    [0.100596423833201, 0.399403576166799, 0.100596423833201],
    [0.100596423833201, 0.100596423833201, 0.399403576166799],
])
_TET_W_4 = np.array([
    -0.0131555555555556,
    0.00762222222222222, 0.00762222222222222,
    0.00762222222222222, 0.00762222222222222,
    0.0248888888888889, 0.0248888888888889, 0.0248888888888889,
    0.0248888888888889, 0.0248888888888889, 0.0248888888888889,
])

_TET_XI_5 = np.array([
    [0.25, 0.25, 0.25],
    [0.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 0.0],
    [1.0 / 3.0, 0.0, 1.0 / 3.0],
    [8.0 / 11.0, 1.0 / 11.0, 1.0 / 11.0],
    [1.0 / 11.0, 1.0 / 11.0, 1.0 / 11.0],
    [1.0 / 11.0, 1.0 / 11.0, 8.0 / 11.0],
    [1.0 / 11.0, 8.0 / 11.0, 1.0 / 11.0],
    [0.433449846426336, 0.0665501535736643, 0.0665501535736643],
    [0.0665501535736643, 0.433449846426336, 0.0665501535736643],
    [0.0665501535736643, 0.0665501535736643, 0.433449846426336],
    [0.0665501535736643, 0.433449846426336, 0.433449846426336],
    [0.433449846426336, 0.0665501535736643, 0.433449846426336],
    [0.433449846426336, 0.433449846426336, 0.0665501535736643],
])
_TET_W_5 = np.array([
    0.0302836780970892,
    0.00602678571428572, 0.00602678571428572,
    0.00602678571428572, 0.00602678571428572,
    0.011645249086029, 0.011645249086029,
    0.011645249086029, 0.011645249086029,
    0.0109491415613865, 0.0109491415613865,
    0.0109491415613865, 0.0109491415613865,
    0.0109491415613865, 0.0109491415613865,
])

_TET_XI_6 = np.array([
    [0.356191386222545, 0.214602871259152, 0.214602871259152],
    [0.214602871259152, 0.214602871259152, 0.214602871259152],
    [0.214602871259152, 0.214602871259152, 0.356191386222545],
    [0.214602871259152, 0.356191386222545, 0.214602871259152],
    [0.877978124396166, 0.0406739585346113, 0.0406739585346113],
    [0.0406739585346113, 0.0406739585346113, 0.0406739585346113],
    [0.0406739585346113, 0.0406739585346113, 0.877978124396166],
    [0.0406739585346113, 0.877978124396166, 0.0406739585346113],
    [0.0329863295731731, 0.322337890142276, 0.322337890142276],
    [0.322337890142276, 0.322337890142276, 0.322337890142276],
    [0.322337890142276, 0.322337890142276, 0.0329863295731731],
    [0.322337890142276, 0.0329863295731731, 0.322337890142276],
    [0.269672331458316, 0.0636610018750175, 0.0636610018750175],
    [0.0636610018750175, 0.269672331458316, 0.0636610018750175],
    [0.0636610018750175, 0.0636610018750175, 0.269672331458316],
    [0.603005664791649, 0.0636610018750175, 0.0636610018750175],
    [0.0636610018750175, 0.603005664791649, 0.0636610018750175],
    [0.0636610018750175, 0.0636610018750175, 0.603005664791649],
    [0.0636610018750175, 0.269672331458316, 0.603005664791649],
    [0.269672331458316, 0.603005664791649, 0.0636610018750175],
    [0.603005664791649, 0.0636610018750175, 0.269672331458316],
    [0.0636610018750175, 0.603005664791649, 0.269672331458316],
    [0.269672331458316, 0.0636610018750175, 0.603005664791649],
    [0.603005664791649, 0.269672331458316, 0.0636610018750175],
])
_TET_W_6 = np.array([
    0.00665379170969465, 0.00665379170969465,
    0.00665379170969465, 0.00665379170969465,
    0.00167953517588678, 0.00167953517588678,
    0.00167953517588678, 0.00167953517588678,
    0.0092261969239424, 0.0092261969239424,
    0.0092261969239424, 0.0092261969239424,
    0.00803571428571428, 0.00803571428571428,
    0.00803571428571428, 0.00803571428571428,
    0.00803571428571428, 0.00803571428571428,
    0.00803571428571428, 0.00803571428571428,
    0.00803571428571428, 0.00803571428571428,
    0.00803571428571428, 0.00803571428571428,
])

_TET_TABLES: dict[int, tuple[NDArray[np.floating], NDArray[np.floating]]] = {
    1: (_TET_XI_1, _TET_W_1),
    2: (_TET_XI_2, _TET_W_2),
    3: (_TET_XI_3, _TET_W_3),
    4: (_TET_XI_4, _TET_W_4),
    5: (_TET_XI_5, _TET_W_5),
    6: (_TET_XI_6, _TET_W_6),
}


def tet_quadrature(degree: int) -> QuadratureRule:
    """Keast quadrature rule on the unit simplex.

    Exact for polynomials of total degree <= ``degree``. Supported
    degrees: 1, 2, 3, 4, 5, 6.

    Caveat: degrees 3 and 4 each carry one negative weight at the tet
    centroid (degree 3: -2/15 ~= -0.133; degree 4: ~= -0.0131). The
    rules still integrate polynomials exactly to their stated degree,
    but negative weights can degrade mass-matrix positivity and
    nonlinear-iteration stability depending on the integrand. Use
    degrees 5 or 6 if positive-weight rules are required.

    Raises :class:`ValueError` if ``degree`` is outside 1..6.
    """
    if degree not in _TET_TABLES:
        raise ValueError(
            f"tet_quadrature supports degrees 1..6; got degree={degree}"
        )
    xi, w = _TET_TABLES[degree]
    return QuadratureRule(xi=xi, w=w)


# ---- Tri tables (Dunavant 1985 + Hammer-Stroud degree 3) -------------------
#
# Domain is the unit triangle with vertices (0, 0)-(1, 0)-(0, 1); weights
# sum to 1/2 (area). Dunavant rules (degrees 1, 2, 4, 5, 6, 10) transcribed
# verbatim from add_fem's cmad/fem_utils/quadrature/quadrature_rule.py
# entries for those degrees — all positive weights, cyclically symmetric.
# Degree 3 is the Hammer-Stroud 4-point rule with one negative weight at
# the centroid (-27/96); Dunavant 1985 deliberately omits degree-3 rules
# because no symmetric positive-weight 3- or 4-point rule exists, so the
# degree-3 entry is independently transcribed from Hammer-Stroud 1956.

_TRI_XI_1 = np.array([[1.0 / 3.0, 1.0 / 3.0]])
_TRI_W_1 = np.array([0.5])

_TRI_XI_2 = np.array([
    [2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0],
    [1.0 / 6.0, 1.0 / 6.0],
])
_TRI_W_2 = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])

_TRI_XI_3 = np.array([
    [1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 5.0, 1.0 / 5.0],
    [3.0 / 5.0, 1.0 / 5.0],
    [1.0 / 5.0, 3.0 / 5.0],
])
_TRI_W_3 = np.array([
    -27.0 / 96.0,
    25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0,
])

_TRI_XI_4 = np.array([
    [0.108103018168070, 0.445948490915965],
    [0.445948490915965, 0.108103018168070],
    [0.445948490915965, 0.445948490915965],
    [0.816847572980459, 0.091576213509771],
    [0.091576213509771, 0.816847572980459],
    [0.091576213509771, 0.091576213509771],
])
_TRI_W_4 = np.array([
    0.111690794839005, 0.111690794839005, 0.111690794839005,
    0.054975871827661, 0.054975871827661, 0.054975871827661,
])

_TRI_XI_5 = np.array([
    [0.333333333333333, 0.333333333333333],
    [0.059715871789770, 0.470142064105115],
    [0.470142064105115, 0.059715871789770],
    [0.470142064105115, 0.470142064105115],
    [0.797426985353087, 0.101286507323456],
    [0.101286507323456, 0.797426985353087],
    [0.101286507323456, 0.101286507323456],
])
_TRI_W_5 = np.array([
    0.112500000000000,
    0.066197076394253, 0.066197076394253, 0.066197076394253,
    0.062969590272414, 0.062969590272414, 0.062969590272414,
])

_TRI_XI_6 = np.array([
    [0.501426509658179, 0.249286745170910],
    [0.249286745170910, 0.501426509658179],
    [0.249286745170910, 0.249286745170910],
    [0.873821971016996, 0.063089014491502],
    [0.063089014491502, 0.873821971016996],
    [0.063089014491502, 0.063089014491502],
    [0.053145049844817, 0.310352451033784],
    [0.636502499121399, 0.053145049844817],
    [0.310352451033784, 0.636502499121399],
    [0.053145049844817, 0.636502499121399],
    [0.636502499121399, 0.310352451033784],
    [0.310352451033784, 0.053145049844817],
])
_TRI_W_6 = np.array([
    0.058393137863189, 0.058393137863189, 0.058393137863189,
    0.025422453185104, 0.025422453185104, 0.025422453185104,
    0.041425537809187, 0.041425537809187, 0.041425537809187,
    0.041425537809187, 0.041425537809187, 0.041425537809187,
])

_TRI_XI_10 = np.array([
    [0.333333333333333, 0.333333333333333],
    [0.004269134091050, 0.497865432954475],
    [0.497865432954475, 0.004269134091050],
    [0.497865432954475, 0.497865432954475],
    [0.143975100541888, 0.428012449729056],
    [0.428012449729056, 0.143975100541888],
    [0.428012449729056, 0.428012449729056],
    [0.630487174513551, 0.184756412743225],
    [0.184756412743225, 0.630487174513551],
    [0.184756412743225, 0.184756412743225],
    [0.959037562856645, 0.020481218571678],
    [0.020481218571678, 0.959037562856645],
    [0.020481218571678, 0.020481218571678],
    [0.035002989897272, 0.136573576256033],
    [0.136573576256033, 0.828423433846695],
    [0.828423433846695, 0.035002989897272],
    [0.136573576256033, 0.035002989897272],
    [0.828423433846695, 0.136573576256033],
    [0.035002989897272, 0.828423433846695],
    [0.037549070258443, 0.332743600588639],
    [0.332743600588639, 0.629707329152919],
    [0.629707329152919, 0.037549070258443],
    [0.332743600588639, 0.037549070258443],
    [0.629707329152919, 0.332743600588639],
    [0.037549070258443, 0.629707329152919],
])
_TRI_W_10 = np.array([
    0.041761699902598,
    0.003614925296028, 0.003614925296028, 0.003614925296028,
    0.037246088960490, 0.037246088960490, 0.037246088960490,
    0.039323236701554, 0.039323236701554, 0.039323236701554,
    0.003464161543554, 0.003464161543554, 0.003464161543554,
    0.014759160167390, 0.014759160167390, 0.014759160167390,
    0.014759160167390, 0.014759160167390, 0.014759160167390,
    0.019789683598031, 0.019789683598031, 0.019789683598031,
    0.019789683598031, 0.019789683598031, 0.019789683598031,
])

_TRI_TABLES: dict[int, tuple[NDArray[np.floating], NDArray[np.floating]]] = {
    1: (_TRI_XI_1, _TRI_W_1),
    2: (_TRI_XI_2, _TRI_W_2),
    3: (_TRI_XI_3, _TRI_W_3),
    4: (_TRI_XI_4, _TRI_W_4),
    5: (_TRI_XI_5, _TRI_W_5),
    6: (_TRI_XI_6, _TRI_W_6),
    10: (_TRI_XI_10, _TRI_W_10),
}


def tri_quadrature(degree: int) -> QuadratureRule:
    """Symmetric quadrature rule on the unit triangle.

    Domain: unit triangle with vertices (0,0)-(1,0)-(0,1). Exact for
    polynomials of total degree <= ``degree``. Supported degrees:
    1, 2, 3, 4, 5, 6, 10. Degrees 7-9 are not provided — the next
    exact rule available in the source tables is degree 10.

    Caveat: degree 3 carries one negative weight at the triangle
    centroid (-27/96 ~= -0.281). The rule still integrates polynomials
    exactly to total degree 3, but negative weights can degrade mass-
    matrix positivity and nonlinear-iteration stability depending on
    the integrand. Use degree 4 (six points, all positive) when
    positive-weight rules are required.

    Raises :class:`ValueError` if ``degree`` is not in
    {1, 2, 3, 4, 5, 6, 10}.
    """
    if degree not in _TRI_TABLES:
        raise ValueError(
            f"tri_quadrature supports degrees 1..6 and 10; "
            f"got degree={degree}"
        )
    xi, w = _TRI_TABLES[degree]
    return QuadratureRule(xi=xi, w=w)
