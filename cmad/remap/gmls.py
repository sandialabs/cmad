"""GMLS reconstruction operators for point cloud to point cloud remap.

A 2D in-plane port of the core of Compadre's GMLS (Sandia's COMpatible
PArticle Discretization and REmap Toolkit, github.com/sandialabs/compadre):
per target point, a local weighted least squares polynomial fit yields
weights that reconstruct a field value and its gradient from nearby source
samples.

The construction is purely geometric: :func:`build_gmls_operators` returns
sparse matrices that, applied to source samples, give the reconstructed value
and gradient at the target points. For a field ``f`` sampled at the source
cloud, ``ops.value @ f`` is the remapped field and ``ops.grad[k] @ f`` is
``df/dx_k`` at the targets.

Per target the steps mirror Compadre: a search for the ``num_basis`` nearest
points sets a local radius, a ball search of ``support_multiplier`` times that
radius collects the stencil, and a weighted least squares polynomial fit
in monomials scaled by the radius (normalized by 1/alpha!, with a Power
weight) gives the stencil weights -- Compadre's "alphas" -- for the value (the constant coefficient)
and each first derivative (the linear coefficients divided by the radius).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import factorial

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr, solve_triangular
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GmlsOperators:
    """Sparse reconstruction operators from a source cloud to target points.

    ``value`` and each entry of ``grad`` are ``(num_target, num_source)``
    sparse matrices. Applied to a source-sampled field, ``value`` gives the
    reconstructed value at the targets and ``grad[k]`` gives ``df/dx_k``.
    """

    value: csr_matrix
    grad: tuple[csr_matrix, ...]
    poly_order: int
    dim: int


def _monomial_exponents(poly_order: int, dim: int) -> NDArray[np.int_]:
    """Exponent rows for a total degree monomial basis up to ``poly_order``.

    Degree 0 first (the constant), then degree 1 (the unit exponents in axis
    order), then higher degrees. So row 0 selects the point value and rows
    ``1..dim`` select the first derivatives.
    """
    rows: list[list[int]] = []
    for total_degree in range(poly_order + 1):
        for combo in combinations_with_replacement(range(dim), total_degree):
            alpha = [0] * dim
            for axis in combo:
                alpha[axis] += 1
            rows.append(alpha)
    return np.array(rows, dtype=int)


def build_gmls_operators(
        source_points: NDArray[np.floating],
        target_points: NDArray[np.floating],
        *,
        poly_order: int = 2,
        support_multiplier: float = 1.6,
        weight_power: int = 2,
) -> GmlsOperators:
    """Build GMLS value and gradient operators from source to target points.

    ``source_points`` ``(num_source, dim)`` and ``target_points``
    ``(num_target, dim)`` are reference coordinates with matching ``dim``.
    ``poly_order`` is the polynomial degree; ``support_multiplier``
    (Compadre's epsilon_multiplier) scales the local radius; ``weight_power``
    is the exponent of the Power weight ``(1 - (r/radius)^2)^weight_power``.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    num_source, dim = source_points.shape
    num_target = target_points.shape[0]
    if target_points.shape[1] != dim:
        raise ValueError(
            f"source/target dim mismatch: {dim} vs {target_points.shape[1]}"
        )

    exponents = _monomial_exponents(poly_order, dim)
    num_basis = exponents.shape[0]
    if num_source < num_basis:
        raise ValueError(
            f"need at least num_basis={num_basis} source points for "
            f"poly_order={poly_order} in {dim}D; got {num_source}"
        )
    inv_factorial = np.array(
        [1.0 / np.prod([factorial(a) for a in row]) for row in exponents]
    )
    # Functionals to extract: column 0 picks the point value (the constant
    # coefficient), columns 1..dim pick the first derivatives (the unit
    # exponent rows). These are columns of the identity on the basis.
    selectors = np.zeros((num_basis, dim + 1))
    selectors[np.arange(dim + 1), np.arange(dim + 1)] = 1.0

    tree = cKDTree(source_points)

    value_rows: list[int] = []
    value_cols: list[int] = []
    value_data: list[float] = []
    grad_rows: list[list[int]] = [[] for _ in range(dim)]
    grad_cols: list[list[int]] = [[] for _ in range(dim)]
    grad_data: list[list[float]] = [[] for _ in range(dim)]

    for target_index in range(num_target):
        center = target_points[target_index]
        distances, _ = tree.query(center, k=num_basis)
        radius = support_multiplier * float(np.atleast_1d(distances)[-1])
        neighbors = np.asarray(tree.query_ball_point(center, radius), dtype=int)

        local = (source_points[neighbors] - center) / radius
        basis = inv_factorial[None, :] * np.prod(
            local[:, None, :] ** exponents[None, :, :], axis=2
        )
        rho_squared = np.sum(local * local, axis=1)
        weight = np.clip(1.0 - rho_squared, 0.0, None) ** weight_power

        weighted_basis = np.sqrt(weight)[:, None] * basis
        _, upper = qr(weighted_basis, mode="economic")
        middle = solve_triangular(upper, selectors, trans="T")
        coeffs = solve_triangular(upper, middle)
        stencils = weight[:, None] * (basis @ coeffs)

        value_rows.extend([target_index] * neighbors.size)
        value_cols.extend(neighbors.tolist())
        value_data.extend(stencils[:, 0].tolist())
        for axis in range(dim):
            grad_rows[axis].extend([target_index] * neighbors.size)
            grad_cols[axis].extend(neighbors.tolist())
            grad_data[axis].extend((stencils[:, 1 + axis] / radius).tolist())

    shape = (num_target, num_source)
    value = csr_matrix((value_data, (value_rows, value_cols)), shape=shape)
    grad = tuple(
        csr_matrix(
            (grad_data[axis], (grad_rows[axis], grad_cols[axis])), shape=shape
        )
        for axis in range(dim)
    )
    return GmlsOperators(
        value=value, grad=grad, poly_order=poly_order, dim=dim
    )
