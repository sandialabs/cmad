"""Reference-space shape functions for linear hex, tet, quad, and tri elements.

Single-IP API: each function returns a single :class:`ShapeFunctionsAtIP`
with reference-frame gradient ``grad_N[a, j] = ∂N_a/∂ξ_j``. Callers
wanting batched evaluation use :func:`jax.vmap`; callers wanting
physical-frame gradients apply the element's isoparametric Jacobian
(``iso_jac``) at the assembly layer.
"""
import jax.numpy as jnp

from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.typing import JaxArray

_HEX_NODE_XI = jnp.array([
    [-1.0, -1.0, -1.0],
    [+1.0, -1.0, -1.0],
    [+1.0, +1.0, -1.0],
    [-1.0, +1.0, -1.0],
    [-1.0, -1.0, +1.0],
    [+1.0, -1.0, +1.0],
    [+1.0, +1.0, +1.0],
    [-1.0, +1.0, +1.0],
])


def hex_linear(xi: JaxArray) -> ShapeFunctionsAtIP:
    """Reference-space trilinear shape functions on [-1,1]³ at one IP.

    Node ordering: (-,-,-), (+,-,-), (+,+,-), (-,+,-),
    (-,-,+), (+,-,+), (+,+,+), (-,+,+). Matches add_fem's
    shape_brick ordering.

    ``xi`` has shape ``(3,)`` — reference-space coordinates in [-1, 1]³.

    Returns ``N`` of shape ``(8,)`` and ``grad_N`` of shape ``(8, 3)``
    with ``grad_N[a, j] = ∂N_a/∂ξ_j``. Reference-frame gradient;
    physical-frame transformation happens at the assembly layer via
    the element's ``iso_jac``.
    """
    # N_a(ξ) = (1/8) Π_k (1 + ξ_k ξ_{k,a}), where ξ_{k,a} is node a's
    # k-th reference coordinate.
    terms = 1.0 + xi * _HEX_NODE_XI              # (8, 3)
    N = jnp.prod(terms, axis=1) / 8.0            # (8,)
    # ∂N_a/∂ξ_j = (1/8) ξ_{j,a} Π_{k≠j} (1 + ξ_k ξ_{k,a}).
    # Enumerated in 3D explicitly to avoid the prod-except-j division
    # trick, which is unsafe at element corners where a term vanishes.
    grad_N = jnp.stack(
        [
            _HEX_NODE_XI[:, 0] * terms[:, 1] * terms[:, 2],
            _HEX_NODE_XI[:, 1] * terms[:, 0] * terms[:, 2],
            _HEX_NODE_XI[:, 2] * terms[:, 0] * terms[:, 1],
        ],
        axis=1,
    ) / 8.0                                      # (8, 3)
    return ShapeFunctionsAtIP(N=N, grad_N=grad_N)


def tet_linear(xi: JaxArray) -> ShapeFunctionsAtIP:
    """Reference-space linear shape functions on the unit simplex at one IP.

    Node ordering: (0,0,0), (1,0,0), (0,1,0), (0,0,1). Matches
    add_fem's shape_tetrahedron ordering and the existing
    :class:`GlobalResidual` toy-test fixture.

    ``xi`` has shape ``(3,)`` — reference-space coordinates in the
    unit simplex ``{(ξ₁, ξ₂, ξ₃) : ξᵢ ≥ 0, ξ₁ + ξ₂ + ξ₃ ≤ 1}``.

    Returns ``N`` of shape ``(4,)`` and ``grad_N`` of shape ``(4, 3)``
    with ``grad_N[a, j] = ∂N_a/∂ξ_j``. Reference-frame gradient;
    constant over the element for linear tet (the ``xi`` argument is
    accepted for API uniformity with :func:`hex_linear`). Physical-
    frame transformation at the assembly layer via the element's
    ``iso_jac``.
    """
    N = jnp.array([1.0 - xi[0] - xi[1] - xi[2], xi[0], xi[1], xi[2]])
    grad_N = jnp.array([
        [-1.0, -1.0, -1.0],
        [+1.0, 0.0, 0.0],
        [0.0, +1.0, 0.0],
        [0.0, 0.0, +1.0],
    ])
    return ShapeFunctionsAtIP(N=N, grad_N=grad_N)


_QUAD_NODE_XI = jnp.array([
    [-1.0, -1.0],
    [+1.0, -1.0],
    [+1.0, +1.0],
    [-1.0, +1.0],
])


def quad_linear(xi: JaxArray) -> ShapeFunctionsAtIP:
    """Reference-space bilinear shape functions on [-1,1]² at one IP.

    Node ordering: (-,-), (+,-), (+,+), (-,+) — CCW from (-1, -1),
    matching the bottom-face ordering of :func:`hex_linear`.

    ``xi`` has shape ``(2,)`` — reference-space coordinates in [-1, 1]².

    Returns ``N`` of shape ``(4,)`` and ``grad_N`` of shape ``(4, 2)``
    with ``grad_N[a, j] = ∂N_a/∂ξ_j``. Reference-frame gradient;
    physical-frame transformation happens at the assembly layer via
    the element's ``iso_jac``.
    """
    # N_a(ξ) = (1/4) (1 + ξ_0 ξ_{0,a}) (1 + ξ_1 ξ_{1,a}), where ξ_{k,a}
    # is node a's k-th reference coordinate.
    terms = 1.0 + xi * _QUAD_NODE_XI              # (4, 2)
    N = jnp.prod(terms, axis=1) / 4.0             # (4,)
    # ∂N_a/∂ξ_0 = (1/4) ξ_{0,a} (1 + ξ_1 ξ_{1,a}); ∂N_a/∂ξ_1 the same
    # with the two directions swapped.
    grad_N = jnp.stack(
        [
            _QUAD_NODE_XI[:, 0] * terms[:, 1],
            _QUAD_NODE_XI[:, 1] * terms[:, 0],
        ],
        axis=1,
    ) / 4.0                                       # (4, 2)
    return ShapeFunctionsAtIP(N=N, grad_N=grad_N)


def tri_linear(xi: JaxArray) -> ShapeFunctionsAtIP:
    """Reference-space linear shape functions on the unit triangle at one IP.

    Node ordering: (0,0), (1,0), (0,1) — the (origin, +x, +y) ordering
    of :func:`tet_linear`'s first three nodes, on the unit triangle used
    by :func:`cmad.fem.quadrature.tri_quadrature`.

    ``xi`` has shape ``(2,)`` — reference-space coordinates in the unit
    triangle ``{(ξ₁, ξ₂) : ξᵢ ≥ 0, ξ₁ + ξ₂ ≤ 1}``.

    Returns ``N`` of shape ``(3,)`` and ``grad_N`` of shape ``(3, 2)``
    with ``grad_N[a, j] = ∂N_a/∂ξ_j``. Reference-frame gradient,
    constant over the element (the ``xi`` argument is accepted for API
    uniformity with :func:`quad_linear`). Physical-frame transformation
    at the assembly layer via the element's ``iso_jac``.
    """
    N = jnp.array([1.0 - xi[0] - xi[1], xi[0], xi[1]])
    grad_N = jnp.array([
        [-1.0, -1.0],
        [+1.0, 0.0],
        [0.0, +1.0],
    ])
    return ShapeFunctionsAtIP(N=N, grad_N=grad_N)
