"""Manufactured-solution test helpers for the FE pipeline.

Problem-agnostic utilities shared across MMS regression tests:

- :func:`make_elastic_parameters` builds a
  :class:`cmad.parameters.parameters.Parameters` for the isotropic
  small strain :class:`cmad.models.elastic.Elastic` model from a
  ``(kappa, mu)`` pair.
- :func:`build_mms_callables` lambdifies a sympy ``u_sym`` plus the
  matching ``b = -div(sigma(u_sym))`` and exposes the symbolic
  ``sigma_sym`` so callers can derive boundary tractions or other
  stress-derived quantities from the same manufactured solution.
- :func:`l2_h1_errors` measures L²/H¹ errors against a callable
  ``u_exact`` via degree-4 isoparametric quadrature.
- :func:`solve_and_measure` runs :func:`cmad.fem.nonlinear_solver.fe_newton_solve`
  once at the requested time and returns the error pair plus the
  converged iteration count.

Underscore-prefixed names in the originating
``test_mms_cube_3d.py`` lose the leading underscore on export
since they're now public helpers within the test package.
"""
from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
import sympy
from jax import jacfwd
from jax.tree_util import tree_map
from numpy.typing import NDArray
from sympy import Matrix, eye, lambdify

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.interpolants import (
    hex_linear,
    quad_linear,
    tet_linear,
    tri_linear,
)
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.fem.quadrature import (
    hex_quadrature,
    quad_quadrature,
    tet_quadrature,
    tri_quadrature,
)
from cmad.models.elastic_stress import (
    compressible_neohookean_cauchy_stress,
    isotropic_linear_elastic_cauchy_stress,
)
from cmad.models.kinematics import cofactor
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray, Params


def make_elastic_parameters(kappa: float, mu: float) -> Parameters:
    """Build a Parameters tree for ``cmad.models.elastic.Elastic``.

    All entries are flagged active; transforms default to identity.
    """
    values: Params = {"elastic": {"kappa": kappa, "mu": mu}}
    active_flags = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Parameters(values, active_flags, transforms)


def build_mms_callables(
        u_sym: sympy.Matrix,
        coord_syms: Sequence[Any],
        kappa: float,
        mu: float,
) -> tuple[
    Callable[
        [NDArray[np.floating] | JaxArray, float | JaxArray],
        NDArray[np.floating] | JaxArray,
    ],
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
    sympy.Matrix,
]:
    """Lambdify ``(body_force, u_exact, grad_u_exact)`` plus return ``sigma_sym``.

    Computes ``sigma = kappa * tr(eps) * I + 2 * mu * dev(eps)`` to
    match :func:`cmad.models.elastic_stress.isotropic_linear_elastic_cauchy_stress`,
    derives the matching body force ``b = -div(sigma(u_sym))``, and
    returns the four-tuple ``(body_force_fn, u_exact, grad_u_exact,
    sigma_sym)``. The body-force callable lambdifies into ``jax`` so
    the FE pipeline's vmap/jit can trace it; the exact-solution
    callables lambdify into ``numpy`` since they're invoked from the
    python error-norm loop. ``sigma_sym`` is exposed so callers can
    re-use the symbolic stress tensor (e.g. to derive boundary
    tractions ``sigma·n̂`` for surface-flux MMS tests).

    ``coord_syms`` is the spatial-symbol sequence; its length sets
    the ambient dimension. ``u_sym`` must be an ndim-by-1 sympy
    Matrix in those symbols.
    """
    n = len(coord_syms)
    grad_u_sym = u_sym.jacobian(list(coord_syms))
    eps_sym = (grad_u_sym + grad_u_sym.T) / 2
    tr_eps = eps_sym.trace()
    dev_eps_sym = eps_sym - (tr_eps / n) * eye(n)
    sigma_sym = kappa * tr_eps * eye(n) + 2 * mu * dev_eps_sym
    b_sym = sympy.simplify(Matrix([
        -sum(sigma_sym[i, j].diff(coord_syms[j]) for j in range(n))
        for i in range(n)
    ]))

    coord_args = tuple(coord_syms)
    b_callable = lambdify(coord_args, b_sym, modules="jax")
    u_callable = lambdify(coord_args, u_sym, modules="numpy")
    grad_u_callable = lambdify(coord_args, grad_u_sym, modules="numpy")

    def body_force_fn(
            coords: NDArray[np.floating] | JaxArray,
            _t: float | JaxArray,
    ) -> NDArray[np.floating] | JaxArray:
        args = tuple(coords[i] for i in range(n))
        return jnp.asarray(b_callable(*args)).reshape(-1)

    def u_exact(coords: NDArray[np.floating]) -> NDArray[np.floating]:
        args = tuple(coords[i] for i in range(n))
        return np.asarray(u_callable(*args)).reshape(-1)

    def grad_u_exact(coords: NDArray[np.floating]) -> NDArray[np.floating]:
        args = tuple(coords[i] for i in range(n))
        return np.asarray(grad_u_callable(*args))

    return body_force_fn, u_exact, grad_u_exact, sigma_sym


def build_finite_mms_callables(
        u_sym: sympy.Matrix,
        coord_syms: Sequence[Any],
        kappa: float,
        mu: float,
) -> tuple[
    Callable[
        [NDArray[np.floating] | JaxArray, float | JaxArray],
        NDArray[np.floating] | JaxArray,
    ],
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
]:
    """Finite deformation counterpart of :func:`build_mms_callables`.

    Body force ``b = -Div_X(P)`` for the first Piola-Kirchhoff stress
    ``P = sigma @ cofactor(F)`` with sigma from
    ``compressible_neohookean_cauchy_stress`` at ``F = I + grad u``. P and
    its divergence come from JAX autodiff of the exact solution, so the
    source matches the model's constitutive exactly (no symbolic inverse).
    Returns ``(body_force_fn, u_exact, grad_u_exact)``.
    """
    n = len(coord_syms)
    coord_args = tuple(coord_syms)
    grad_u_sym = u_sym.jacobian(list(coord_syms))
    u_jax = lambdify(coord_args, u_sym, modules="jax")
    u_callable = lambdify(coord_args, u_sym, modules="numpy")
    grad_u_callable = lambdify(coord_args, grad_u_sym, modules="numpy")
    params: Params = {"elastic": {"kappa": kappa, "mu": mu}}

    def u_of_X(X: JaxArray) -> JaxArray:
        return jnp.asarray(u_jax(*[X[i] for i in range(n)])).reshape(n)

    def pk1_of_X(X: JaxArray) -> JaxArray:
        F = jnp.eye(n) + jacfwd(u_of_X)(X)
        sigma = compressible_neohookean_cauchy_stress(F, params)
        return sigma @ cofactor(F)

    def body_force_fn(
            coords: NDArray[np.floating] | JaxArray,
            _t: float | JaxArray,
    ) -> NDArray[np.floating] | JaxArray:
        dP = jacfwd(pk1_of_X)(jnp.asarray(coords))
        return -jnp.einsum("iJJ->i", dP)

    def u_exact(coords: NDArray[np.floating]) -> NDArray[np.floating]:
        args = tuple(coords[i] for i in range(n))
        return np.asarray(u_callable(*args)).reshape(-1)

    def grad_u_exact(coords: NDArray[np.floating]) -> NDArray[np.floating]:
        args = tuple(coords[i] for i in range(n))
        return np.asarray(grad_u_callable(*args))

    return body_force_fn, u_exact, grad_u_exact


def build_plane_strain_mms_callables(
        u_sym: sympy.Matrix,
        coord_syms: Sequence[Any],
        kappa: float,
        mu: float,
) -> tuple[
    Callable[
        [NDArray[np.floating] | JaxArray, float | JaxArray],
        NDArray[np.floating] | JaxArray,
    ],
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
]:
    """Plane strain MMS source matching ``Elastic(PLANE_STRAIN)``.

    The body force is ``b = -div(sigma)``, where sigma is the 2x2 block
    of the isotropic linear elastic stress at ``F = I + grad u`` with the
    out of plane stretch set to 1 -- the same block the GR contracts.
    Built by autodiff of ``u_sym`` so it matches the model exactly.
    Returns ``(body_force_fn, u_exact, grad_u_exact)``; ``coord_syms``
    has length 2.
    """
    n = len(coord_syms)
    coord_args = tuple(coord_syms)
    grad_u_sym = u_sym.jacobian(list(coord_syms))
    u_jax = lambdify(coord_args, u_sym, modules="jax")
    u_callable = lambdify(coord_args, u_sym, modules="numpy")
    grad_u_callable = lambdify(coord_args, grad_u_sym, modules="numpy")
    params: Params = {"elastic": {"kappa": kappa, "mu": mu}}

    def u_of_X(X: JaxArray) -> JaxArray:
        return jnp.asarray(u_jax(*[X[i] for i in range(n)])).reshape(n)

    def sigma_in_plane_of_X(X: JaxArray) -> JaxArray:
        grad_u = jacfwd(u_of_X)(X)                       # (2, 2)
        F_2D = jnp.eye(2) + grad_u
        F = jnp.block([
            [F_2D, jnp.zeros((2, 1))],
            [jnp.zeros((1, 2)), jnp.ones((1, 1))],
        ])
        sigma = isotropic_linear_elastic_cauchy_stress(F, params)   # (3, 3)
        return sigma[:2, :2]

    def body_force_fn(
            coords: NDArray[np.floating] | JaxArray,
            _t: float | JaxArray,
    ) -> NDArray[np.floating] | JaxArray:
        # b_i = -d(sigma_ij)/dx_j.
        dsigma = jacfwd(sigma_in_plane_of_X)(jnp.asarray(coords))   # (2,2,2)
        return -jnp.einsum("ijj->i", dsigma)

    def u_exact(coords: NDArray[np.floating]) -> NDArray[np.floating]:
        args = tuple(coords[i] for i in range(n))
        return np.asarray(u_callable(*args)).reshape(-1)

    def grad_u_exact(coords: NDArray[np.floating]) -> NDArray[np.floating]:
        args = tuple(coords[i] for i in range(n))
        return np.asarray(grad_u_callable(*args))

    return body_force_fn, u_exact, grad_u_exact


def l2_h1_errors(
        fe_problem: FEProblem,
        U_solved: NDArray[np.floating] | JaxArray,
        u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        grad_u_exact: Callable[
            [NDArray[np.floating]], NDArray[np.floating],
        ],
) -> tuple[float, float]:
    """L² and H¹ errors against ``u_exact`` via degree-4 quadrature.

    Pure-numpy element-loop (test-only path, no jit). Reference-frame
    shape functions are precomputed once per quadrature rule; the
    isoparametric Jacobian and physical-frame gradient are evaluated
    with numpy for each ``(elem, ip)`` pair.
    """
    mesh = fe_problem.mesh
    dof_map = fe_problem.dof_map
    fam = mesh.element_family
    ndims = mesh.nodes.shape[1]
    if fam == ElementFamily.HEX_LINEAR:
        norm_quad = hex_quadrature(degree=4)
        interpolant = hex_linear
    elif fam == ElementFamily.TET_LINEAR:
        norm_quad = tet_quadrature(degree=4)
        interpolant = tet_linear
    elif fam == ElementFamily.QUAD_LINEAR:
        norm_quad = quad_quadrature(degree=4)
        interpolant = quad_linear
    else:
        norm_quad = tri_quadrature(degree=4)
        interpolant = tri_linear

    nips = norm_quad.xi.shape[0]
    nnodes = mesh.connectivity.shape[1]
    N_ref = np.empty((nips, nnodes), dtype=np.float64)
    grad_N_ref = np.empty((nips, nnodes, ndims), dtype=np.float64)
    for ip in range(nips):
        sh = interpolant(jnp.asarray(norm_quad.xi[ip]))
        N_ref[ip] = np.asarray(sh.N)
        grad_N_ref[ip] = np.asarray(sh.grad_N)
    w = np.asarray(norm_quad.w, dtype=np.float64)

    block_offset = dof_map.block_offsets[0]
    ndofs = dof_map.num_dofs_per_basis_fn[0]

    L2_sq = 0.0
    H1_grad_sq = 0.0

    for elem_idx in range(mesh.connectivity.shape[0]):
        node_ids = mesh.connectivity[elem_idx]
        X_elem = mesh.nodes[node_ids]
        eq = (
            block_offset
            + node_ids[:, None] * ndofs
            + np.arange(ndofs)[None, :]
        ).ravel()
        U_elem = U_solved[eq].reshape(-1, ndofs)

        for ip in range(nips):
            iso_jac = X_elem.T @ grad_N_ref[ip]
            iso_jac_det = float(np.linalg.det(iso_jac))
            grad_N_phys = grad_N_ref[ip] @ np.linalg.inv(iso_jac)
            dv = iso_jac_det * float(w[ip])

            coords_ip = N_ref[ip] @ X_elem
            u_h_ip = N_ref[ip] @ U_elem
            grad_u_h_ip = U_elem.T @ grad_N_phys

            u_ex_ip = u_exact(coords_ip)
            grad_u_ex_ip = grad_u_exact(coords_ip)

            L2_sq += float(np.sum((u_h_ip - u_ex_ip) ** 2)) * dv
            H1_grad_sq += float(
                np.sum((grad_u_h_ip - grad_u_ex_ip) ** 2),
            ) * dv

    return float(np.sqrt(L2_sq)), float(np.sqrt(L2_sq + H1_grad_sq))


def solve_and_measure(
        fe_problem: FEProblem,
        u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        grad_u_exact: Callable[
            [NDArray[np.floating]], NDArray[np.floating],
        ],
        t: float = 1.0,
) -> tuple[float, float]:
    """Solve ``fe_problem`` at time ``t`` and measure errors.

    Builds a single-step :class:`FEState` seeded with ``U=0``, runs
    :func:`fe_newton_solve` once, and returns ``(L2, H1)``.
    """
    state = FEState.from_problem(fe_problem)
    params_by_block = params_by_block_from_models(fe_problem)
    U_solved, _ = fe_newton_solve(
        fe_problem, params_by_block,
        U_prev=state.U_at(0), t=t,
    )
    L2, H1 = l2_h1_errors(fe_problem, U_solved, u_exact, grad_u_exact)
    return L2, H1
