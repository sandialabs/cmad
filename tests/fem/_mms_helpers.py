"""Manufactured-solution test helpers for the FE pipeline.

Problem-agnostic utilities shared across MMS regression tests:

- :func:`make_elastic_parameters` builds a
  :class:`cmad.parameters.parameters.Parameters` for the isotropic
  small-strain :class:`cmad.models.elastic.Elastic` model from a
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
from jax.tree_util import tree_map
from numpy.typing import NDArray
from sympy import Matrix, eye, lambdify

from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.interpolants import hex_linear, tet_linear
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.fem.quadrature import hex_quadrature, tet_quadrature
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
        [NDArray[np.floating] | JaxArray, float],
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
            coords: NDArray[np.floating] | JaxArray, _t: float,
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


def l2_h1_errors(
        fe_problem: FEProblem,
        U_solved: NDArray[np.floating],
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
    if fam == ElementFamily.HEX_LINEAR:
        norm_quad = hex_quadrature(degree=4)
        interpolant = hex_linear
    else:
        norm_quad = tet_quadrature(degree=4)
        interpolant = tet_linear

    nips = norm_quad.xi.shape[0]
    nnodes = mesh.connectivity.shape[1]
    N_ref = np.empty((nips, nnodes), dtype=np.float64)
    grad_N_ref = np.empty((nips, nnodes, 3), dtype=np.float64)
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
) -> tuple[float, float, int]:
    """Solve ``fe_problem`` at time ``t`` and measure errors.

    Builds a single-step :class:`FEState` seeded with ``U=0``, runs
    :func:`fe_newton_solve` once, and returns ``(L2, H1, n_iters)``.
    Callers that want to bound the iteration count (e.g. closed-form
    linear MMS expecting one Newton step) add their own assertion on
    the returned ``n_iters``.
    """
    state = FEState.from_problem(fe_problem)
    U_solved, n_iters, _, _ = fe_newton_solve(
        fe_problem, U_prev=state.U_at(0), t=t,
    )
    L2, H1 = l2_h1_errors(fe_problem, U_solved, u_exact, grad_u_exact)
    return L2, H1, n_iters
