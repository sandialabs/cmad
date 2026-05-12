"""Global Newton driver for the FE forward problem."""
from collections.abc import Mapping
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from numpy.typing import NDArray

from cmad.fem.assembly import assemble_global
from cmad.fem.fe_problem import FEProblem
from cmad.fem.sparse_solve import (
    _embedded_bc_enforce,
    cg_jax,
    gmres_jax,
    spsolve_jax,
)
from cmad.typing import JaxArray, Params


def _fe_newton_primal(
        fe_problem: FEProblem,
        params_by_block: Mapping[str, Params],
        U_prev: JaxArray,
        t: float,
        xi_prev_by_block: Mapping[str, JaxArray],
        max_iters: int,
        abs_tol: float,
        rel_tol: float,
        print_global_convergence: bool,
        linear_solver: str,
) -> JaxArray:
    """Forward Newton iteration: ``lax.while_loop`` + ``spsolve_jax``.

    Each body iteration assembles ``(K_bcoo, R)`` once, builds the
    embedded-BC tangent via :func:`_embedded_bc_enforce`, and solves
    ``K · dU = -r`` via :func:`spsolve_jax`. The convergence check is
    on ``r`` at the iterate ``U_n`` (computed alongside ``K`` in the
    same assembly), so each iter costs one ``assemble_global``.
    """
    dof_map = fe_problem.dof_map
    presc_idx = jnp.asarray(dof_map.prescribed_indices)
    presc_vals = jnp.asarray(dof_map.evaluate_prescribed_values(t))
    alpha = fe_problem.bc_diag_scale

    U_init = U_prev.at[presc_idx].set(presc_vals)

    _, R_init, _ = assemble_global(
        fe_problem, params_by_block, U_init, U_prev, t,
        xi_prev_by_block=xi_prev_by_block,
    )
    r_init = R_init.at[presc_idx].set(
        alpha * (U_init[presc_idx] - presc_vals)
    )
    R0 = jnp.maximum(jnp.linalg.norm(r_init), abs_tol)

    def cond(state):
        i, _, R_norm, R_norm_0 = state
        return (i < max_iters) & (R_norm >= abs_tol) & (
            R_norm >= rel_tol * R_norm_0
        )

    def body(state):
        i, U, _, R_norm_0 = state
        K_bcoo, R_assembled, _ = assemble_global(
            fe_problem, params_by_block, U, U_prev, t,
            xi_prev_by_block=xi_prev_by_block,
        )
        r = R_assembled.at[presc_idx].set(
            alpha * (U[presc_idx] - presc_vals)
        )
        R_norm = jnp.linalg.norm(r)
        if print_global_convergence:
            jax.debug.print(" > ({k}) Newton iteration", k=i + 1)
            jax.debug.print(
                " > absolute ||R|| = {abs_r:.6e}", abs_r=R_norm,
            )
            jax.debug.print(
                " > relative ||R|| = {rel_r:.6e}",
                rel_r=R_norm / R_norm_0,
            )
        K_data = _embedded_bc_enforce(
            K_bcoo, presc_idx, presc_diag_scale=alpha,
        )
        if linear_solver == "cg":
            dU = cg_jax(K_data, fe_problem.embedded_sparsity, -r)
        elif linear_solver == "gmres":
            dU = gmres_jax(K_data, fe_problem.embedded_sparsity, -r)
        elif linear_solver == "direct":
            dU = spsolve_jax(K_data, fe_problem.embedded_sparsity, -r)
        else:
            raise ValueError(
                f"unknown linear_solver {linear_solver!r}; "
                f"expected 'direct', 'cg', or 'gmres'"
            )
        U_new = U + dU
        return (i + 1, U_new, R_norm, R_norm_0)

    _, U_star, _, _ = lax.while_loop(
        cond, body, (0, U_init, R0, R0),
    )
    return U_star


def fe_newton_solve(
        fe_problem: FEProblem,
        params_by_block: Mapping[str, Params],
        U_prev: NDArray[np.floating] | JaxArray,
        t: float = 0.0,
        xi_prev_by_block: Mapping[str, NDArray[np.floating] | JaxArray]
        | None = None,
        max_iters: int = 20,
        abs_tol: float = 1e-10,
        rel_tol: float = 1e-10,
        print_global_convergence: bool = False,
        linear_solver: str = "direct",
) -> tuple[JaxArray, dict[str, JaxArray]]:
    """Quasi-static global Newton driver for the FE forward problem.

    Nonlinear convention: ``K = dR/dU`` is the tangent stiffness and
    ``R(U) = R_int(U) - F_ext`` is the residual (body force folded
    into ``R`` by the assembly — no separate ``F`` vector). Each
    Newton step solves ``K · dU = -r`` for the embedded-BC residual
    ``r`` defined as ``r[free] = R(U)[free]`` and
    ``r[prescribed] = α · (U[prescribed] - prescribed_vals(t))``
    with ``α = fe_problem.bc_diag_scale``.

    Forward iteration is :func:`jax.lax.while_loop` over Newton
    steps. Each body call assembles ``(K_bcoo, R)`` once, builds the
    embedded-BC tangent via
    :func:`cmad.fem.sparse_solve._embedded_bc_enforce` (symmetric
    form: prescribed rows AND columns zeroed, ``α`` on the
    prescribed diagonal — block-diagonal ``K_ff | α · I_P``), and
    solves ``K · dU = -r`` via
    :func:`cmad.fem.sparse_solve.spsolve_jax` (sparse direct via
    :func:`scipy.sparse.linalg.spsolve` through
    :func:`jax.pure_callback`).

    AD over the converged ``(U_star, xi_solved)`` is provided by an
    inner :func:`jax.custom_jvp` rule. The JVP rule is the IFT
    linear sensitivity equation
    ``K · U_star_dot = -∂r/∂(p) · p_dot`` solved through
    ``spsolve_jax``: the ``K``-side cotangent flows automatically
    via :func:`jax.lax.custom_linear_solve`'s VJP rule. JAX
    auto-transposes the JVP for :func:`jax.grad`; HVPs via
    forward-over-reverse re-invoke the JVP rule with non-zero
    ``K_data_dot``. ``custom_jvp`` (rather than ``custom_vjp``)
    keeps forward-mode AD available for HVPs and
    :func:`jax.hessian`.

    Initial iterate ``U_prev`` with prescribed dofs overwritten by
    ``DofMap.evaluate_prescribed_values(t)`` — quasi-static warm
    start: interior values from the previous step, boundary values
    at the current step's BC targets. Same driver handles
    homogeneous and non-homogeneous Dirichlet.

    ``params_by_block`` is required and threads explicitly through
    the assembly call chain — pass tracer-leaved per-block params
    for AD callers, or build via
    :func:`cmad.fem.assembly.params_by_block_from_models` for
    imperative callers.

    ``xi_prev_by_block`` is the previous time-step's converged xi
    keyed by COUPLED block; required when the FE problem has any
    COUPLED block, ignored otherwise. ``xi_prev`` stays fixed
    across global Newton iterations; the per-IP local Newton inside
    the COUPLED kernel re-solves for ``xi(U_iter, xi_prev)`` every
    iteration. Returned ``xi_solved_by_block`` is the converged
    value, re-extracted by one ``assemble_global`` call at
    ``U_star`` after the loop. Empty dict for CLOSED_FORM-only
    problems. A missing COUPLED-block entry surfaces as a
    ``ValueError`` from
    :func:`cmad.fem.assembly.assemble_element_block` on the first
    body iteration.

    Returns ``(U_solved, xi_solved_by_block)``. Outputs are JAX
    arrays.
    """
    U_prev_jax = jnp.asarray(U_prev, dtype=jnp.float64)
    xi_prev_jax: dict[str, JaxArray] = (
        {k: jnp.asarray(v) for k, v in xi_prev_by_block.items()}
        if xi_prev_by_block is not None else {}
    )
    return _fe_newton_solve_ad(
        fe_problem, params_by_block, U_prev_jax, t, xi_prev_jax,
        max_iters, abs_tol, rel_tol, print_global_convergence,
        linear_solver,
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 5, 6, 7, 8, 9))
def _fe_newton_solve_ad(
        fe_problem: FEProblem,
        params_by_block: Mapping[str, Params],
        U_prev: JaxArray,
        t: float,
        xi_prev_by_block: dict[str, JaxArray],
        max_iters: int,
        abs_tol: float,
        rel_tol: float,
        print_global_convergence: bool,
        linear_solver: str,
) -> tuple[JaxArray, dict[str, JaxArray]]:
    """AD-decorated inner driver. JaxArray inputs only.

    Splitting the public ``fe_newton_solve`` from this inner form
    keeps the boundary ``np.ndarray → jnp.ndarray`` conversion
    outside the ``custom_jvp``-tracked function body, so the diff
    args are uniformly typed for the JVP rule.
    """
    U_star = _fe_newton_primal(
        fe_problem, params_by_block, U_prev, t, xi_prev_by_block,
        max_iters, abs_tol, rel_tol, print_global_convergence,
        linear_solver,
    )
    _, _, xi_solved = assemble_global(
        fe_problem, params_by_block, U_star, U_prev, t,
        xi_prev_by_block=xi_prev_by_block,
    )
    return U_star, xi_solved


@_fe_newton_solve_ad.defjvp
def _fe_newton_solve_ad_jvp(
        fe_problem: FEProblem,
        max_iters: int, abs_tol: float, rel_tol: float,
        print_global_convergence: bool,
        linear_solver: str,
        primals, tangents,
):
    """IFT linear-sensitivity JVP for :func:`_fe_newton_solve_ad`.

    For ``r(U, p) = 0`` with ``p = (params, U_prev, t, xi_prev)``,
    ``U_star_dot = -K^{-1} · (∂r/∂p · p_dot)`` with ``K = ∂r/∂U`` at
    ``U_star``. ``∂r/∂p · p_dot`` is computed by
    ``jax.jvp(r, p, p_dot)`` at fixed ``U_star``; ``K`` is the
    ``_embedded_bc_enforce``-applied assembled tangent at ``U_star``;
    the linear solve goes through :func:`spsolve_jax` so the ``K``
    cotangent flows automatically when JAX auto-transposes the rule.
    ``xi_solved_dot`` follows from chain rule: the assembly's xi
    output is differentiated jointly w.r.t. ``U_star`` (with tangent
    ``U_star_dot``) and w.r.t. ``p`` (with tangent ``p_dot``).
    """
    params_by_block, U_prev, t, xi_prev_by_block = primals
    p_dot = tangents

    U_star, xi_solved = _fe_newton_solve_ad(
        fe_problem, params_by_block, U_prev, t, xi_prev_by_block,
        max_iters, abs_tol, rel_tol, print_global_convergence,
        linear_solver,
    )

    presc_idx = jnp.asarray(fe_problem.dof_map.prescribed_indices)
    alpha = fe_problem.bc_diag_scale

    def r_of_p(params_, Up_, t_, xp_):
        pv = jnp.asarray(
            fe_problem.dof_map.evaluate_prescribed_values(t_),
        )
        U_star_clamped = U_star.at[presc_idx].set(pv)
        _, R_local, _ = assemble_global(
            fe_problem, params_, U_star_clamped, Up_, t_,
            xi_prev_by_block=xp_,
        )
        return R_local.at[presc_idx].set(
            alpha * (U_star[presc_idx] - pv)
        )

    _, Rp_dot = jax.jvp(
        r_of_p, (params_by_block, U_prev, t, xi_prev_by_block), p_dot,
    )

    K_bcoo, _, _ = assemble_global(
        fe_problem, params_by_block, U_star, U_prev, t,
        xi_prev_by_block=xi_prev_by_block,
    )
    K_data = _embedded_bc_enforce(
        K_bcoo, presc_idx, presc_diag_scale=alpha,
    )

    if linear_solver == "cg":
        U_star_dot = cg_jax(K_data, fe_problem.embedded_sparsity, -Rp_dot)
    elif linear_solver == "gmres":
        U_star_dot = gmres_jax(
            K_data, fe_problem.embedded_sparsity, -Rp_dot,
        )
    elif linear_solver == "direct":
        U_star_dot = spsolve_jax(K_data, fe_problem.embedded_sparsity, -Rp_dot)
    else:
        raise ValueError(
            f"unknown linear_solver {linear_solver!r}; "
            f"expected 'direct', 'cg', or 'gmres'"
        )

    def xi_of_U_p(U_, params_, Up_, t_, xp_):
        _, _, xi_local = assemble_global(
            fe_problem, params_, U_, Up_, t_,
            xi_prev_by_block=xp_,
        )
        return xi_local

    _, xi_solved_dot = jax.jvp(
        xi_of_U_p,
        (U_star, params_by_block, U_prev, t, xi_prev_by_block),
        (U_star_dot, *p_dot),
    )

    primals_out = (U_star, xi_solved)
    tangents_out = (U_star_dot, xi_solved_dot)
    return primals_out, tangents_out
