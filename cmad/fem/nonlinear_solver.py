"""Global Newton driver and BC-enforcement helpers for the FE forward problem."""
from collections.abc import Mapping
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
from jax import lax
from numpy.typing import NDArray

from cmad.fem.assembly import assemble_global
from cmad.fem.fe_problem import FEProblem
from cmad.fem.sparse_solve import _embedded_bc_enforce, spsolve_jax
from cmad.typing import JaxArray, Params


def apply_strong_dirichlet(
        K_coo: scipy.sparse.coo_matrix,
        R: NDArray[np.floating],
        prescribed_indices: NDArray[np.intp],
        dbc_residual: NDArray[np.floating],
        scale: float | NDArray[np.floating] | None = None,
) -> tuple[scipy.sparse.csr_matrix, NDArray[np.floating], float | NDArray[np.floating]]:
    """Strong-enforce Dirichlet BCs by zeroing prescribed rows + columns.

    Rebuilds K's COO triplets with prescribed-row-or-column entries
    dropped and scaled-identity entries appended at the prescribed
    diagonal; sets ``R[prescribed] = scale * dbc_residual``.

    ``dbc_residual`` is the residual the caller wants at each DBC dof
    after enforcement, *not* the BC target value itself. Semantically
    it is ``U[prescribed] - bc_target`` for the Newton formulation
    (with ``dU[prescribed] = -dbc_residual`` from the linear solve,
    so one Newton update drives ``U[prescribed]`` toward the target).

    Homogeneous DBC with U pre-loaded to the BC target: pass
    ``dbc_residual = 0``. The Newton update at prescribed dofs is
    zero and ``U[prescribed]`` is preserved at the target across
    iterations.
    Non-homogeneous DBC without pre-loading: pass
    ``dbc_residual = U[prescribed] - bc_target`` to drive the
    Newton update toward the target.

    ``scale`` defaults to ``mean(|diag(K)|[unprescribed])`` — single
    scalar with the same units as the unprescribed diagonal, so the
    prescribed-diagonal entries don't pollute the spectrum under
    iterative solvers (CG, AMG, field-split). Multi-physics decks may
    pass a per-prescribed-dof ``(P,)`` array (e.g. with per-block
    means) for finer scaling.

    Symmetric (rows-and-columns) form, kept for imperative scipy
    call sites and for the future iterative-solver path. The traced
    Newton driver in :func:`fe_newton_solve` instead uses
    :func:`cmad.fem.sparse_solve._embedded_bc_enforce`'s asymmetric
    (rows-only, identity 1.0) form, which suffices for sparse direct
    and matches ``∂r/∂U`` exactly for the embedded-BC residual the
    JVP rule needs.
    """
    n = R.shape[0]
    rows, cols, vals = K_coo.row, K_coo.col, K_coo.data
    p = np.asarray(prescribed_indices, dtype=np.intp)

    scale_resolved: float | NDArray[np.floating]
    if scale is None:
        diag = np.zeros(n, dtype=np.float64)
        diag_mask = (rows == cols)
        np.add.at(diag, rows[diag_mask], vals[diag_mask])
        unpresc = np.ones(n, dtype=bool)
        unpresc[prescribed_indices] = False
        scale_resolved = float(np.mean(np.abs(diag[unpresc])))
    else:
        scale_resolved = scale

    if isinstance(scale_resolved, np.ndarray):
        diag_vals = scale_resolved.astype(np.float64)
    else:
        diag_vals = np.full(p.shape[0], float(scale_resolved), dtype=np.float64)

    is_presc_row = np.isin(rows, p)
    is_presc_col = np.isin(cols, p)
    keep = ~(is_presc_row | is_presc_col)

    rows_out = np.concatenate([rows[keep], p])
    cols_out = np.concatenate([cols[keep], p])
    vals_out = np.concatenate([vals[keep], diag_vals])

    K_csr = scipy.sparse.coo_matrix(
        (vals_out, (rows_out, cols_out)), shape=(n, n),
    ).tocsr()

    R_enforced = R.copy()
    R_enforced[p] = diag_vals * np.asarray(dbc_residual).astype(R.dtype)

    return K_csr, R_enforced, scale_resolved


def _fe_newton_primal(
        fe_problem: FEProblem,
        params_by_block: Mapping[str, Params],
        U_prev: JaxArray,
        t: float,
        xi_prev_by_block: Mapping[str, JaxArray],
        max_iters: int,
        abs_tol: float,
        rel_tol: float,
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

    U_init = U_prev.at[presc_idx].set(presc_vals)

    _, R_init, _ = assemble_global(
        fe_problem, params_by_block, U_init, U_prev, t,
        xi_prev_by_block=xi_prev_by_block,
    )
    r_init = R_init.at[presc_idx].set(U_init[presc_idx] - presc_vals)
    R0 = jnp.maximum(jnp.linalg.norm(r_init), 1.0)

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
        r = R_assembled.at[presc_idx].set(U[presc_idx] - presc_vals)
        K_data, K_rows, K_cols = _embedded_bc_enforce(K_bcoo, presc_idx)
        n = K_bcoo.shape[0]
        dU = spsolve_jax(K_data, K_rows, K_cols, n, -r)
        U_new = U + dU
        return (i + 1, U_new, jnp.linalg.norm(r), R_norm_0)

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
) -> tuple[JaxArray, dict[str, JaxArray]]:
    """Quasi-static global Newton driver for the FE forward problem.

    Nonlinear convention: ``K = dR/dU`` is the tangent stiffness and
    ``R(U) = R_int(U) - F_ext`` is the residual (body force folded
    into ``R`` by the assembly — no separate ``F`` vector). Each
    Newton step solves ``K · dU = -r`` for the embedded-BC residual
    ``r`` defined as ``r[free] = R(U)[free]`` and
    ``r[prescribed] = U[prescribed] - prescribed_vals(t)``.

    Forward iteration is :func:`jax.lax.while_loop` over Newton
    steps. Each body call assembles ``(K_bcoo, R)`` once, builds the
    embedded-BC tangent via
    :func:`cmad.fem.sparse_solve._embedded_bc_enforce` (asymmetric
    form: prescribed rows zeroed, identity 1.0 on the prescribed
    diagonal), and solves ``K · dU = -r`` via
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
        max_iters, abs_tol, rel_tol,
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 5, 6, 7))
def _fe_newton_solve_ad(
        fe_problem: FEProblem,
        params_by_block: Mapping[str, Params],
        U_prev: JaxArray,
        t: float,
        xi_prev_by_block: dict[str, JaxArray],
        max_iters: int,
        abs_tol: float,
        rel_tol: float,
) -> tuple[JaxArray, dict[str, JaxArray]]:
    """AD-decorated inner driver. JaxArray inputs only.

    Splitting the public ``fe_newton_solve`` from this inner form
    keeps the boundary ``np.ndarray → jnp.ndarray`` conversion
    outside the ``custom_jvp``-tracked function body, so the diff
    args are uniformly typed for the JVP rule.
    """
    U_star = _fe_newton_primal(
        fe_problem, params_by_block, U_prev, t, xi_prev_by_block,
        max_iters, abs_tol, rel_tol,
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
        max_iters, abs_tol, rel_tol,
    )

    presc_idx = jnp.asarray(fe_problem.dof_map.prescribed_indices)

    def r_of_p(params_, Up_, t_, xp_):
        _, R_local, _ = assemble_global(
            fe_problem, params_, U_star, Up_, t_,
            xi_prev_by_block=xp_,
        )
        pv = jnp.asarray(
            fe_problem.dof_map.evaluate_prescribed_values(t_),
        )
        return R_local.at[presc_idx].set(U_star[presc_idx] - pv)

    _, Rp_dot = jax.jvp(
        r_of_p, (params_by_block, U_prev, t, xi_prev_by_block), p_dot,
    )

    K_bcoo, _, _ = assemble_global(
        fe_problem, params_by_block, U_star, U_prev, t,
        xi_prev_by_block=xi_prev_by_block,
    )
    K_data, K_rows, K_cols = _embedded_bc_enforce(K_bcoo, presc_idx)
    n = K_bcoo.shape[0]

    U_star_dot = spsolve_jax(K_data, K_rows, K_cols, n, -Rp_dot)

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
