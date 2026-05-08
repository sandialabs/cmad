"""Global Newton driver and BC-enforcement helpers for the FE forward problem."""
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import scipy.sparse
from jax import jacfwd, lax
from numpy.typing import NDArray

from cmad.fem.assembly import assemble_global
from cmad.fem.fe_problem import FEProblem
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
) -> tuple[JaxArray, dict[str, JaxArray], int, JaxArray]:
    """Quasi-static global Newton driver for the FE forward problem.

    Nonlinear convention: ``K = dR/dU`` is the tangent stiffness and
    ``R(U) = R_int(U) - F_ext`` is the residual (body force folded
    into ``R`` by the assembly — no separate ``F`` vector). Each
    Newton step solves ``K · dU = -R``.

    Wraps the Newton iteration in :func:`jax.lax.custom_root` so the
    solve is JAX-traceable end to end. Three pieces close over
    ``fe_problem`` / ``params_by_block`` / ``U_prev`` / ``t`` /
    ``xi_prev_by_block``:

    - ``residual(U)``: the embedded-BC residual that's zero at the
      converged iterate. ``residual[free] = R(U)[free]``;
      ``residual[prescribed] = U[prescribed] - prescribed_vals(t)``
      (scale=1 hardcoded — the prescribed block of
      ``dresidual/dU`` is decoupled and the prescribed targets are
      parameter-independent at fixed ``t``, so ``dU*/dp`` at
      prescribed dofs is scale-invariant; hardcoding 1 keeps ``K``
      out of ``residual``'s data path).
    - ``solve``: imperative ``lax.while_loop`` over Newton
      iterations. Body assembles ``(R, K)``, applies strong DBC
      inline (zero prescribed rows + cols, scaled identity on the
      prescribed diagonal, zero ``R_enf[prescribed]``), and solves
      ``K_enf @ dU = -R_enf`` via dense ``jnp.linalg.solve``. The
      prescribed-diagonal scale is ``mean(|diag(K)|[free])`` so
      enforcement entries match the magnitude of the unprescribed
      diagonal — same shape as :func:`apply_strong_dirichlet` (the
      scipy-numpy helper used by any imperative call site).
    - ``tangent_solve``: dense
      ``jacfwd(lin_residual)(zeros) → jnp.linalg.solve`` for the
      IFT inversion that ``lax.custom_root`` invokes for both
      forward-mode and reverse-mode AD.

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
    ``U_star`` after the root solve. Empty dict for CLOSED_FORM-
    only problems. A missing COUPLED-block entry surfaces as a
    ``ValueError`` from
    :func:`cmad.fem.assembly.assemble_element_block` on the first
    body iteration.

    Returns ``(U_solved, xi_solved_by_block, n_iters, R_norm)``.
    Outputs are JAX arrays. ``n_iters`` is ``-1`` (the
    ``lax.custom_root`` ``solve`` callable's loop counter doesn't
    escape its boundary); concrete callers that previously logged
    iteration counts should fall back to ``R_norm`` for
    convergence reporting.
    """
    dof_map = fe_problem.dof_map
    prescribed_indices = jnp.asarray(dof_map.prescribed_indices)
    prescribed_vals = jnp.asarray(
        dof_map.evaluate_prescribed_values(t),
    )

    U_prev_jax = jnp.asarray(U_prev, dtype=jnp.float64)
    xi_prev_jax = (
        {k: jnp.asarray(v) for k, v in xi_prev_by_block.items()}
        if xi_prev_by_block is not None else None
    )

    def residual(U):
        _, R, _ = assemble_global(
            fe_problem, params_by_block, U, U_prev_jax, t,
            xi_prev_by_block=xi_prev_jax,
        )
        return R.at[prescribed_indices].set(
            U[prescribed_indices] - prescribed_vals,
        )

    def solve(residual_fn, U_init):
        def cond(state):
            i, _, R_norm, R_norm_0 = state
            return (i < max_iters) & (R_norm >= abs_tol) & (
                R_norm >= rel_tol * R_norm_0
            )

        def body(state):
            i, U, _, R_norm_0 = state
            K_bcoo, R, _ = assemble_global(
                fe_problem, params_by_block, U, U_prev_jax, t,
                xi_prev_by_block=xi_prev_jax,
            )
            K = K_bcoo.todense()

            n = K.shape[0]
            p_mask = jnp.zeros(n, bool).at[prescribed_indices].set(True)
            free_mask = (~p_mask).astype(K.dtype)
            scale = (jnp.sum(jnp.abs(jnp.diag(K)) * free_mask)
                     / jnp.sum(free_mask))
            K_enf = jnp.where(
                p_mask[:, None] | p_mask[None, :], 0.0, K,
            )
            K_enf = K_enf.at[
                prescribed_indices, prescribed_indices,
            ].set(scale)
            R_enf = R.at[prescribed_indices].set(0.0)

            dU = jnp.linalg.solve(K_enf, -R_enf)
            U_new = U + dU
            return (
                i + 1, U_new,
                jnp.linalg.norm(residual_fn(U_new)), R_norm_0,
            )

        R0 = jnp.maximum(jnp.linalg.norm(residual_fn(U_init)), 1.0)
        _, U_star, _, _ = lax.while_loop(
            cond, body, (0, U_init, R0, R0),
        )
        return U_star

    def tangent_solve(lin_residual, y):
        K_lin = jacfwd(lin_residual)(jnp.zeros_like(y))
        return jnp.linalg.solve(K_lin, y)

    U_init = U_prev_jax.at[prescribed_indices].set(prescribed_vals)
    U_star = lax.custom_root(residual, U_init, solve, tangent_solve)

    _, R_final, xi_solved_by_block = assemble_global(
        fe_problem, params_by_block, U_star, U_prev_jax, t,
        xi_prev_by_block=xi_prev_jax,
    )
    R_norm_final = jnp.linalg.norm(
        R_final.at[prescribed_indices].set(0.0),
    )
    return U_star, xi_solved_by_block, -1, R_norm_final
