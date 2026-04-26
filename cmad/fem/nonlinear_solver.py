"""Global Newton driver and BC-enforcement helpers for the FE forward problem."""
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from cmad.fem.assembly import assemble_global
from cmad.fem.fe_problem import FEProblem


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
        U_prev: NDArray[np.floating],
        t: float = 0.0,
        U_init: NDArray[np.floating] | None = None,
        max_iters: int = 20,
        abs_tol: float = 1e-10,
        rel_tol: float = 1e-10,
) -> tuple[NDArray[np.floating], int, float]:
    """Quasi-static global Newton driver for the FE forward problem.

    Each iteration assembles the global ``(K, R)`` via
    :func:`assemble_global`, applies Dirichlet boundary conditions by
    strong enforcement, and solves the sparse system
    ``K_csr @ dU = -R_enforced`` via direct factorization
    (``scipy.sparse.linalg.spsolve``). Returns
    ``(U_solved, n_iters, final_R_norm)``.

    The initial ``U`` defaults to zeros over all dofs and has its
    prescribed-dof entries overwritten with the BC target evaluated at
    ``t`` via ``DofMap.evaluate_prescribed_values(t)``.
    ``U[prescribed]`` stays at the BC target across iterations because
    strong enforcement zeros the corresponding rows of ``dU``
    (``dbc_residual = 0`` since ``U[prescribed] - bc_target = 0``
    after pre-loading) — the same driver pattern handles both
    homogeneous and non-homogeneous Dirichlet.

    For linear-elastic + closed-form Cauchy the residual is linear in
    U and Newton converges in one iteration; the loop is structured to
    accept multi-iteration convergence for nonlinear materials. The
    caller appends ``(U_solved, xi_by_block, t)`` into the
    :class:`FEState` history; this driver is stateless.
    """
    n_dofs = fe_problem.dof_map.num_total_dofs
    U = (
        np.zeros(n_dofs, dtype=np.float64)
        if U_init is None
        else U_init.astype(np.float64).copy()
    )
    prescribed_values = fe_problem.dof_map.evaluate_prescribed_values(t)
    U[fe_problem.dof_map.prescribed_indices] = prescribed_values

    R_norm_0: float = 1.0
    R_norm: float = 0.0
    dbc_residual = np.zeros(
        fe_problem.dof_map.num_prescribed_dofs, dtype=np.float64,
    )

    for it in range(max_iters):
        K_coo, R = assemble_global(fe_problem, U, U_prev, t)

        K_csr, R_enforced, _ = apply_strong_dirichlet(
            K_coo, R,
            fe_problem.dof_map.prescribed_indices,
            dbc_residual,
        )

        R_norm = float(np.linalg.norm(R_enforced))
        if it == 0:
            R_norm_0 = R_norm if R_norm > 0.0 else 1.0

        if R_norm < abs_tol or R_norm / R_norm_0 < rel_tol:
            return U, it, R_norm

        dU = np.asarray(
            scipy.sparse.linalg.spsolve(K_csr.tocsc(), -R_enforced),
        )
        U = U + dU

    raise RuntimeError(
        f"fe_newton_solve did not converge in {max_iters} iters; "
        f"final R_norm = {R_norm}"
    )
