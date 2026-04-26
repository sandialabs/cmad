"""Global Newton driver and BC-enforcement helpers for the FE forward problem."""
import numpy as np
import scipy.sparse
from numpy.typing import NDArray


def apply_strong_dirichlet(
        K_coo: scipy.sparse.coo_matrix,
        R: NDArray[np.floating],
        prescribed_indices: NDArray[np.intp],
        prescribed_values: NDArray[np.floating],
        scale: float | NDArray[np.floating] | None = None,
) -> tuple[scipy.sparse.csr_matrix, NDArray[np.floating], float | NDArray[np.floating]]:
    """Strong-enforce Dirichlet BCs by zeroing prescribed rows + columns.

    Rebuilds K's COO triplets with prescribed-row-or-column entries
    dropped and scaled-identity entries appended at the prescribed
    diagonal; sets ``R[prescribed] = scale * prescribed_values``.

    Homogeneous Dirichlet: pass ``prescribed_values = 0``. The enforced
    residual at prescribed dofs is zero, the Newton update at those
    dofs is zero, and ``U[prescribed]`` is preserved at the BC target
    across iterations (provided the caller initialized it there).
    Non-homogeneous Dirichlet: pass
    ``prescribed_values = U[prescribed] - bc_target`` to drive the
    Newton update toward the BC target.

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
    R_enforced[p] = diag_vals * np.asarray(prescribed_values).astype(R.dtype)

    return K_csr, R_enforced, scale_resolved
