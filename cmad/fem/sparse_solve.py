"""Sparse direct solve and embedded-BC enforcement for the FE Newton driver.

Two helpers:

- :func:`spsolve_jax` solves ``K x = b`` via
  :func:`scipy.sparse.linalg.spsolve` through :func:`jax.pure_callback`,
  with full forward-mode JVP and reverse-mode VJP supplied by
  :func:`jax.lax.custom_linear_solve`. ``K`` is given as COO triplets
  ``(K_data, K_rows, K_cols)`` of shape ``(n, n)``.

- :func:`_embedded_bc_enforce` rewrites a global :class:`BCOO`
  tangent ``K`` for the embedded-BC formulation: prescribed rows
  zeroed (off-diagonal entries included), identity entries appended
  on the prescribed diagonal. Used by both the Newton body and the
  ``@custom_jvp`` rule of :func:`fe_newton_solve` so the matvec /
  solve closures are structurally consistent.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from jax import lax
from jax.experimental.sparse import BCOO, BCSR

from cmad.typing import JaxArray

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem


def _spsolve_callback(
        K_data: JaxArray, K_rows: JaxArray, K_cols: JaxArray,
        n: int, b: JaxArray,
) -> JaxArray:
    """Forward-only :func:`jax.pure_callback` into scipy.

    ``n`` must be a Python int (matrix size, statically known —
    captured into the callback closure). The callback has no AD
    rules; AD on :func:`spsolve_jax` is the responsibility of
    :func:`jax.lax.custom_linear_solve`.

    ``vmap_method="sequential"`` makes the vmap behavior explicit:
    a sparse direct solve has no batched form, so a vmap over RHSs
    (e.g. column-by-column Hessian build) loops sequentially —
    correct and visible at the call site.
    """
    def _scipy_spsolve(
            K_data_np: np.ndarray, K_rows_np: np.ndarray,
            K_cols_np: np.ndarray, b_np: np.ndarray,
    ) -> np.ndarray:
        K_csr = scipy.sparse.coo_matrix(
            (K_data_np, (K_rows_np, K_cols_np)), shape=(n, n),
        ).tocsr()
        return np.asarray(scipy.sparse.linalg.spsolve(K_csr, b_np))

    return jax.pure_callback(
        _scipy_spsolve,
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        K_data, K_rows, K_cols, b,
        vmap_method="sequential",
    )


def spsolve_jax(
        K_data: JaxArray, K_rows: JaxArray, K_cols: JaxArray,
        n: int, b: JaxArray,
) -> JaxArray:
    """Solve ``K x = b`` for sparse COO ``K`` with full JAX AD support.

    Forward solve goes through :func:`scipy.sparse.linalg.spsolve` via
    :func:`jax.pure_callback`. AD rules are supplied by
    :func:`jax.lax.custom_linear_solve` from the ``(matvec, solve,
    transpose_solve)`` triple:

    - **Forward-mode JVP.** ``x_dot = solve(b_dot - matvec_dot(x))``
      where ``matvec_dot(x)`` is JAX's JVP of ``matvec`` at ``x``
      along ``K_data_dot`` — exactly the textbook ``K_dot · x``
      sensitivity term, computed by JAX through plain forward-mode
      AD on the COO scatter-add.
    - **Reverse-mode VJP.** ``λ = transpose_solve(x_bar)``;
      ``b_bar = λ``; the ``K_data`` cotangent comes from
      ``jax.vjp(matvec, x)(-λ)`` which evaluates to
      ``-λ[K_rows] · x[K_cols]`` on ``K_data`` — the canonical
      sparse VJP formula.

    Composes for HVPs via forward-over-reverse: the outer JVP through
    :func:`jax.grad` re-enters this primitive's JVP rule with a non-
    zero ``K_data_dot`` (because ``K_data`` carries tangents from
    the upstream params), and JAX's auto-transposition handles the
    rest.

    ``K^T`` in COO is ``K_rows ↔ K_cols`` swapped; passed to scipy
    as a fresh COO and converted to CSR via ``.tocsr()`` (which sums
    duplicate ``(row, col)`` entries during conversion if the
    embedded-BC zeroing leaves any in place).

    A direct ``@custom_jvp`` over the ``pure_callback`` would not
    auto-transpose: ``pure_callback`` has no transpose rule, and
    JAX's auto-transposition tracks operation linearity rather than
    argument semantics — it can't infer that swapping ``K_rows`` and
    ``K_cols`` realizes ``K^T``. ``lax.custom_linear_solve`` factors
    that cleanly: ``matvec`` declares the operator, ``transpose_solve``
    declares the adjoint, and JAX's built-in JVP / VJP rules cover
    any ``(matvec, solve, transpose_solve)`` triple.
    """

    def matvec(x: JaxArray) -> JaxArray:
        return jnp.zeros_like(x).at[K_rows].add(K_data * x[K_cols])

    def solve(_unused_matvec, rhs: JaxArray) -> JaxArray:
        return _spsolve_callback(K_data, K_rows, K_cols, n, rhs)

    def transpose_solve(_unused_vecmat, rhs: JaxArray) -> JaxArray:
        return _spsolve_callback(K_data, K_cols, K_rows, n, rhs)

    return lax.custom_linear_solve(
        matvec, b, solve, transpose_solve=transpose_solve,
        symmetric=False,
    )


def _pcg_loop(
        matvec: Callable[[JaxArray], JaxArray],
        b: JaxArray,
        precon: Callable[[JaxArray], JaxArray],
        rtol: float,
        max_iters: int | None,
) -> tuple[JaxArray, JaxArray]:
    """Preconditioned CG via manual ``lax.while_loop`` with iter count.

    Same Hestenes-Stiefel algorithm as
    :func:`jax.scipy.sparse.linalg.cg`, restructured to expose the
    iteration counter through the loop carry. Convergence test on
    the unpreconditioned residual:
    ``|r|^2 <= rtol^2 * |b|^2``.

    Used by :func:`cg_jax_with_iters` to expose the counter
    through its return tuple. :func:`cg_jax` defers to
    :func:`jax.scipy.sparse.linalg.cg` so CMAD inherits whatever
    upstream JAX does. Iter count from this loop may differ from
    the JAX-native CG by ±1 on convergence-test rounding;
    acceptable for diagnostic purposes.

    ``max_iters=None`` selects the
    :func:`jax.scipy.sparse.linalg.cg` default of
    ``10 * b.shape[0]``.
    """
    if max_iters is None:
        max_iters = 10 * b.shape[0]

    x0 = jnp.zeros_like(b)
    r0 = b - matvec(x0)
    z0 = precon(r0)
    p0 = z0
    rz0 = jnp.dot(r0, z0)
    tol_sq = (rtol ** 2) * jnp.dot(b, b)

    def cond(state: tuple) -> JaxArray:
        i, _x, r, _z, _p, _rz = state
        return (i < max_iters) & (jnp.dot(r, r) > tol_sq)

    def body(state: tuple) -> tuple:
        i, x, r, _z, p, rz = state
        Ap = matvec(p)
        alpha = rz / jnp.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = precon(r_new)
        rz_new = jnp.dot(r_new, z_new)
        beta = rz_new / rz
        p_new = z_new + beta * p
        return (i + 1, x_new, r_new, z_new, p_new, rz_new)

    initial = (jnp.int32(0), x0, r0, z0, p0, rz0)
    final = lax.while_loop(cond, body, initial)
    i_final, x_final = final[0], final[1]
    return x_final, i_final


def _cg_jax_operator(
        K_data: JaxArray, sparsity: EmbeddedSparsity,
) -> tuple[
    Callable[[JaxArray], JaxArray],
    Callable[[JaxArray], JaxArray],
]:
    """Build the BCSR matvec + Jacobi preconditioner for ``cg_jax``.

    Shared setup between :func:`cg_jax` and
    :func:`cg_jax_with_iters`; centralizing the dedup + BCSR
    construction + diagonal extraction keeps both callers' matvec
    and preconditioner definitions in lockstep even though their
    inner CG implementations differ.
    """
    unique_data = jnp.zeros(
        sparsity.num_unique, dtype=K_data.dtype,
    ).at[sparsity.segment_ids].add(K_data[sparsity.perm])

    K_bcsr = BCSR(
        (unique_data, sparsity.col_indices, sparsity.indptr),
        shape=(sparsity.n, sparsity.n),
    )
    diag = unique_data[sparsity.diag_idx]

    def matvec(x: JaxArray) -> JaxArray:
        return K_bcsr @ x

    def precon(x: JaxArray) -> JaxArray:
        return x / diag

    return matvec, precon


def cg_jax(
        K_data: JaxArray, sparsity: EmbeddedSparsity, b: JaxArray,
        rtol: float = 1e-10, max_iters: int | None = None,
) -> JaxArray:
    """Solve ``K x = b`` for symmetric positive-definite K via CG.

    Built on :func:`jax.scipy.sparse.linalg.cg` (no scipy callback,
    fully jit-traceable). The matvec is
    :class:`jax.experimental.sparse.BCSR` matrix-vector
    multiplication against the pre-built sparsity cache; the
    cache's ``perm`` + ``segment_ids`` gather + dedup ``K_data``
    (the embedded-BC COO data) into the unique CSR data buffer
    once per CG call. AD via
    :func:`jax.lax.custom_linear_solve` with ``symmetric=True``:
    ``K^T = K`` means the cotangent path re-uses ``solve`` (no
    separate ``transpose_solve``).

    ``K_data`` is the embedded-BC COO data buffer produced by
    :func:`_embedded_bc_enforce` — length
    ``nnz_assembled + n_presc``, with rows/cols touching
    prescribed dofs zeroed by the mask and ``alpha`` appended at
    ``(presc, presc)``. The cache references only the kept
    positions (free-free assembled entries + appended alpha
    entries), so structural zeros never participate in the
    matvec.

    The Jacobi preconditioner reads the unique-data diagonal via
    ``sparsity.diag_idx``. Each row has exactly one diagonal entry
    in the cache (validated at construction).

    For non-SPD K this will silently produce wrong results; the
    caller must ensure K is SPD. The embedded-BC symmetric form
    plus a self-adjoint underlying physics (e.g. small-strain
    elasticity) gives this.

    Unlike :func:`spsolve_jax` (which routes through scipy via
    :func:`jax.pure_callback` with ``vmap_method="sequential"``),
    this CG path is fully JAX-native and composes with
    :func:`jax.vmap` as a single batched :func:`jax.lax.while_loop`
    — a real advantage for Hessian-column batches and any other
    vmap-over-RHS pattern. The batched while_loop iterates until
    all batch elements converge (OR-reduced cond), so the slowest
    column dictates the iteration count for the batch.

    When the iteration count is needed (without AD), call
    :func:`cg_jax_with_iters` instead.
    """
    matvec, precon = _cg_jax_operator(K_data, sparsity)

    def solve(_unused_matvec, rhs: JaxArray) -> JaxArray:
        x, _info = jax.scipy.sparse.linalg.cg(
            matvec, rhs, M=precon, tol=rtol, maxiter=max_iters,
        )
        return x

    return lax.custom_linear_solve(
        matvec, b, solve, symmetric=True,
    )


def cg_jax_with_iters(
        K_data: JaxArray, sparsity: EmbeddedSparsity, b: JaxArray,
        rtol: float = 1e-10, max_iters: int | None = None,
) -> tuple[JaxArray, JaxArray]:
    """CG returning ``(x, iter_count)``.

    Same matvec + Jacobi preconditioner as :func:`cg_jax` via
    :func:`_cg_jax_operator`. Inner iteration is :func:`_pcg_loop`
    rather than :func:`jax.scipy.sparse.linalg.cg` so the loop
    counter is surfaced through the return tuple;
    :func:`jax.lax.custom_linear_solve`'s ``solve`` callback can
    return only a single output, so :func:`cg_jax` drops the
    count.

    Doesn't participate in JAX AD (no
    :func:`jax.lax.custom_linear_solve` wrapper). Call sites that
    need to differentiate through the linear solve must use
    :func:`cg_jax`.

    Iter count may differ from :func:`jax.scipy.sparse.linalg.cg`
    by ±1 on convergence-test rounding; acceptable for diagnostic
    purposes.
    """
    matvec, precon = _cg_jax_operator(K_data, sparsity)
    return _pcg_loop(matvec, b, precon, rtol, max_iters)


def _embedded_bc_enforce(
        K_bcoo: BCOO, presc_idx: JaxArray,
        presc_diag_scale: float = 1.0,
) -> tuple[JaxArray, JaxArray, JaxArray]:
    """Embedded-BC symmetric form on a :class:`BCOO` tangent.

    Returns COO triplets ``(K_data, K_rows, K_cols)`` for ``K`` with:

    - prescribed rows AND prescribed columns zeroed (every entry
      with either index in ``presc_idx``), via a mask multiplication
      on ``K_bcoo.data``;
    - scaled-identity entries with value ``presc_diag_scale``
      appended at ``(presc_idx, presc_idx)``.

    Net structure is block-diagonal:
    ``[K_ff (assembled) | 0; 0 | α · I_P]`` where ``K_ff`` is the
    free-free block of the assembled tangent and ``α · I_P`` is the
    prescribed self-block; the appended α keeps the prescribed
    block invertible after row+column zeroing.

    ``presc_diag_scale`` keeps the prescribed-row pivot on the same
    order of magnitude as the assembled diagonal so that Jacobi /
    ILU preconditioners see a well-conditioned matrix. The Newton
    solution is invariant to the choice provided the residual at
    prescribed dofs is rescaled by the same factor, since the per-
    row equation ``α·dU = -α·(U[presc] - presc_vals)`` produces
    ``dU[presc] = presc_vals - U[presc]`` regardless of α. Callers
    pass ``fe_problem.bc_diag_scale``.

    The symmetric form preserves the IFT-based JVP rule's
    ``K = ∂r/∂U_star`` invariant only when the residual function
    also zeros its dependence on ``U[presc]`` — see the matching
    clamp in :func:`cmad.fem.nonlinear_solver._fe_newton_solve_ad_jvp`'s
    ``r_of_p`` closure. With that pair in place, ``∂r/∂U_star``
    matches this output's row-and-column-zeroed structure exactly.

    When ``K_bcoo`` already has an entry at a prescribed ``(i, i)``,
    the output contains two COO entries at that position: the
    original (zeroed by the row/column mask) and the appended
    scaled-identity. Both consumer paths handle the duplicate
    correctly: :class:`scipy.sparse.coo_matrix` sums duplicates
    during ``.tocsr()`` conversion inside :func:`_spsolve_callback`,
    and JAX's scatter-add ``matvec`` accumulates them. The effective
    prescribed diagonal is therefore ``presc_diag_scale``.
    """
    rows = K_bcoo.indices[:, 0]
    cols = K_bcoo.indices[:, 1]
    n = K_bcoo.shape[0]
    p_mask = jnp.zeros(n, dtype=bool).at[presc_idx].set(True)

    keep = ~(p_mask[rows] | p_mask[cols])
    data_zeroed = K_bcoo.data * keep

    add_data = jnp.full(
        presc_idx.shape[0], presc_diag_scale, dtype=K_bcoo.data.dtype,
    )

    K_data = jnp.concatenate([data_zeroed, add_data])
    K_rows = jnp.concatenate([rows, presc_idx])
    K_cols = jnp.concatenate([cols, presc_idx])
    return K_data, K_rows, K_cols


@dataclass(frozen=True)
class EmbeddedSparsity:
    """CSR cache for the embedded-BC global tangent.

    Mesh + DOF map + Dirichlet BC indices determine the structural
    nonzero pattern of the BC-enforced K entirely; only data values
    vary across Newton iters. ``EmbeddedSparsity`` pre-computes the
    sort permutation, dedup segment ids, and CSR
    ``(indptr, col_indices)`` arrays so consumers can build a
    JAX-native BCSR or scipy CSR per iter without re-running the
    sort + dedup + indptr-build.

    Structural zeros are filtered out at construction. Under
    symmetric BC enforcement the ``(free, presc)`` and
    ``(presc, free)`` blocks of K are zero, so assembled COO entries
    at those positions contribute nothing to matvec or solve;
    they're omitted from the cached pattern. What remains is the
    assembled ``(free, free)`` block plus the appended ``alpha``
    entries at the prescribed diagonal that
    :func:`_embedded_bc_enforce` adds at runtime.

    Operator shape stays ``n x n`` (no DOF elimination); consumers
    keep the uniform embedded-form API produced by
    :func:`_embedded_bc_enforce`.

    Field semantics:

    - ``perm``: ``(nnz_kept,)`` permutation into the BC-enforced
      ``K_data`` (length ``nnz_assembled + n_presc``) that selects
      the kept positions in lex-sorted ``(row, col)`` order;
      ``K_data[perm]`` yields the relevant values.
    - ``segment_ids``: ``(nnz_kept,)`` maps each permuted entry to
      its unique ``(row, col)`` group;
      ``unique_data = segment_sum(K_data[perm], segment_ids,
      num_segments=num_unique)``.
    - ``num_unique``: number of unique ``(row, col)`` entries —
      length of the materialized CSR data buffer.
    - ``indptr``: ``(n+1,)`` CSR row pointers over unique entries.
    - ``col_indices``: ``(num_unique,)`` sorted column indices, one
      per unique entry.
    - ``diag_idx``: ``(n,)`` index into ``unique_data`` of each
      row's diagonal entry; lets diagonal-needing consumers (Jacobi
      preconditioner, residual checks) read the diagonal via
      ``unique_data[diag_idx]`` rather than a per-call scatter-add.
    - ``n``: matrix size (``dof_map.num_total_dofs``).
    """
    perm: JaxArray
    segment_ids: JaxArray
    num_unique: int
    indptr: JaxArray
    col_indices: JaxArray
    diag_idx: JaxArray
    n: int


def build_embedded_sparsity(
        fe_problem: FEProblem,
) -> EmbeddedSparsity:
    """Pre-compute the :class:`EmbeddedSparsity` for ``fe_problem``.

    Walks the assembled COO from
    :func:`cmad.fem.assembly.assembled_coo_indices`, filters
    structural zeros (entries whose row or column is prescribed),
    and appends ``(presc_idx, presc_idx)`` positions for the
    ``alpha`` diagonal block that :func:`_embedded_bc_enforce`
    adds at runtime. The lex-sort permutation, dedup segment ids,
    and CSR row pointers are derived from the kept set.

    When an assembled ``(presc, presc)`` entry coincides with an
    appended ``alpha`` position, the dedup folds the two into one
    unique entry with value ``alpha`` (zero from the runtime mask
    plus appended ``alpha``) — handled by the segment_sum without
    special-casing.
    """
    from cmad.fem.assembly import assembled_coo_indices

    rows_asm, cols_asm = assembled_coo_indices(fe_problem)
    presc_idx = np.asarray(
        fe_problem.dof_map.prescribed_indices, dtype=np.intp,
    )
    n = int(fe_problem.dof_map.num_total_dofs)
    nnz_asm = rows_asm.shape[0]
    n_presc = presc_idx.shape[0]

    is_presc = np.zeros(n, dtype=bool)
    is_presc[presc_idx] = True

    free_free_mask = ~is_presc[rows_asm] & ~is_presc[cols_asm]
    ff_positions = np.where(free_free_mask)[0].astype(np.intp)

    appended_positions = np.arange(
        nnz_asm, nnz_asm + n_presc, dtype=np.intp,
    )
    kept_positions = np.concatenate([ff_positions, appended_positions])

    full_rows = np.concatenate([rows_asm, presc_idx])
    full_cols = np.concatenate([cols_asm, presc_idx])
    kept_rows = full_rows[kept_positions]
    kept_cols = full_cols[kept_positions]

    sort_perm = np.lexsort((kept_cols, kept_rows))
    perm = kept_positions[sort_perm]
    sorted_rows = kept_rows[sort_perm]
    sorted_cols = kept_cols[sort_perm]

    nnz_kept = sorted_rows.shape[0]
    is_new_group = np.empty(nnz_kept, dtype=bool)
    is_new_group[0] = True
    is_new_group[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | (
        sorted_cols[1:] != sorted_cols[:-1]
    )
    segment_ids = (np.cumsum(is_new_group) - 1).astype(np.intp)
    num_unique = int(segment_ids[-1]) + 1

    unique_rows = sorted_rows[is_new_group]
    col_indices = sorted_cols[is_new_group].astype(np.intp)

    indptr = np.searchsorted(
        unique_rows, np.arange(n + 1), side="left",
    ).astype(np.intp)

    is_diag = unique_rows == col_indices
    diag_positions = np.where(is_diag)[0].astype(np.intp)
    diag_rows = unique_rows[diag_positions]
    diag_idx = np.full(n, -1, dtype=np.intp)
    diag_idx[diag_rows] = diag_positions
    if (diag_idx < 0).any():
        missing = int(np.where(diag_idx < 0)[0][0])
        raise ValueError(
            f"row {missing} has no diagonal entry in the BC-enforced "
            f"K sparsity pattern; FE assembly is expected to emit a "
            f"(row, row) entry for every dof"
        )

    return EmbeddedSparsity(
        perm=jnp.asarray(perm),
        segment_ids=jnp.asarray(segment_ids),
        num_unique=num_unique,
        indptr=jnp.asarray(indptr),
        col_indices=jnp.asarray(col_indices),
        diag_idx=jnp.asarray(diag_idx),
        n=n,
    )
