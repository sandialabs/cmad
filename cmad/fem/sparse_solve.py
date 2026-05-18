"""Sparse direct solve and embedded-BC enforcement for the FE Newton driver.

Two helpers:

- :func:`spsolve_jax` solves ``K x = b`` via
  :func:`scipy.sparse.linalg.spsolve` through :func:`jax.pure_callback`,
  with full forward-mode JVP and reverse-mode VJP supplied by
  :func:`jax.lax.custom_linear_solve`. ``K`` is given by its data
  buffer + a pre-built :class:`EmbeddedSparsity` describing its
  static CSR structure.

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
import pyamg
import scipy.sparse
import scipy.sparse.linalg
from jax import lax
from jax.experimental.sparse import BCOO, BCSR

from cmad.typing import JaxArray

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem


def spsolve_jax(
        K_data: JaxArray, sparsity: EmbeddedSparsity, b: JaxArray,
) -> JaxArray:
    """Solve ``K x = b`` for sparse ``K`` with full JAX AD support.

    ``K`` is described by ``(K_data, sparsity)``: the embedded-BC COO
    data buffer + the pre-built :class:`EmbeddedSparsity` cache that
    encodes the static CSR structure (sort permutation, dedup
    segment ids, ``indptr`` + ``col_indices``). Mirrors
    :func:`cg_jax`'s signature so the two solver entry points are
    interchangeable at FE-Newton call sites.

    Forward solve goes through :func:`scipy.sparse.linalg.spsolve`
    via :func:`jax.pure_callback`: the cache + ``K_data`` are
    deduped into ``unique_data`` on the JAX side, and the callback
    builds a :class:`scipy.sparse.csr_matrix` directly from
    ``(unique_data, sparsity.col_indices, sparsity.indptr)``.
    ``transpose_solve`` reuses the same CSR and passes its ``.T``
    (a zero-copy :class:`scipy.sparse.csc_matrix` view of ``K^T``)
    to ``spsolve``.

    AD rules are supplied by :func:`jax.lax.custom_linear_solve`
    from the ``(matvec, solve, transpose_solve)`` triple:

    - **Matvec** is :class:`jax.experimental.sparse.BCSR` matmul
      against the deduped data buffer (same pattern as
      :func:`cg_jax`'s matvec; the cache references only the kept
      structural entries).
    - **Forward-mode JVP.** ``x_dot = solve(b_dot - matvec_dot(x))``
      where ``matvec_dot(x)`` is JAX's JVP of the BCSR matvec at
      ``x`` along ``K_data_dot`` — exactly the textbook
      ``K_dot · x`` sensitivity term, computed by JAX through plain
      forward-mode AD on the dedup + BCSR matmul.
    - **Reverse-mode VJP.** ``λ = transpose_solve(x_bar)``;
      ``b_bar = λ``; the ``K_data`` cotangent comes from
      ``jax.vjp(matvec, x)(-λ)``.

    Composes for HVPs via forward-over-reverse: the outer JVP
    through :func:`jax.grad` re-enters this primitive's JVP rule
    with a non-zero ``K_data_dot`` (because ``K_data`` carries
    tangents from the upstream params), and JAX's auto-transposition
    handles the rest.

    ``vmap_method="expand_dims"`` on the callbacks: under
    :func:`jax.vmap` with K captured from the outer scope, the
    operator side broadcasts to a leading length-1 batch axis
    while ``rhs`` carries the real batch — the callback receives
    ``unique_data_np`` of shape ``(1, num_unique)`` and ``b_np``
    of shape ``(B, n)`` in a single host call. The scipy side
    then factors K once via :func:`scipy.sparse.linalg.splu` and
    back-substitutes across columns, amortizing the LU over the
    batch. Nested vmap (e.g. forward-over-reverse for Hessians)
    stacks additional length-1 axes on ``unique_data_np`` and
    additional batch axes on ``b_np``; the callbacks squeeze the
    K side and flatten the b side before scipy, then restore the
    caller's batch shape on return. Outside vmap both inputs
    arrive 1D and the callback dispatches a single
    :func:`scipy.sparse.linalg.spsolve` instead.

    A direct ``@custom_jvp`` over the ``pure_callback`` would not
    auto-transpose: ``pure_callback`` has no transpose rule, and
    JAX's auto-transposition tracks operation linearity rather than
    argument semantics. :func:`jax.lax.custom_linear_solve` factors
    that cleanly: ``matvec`` declares the operator,
    ``transpose_solve`` declares the adjoint, and JAX's built-in
    JVP / VJP rules cover any ``(matvec, solve, transpose_solve)``
    triple.
    """
    unique_data = jnp.zeros(
        sparsity.num_unique, dtype=K_data.dtype,
    ).at[sparsity.segment_ids].add(K_data[sparsity.perm])

    K_bcsr = BCSR(
        (unique_data, sparsity.col_indices, sparsity.indptr),
        shape=(sparsity.n, sparsity.n),
    )

    col_np = np.asarray(sparsity.col_indices)
    indptr_np = np.asarray(sparsity.indptr)
    n = sparsity.n

    def matvec(x: JaxArray) -> JaxArray:
        return K_bcsr @ x

    def _build_csr(unique_data_np: np.ndarray) -> scipy.sparse.csr_matrix:
        # Under ``vmap_method="expand_dims"``, stationary K under
        # :func:`jax.vmap` gains a leading length-1 axis per vmap
        # layer; nested AD (e.g. forward-over-reverse for Hessians)
        # can stack several such axes. Squeeze them all off, then
        # coerce to a plain :class:`numpy.ndarray` since
        # :func:`scipy.sparse.csr_matrix` rejects ``jax.Array``.
        ud = np.asarray(unique_data_np)
        while ud.ndim > 1 and ud.shape[0] == 1:
            ud = ud[0]
        return scipy.sparse.csr_matrix(
            (ud, col_np, indptr_np), shape=(n, n),
        )

    def _multi_back_sub(
            K_csc: scipy.sparse.csc_matrix, b_np: np.ndarray,
    ) -> np.ndarray:
        # Solve ``K x = b`` for one factor and arbitrarily many RHS
        # columns laid out in any number of leading batch axes.
        # Flattens batch axes for :class:`scipy.sparse.linalg.SuperLU`
        # (which accepts only 1D or 2D ``b``), then restores the
        # caller's batch shape on the return.
        b_arr = np.asarray(b_np)
        batch_shape = b_arr.shape[:-1]
        b_2d_T = np.ascontiguousarray(b_arr.reshape(-1, b_arr.shape[-1]).T)
        lu = scipy.sparse.linalg.splu(K_csc)
        return lu.solve(b_2d_T).T.reshape(*batch_shape, b_arr.shape[-1])

    def _scipy_solve(
            unique_data_np: np.ndarray, b_np: np.ndarray,
    ) -> np.ndarray:
        K_csr = _build_csr(unique_data_np)
        if b_np.ndim == 1:
            return np.asarray(scipy.sparse.linalg.spsolve(K_csr, b_np))
        return _multi_back_sub(K_csr.tocsc(), b_np)

    def _scipy_transpose_solve(
            unique_data_np: np.ndarray, b_np: np.ndarray,
    ) -> np.ndarray:
        K_csr = _build_csr(unique_data_np)
        if b_np.ndim == 1:
            return np.asarray(scipy.sparse.linalg.spsolve(K_csr.T, b_np))
        return _multi_back_sub(K_csr.T.tocsc(), b_np)

    def solve(_unused_matvec, rhs: JaxArray) -> JaxArray:
        return jax.pure_callback(
            _scipy_solve,
            jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            unique_data, rhs,
            vmap_method="expand_dims",
        )

    def transpose_solve(_unused_vecmat, rhs: JaxArray) -> JaxArray:
        return jax.pure_callback(
            _scipy_transpose_solve,
            jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            unique_data, rhs,
            vmap_method="expand_dims",
        )

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


def _bcsr_jacobi_operator(
        K_data: JaxArray, sparsity: EmbeddedSparsity,
) -> tuple[
    Callable[[JaxArray], JaxArray],
    Callable[[JaxArray], JaxArray],
]:
    """Build the BCSR matvec + Jacobi preconditioner.

    Shared setup between :func:`cg_jax`, :func:`cg_jax_with_iters`,
    and :func:`gmres_jax`; centralizing the dedup + BCSR
    construction + diagonal extraction keeps all callers' matvec
    and preconditioner definitions in lockstep even though their
    inner iterative solver implementations differ.
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
    matvec, precon = _bcsr_jacobi_operator(K_data, sparsity)

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
    :func:`_bcsr_jacobi_operator`. Inner iteration is :func:`_pcg_loop`
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
    matvec, precon = _bcsr_jacobi_operator(K_data, sparsity)
    return _pcg_loop(matvec, b, precon, rtol, max_iters)


def gmres_jax(
        K_data: JaxArray, sparsity: EmbeddedSparsity, b: JaxArray,
        rtol: float = 1e-10, max_iters: int | None = None,
        restart: int = 20,
) -> JaxArray:
    """Solve ``K x = b`` for general (possibly non-symmetric) K via GMRES.

    Built on :func:`jax.scipy.sparse.linalg.gmres` running restarted
    GMRES(``restart``), no scipy callback (fully jit-traceable). The
    matvec is :class:`jax.experimental.sparse.BCSR` matrix-vector
    multiplication against the pre-built sparsity cache; the cache's
    ``perm`` + ``segment_ids`` gather + dedup ``K_data`` into the
    unique CSR data buffer once per call (same operator construction
    as :func:`cg_jax`).

    AD via :func:`jax.lax.custom_linear_solve` with ``symmetric=False``:
    the forward solve goes through ``solve``; the adjoint / transpose
    solve goes through ``transpose_solve``, which runs GMRES against
    the auto-transposed ``vecmat`` JAX supplies (the linear matvec is
    JVP-transposable; no precomputed K^T cache is required). The
    Jacobi (diagonal) preconditioner is symmetric, so the same
    ``precon`` is reused on the transpose path — appropriate for any
    diagonal preconditioner, the only kind currently available from
    the cache.

    Trade-off vs :func:`cg_jax` on SPD K: CG converges in ``O(√κ)``
    iters with bounded memory, while restarted GMRES needs ``O(κ)``
    matvecs and stores the Krylov basis up to ``restart``. Use
    :func:`cg_jax` when SPD is known (small-strain elasticity with
    self-adjoint kernels); use :func:`gmres_jax` for general K
    (follower loads, non-associative plasticity) and as a comparison
    point for non-symmetric workloads.

    Like :func:`cg_jax`, this path is fully JAX-native and composes
    with :func:`jax.vmap` as a single batched ``while_loop``; the
    slowest column dictates the iteration count for the batch.
    """
    matvec, precon = _bcsr_jacobi_operator(K_data, sparsity)

    def solve(matvec_: Callable[[JaxArray], JaxArray],
              rhs: JaxArray) -> JaxArray:
        x, _info = jax.scipy.sparse.linalg.gmres(
            matvec_, rhs, M=precon, tol=rtol, maxiter=max_iters,
            restart=restart,
        )
        return x

    def transpose_solve(vecmat: Callable[[JaxArray], JaxArray],
                        rhs: JaxArray) -> JaxArray:
        x, _info = jax.scipy.sparse.linalg.gmres(
            vecmat, rhs, M=precon, tol=rtol, maxiter=max_iters,
            restart=restart,
        )
        return x

    return lax.custom_linear_solve(
        matvec, b, solve, transpose_solve=transpose_solve,
        symmetric=False,
    )


def cg_amg_jax(
        K_data: JaxArray, sparsity: EmbeddedSparsity, b: JaxArray,
        rtol: float = 1e-10, max_iters: int | None = None,
        *, pyamg_kwargs: dict | None = None,
) -> JaxArray:
    """Solve ``K x = b`` for SPD K via scipy CG preconditioned by
    pyamg's smoothed-aggregation algebraic multigrid.

    The CG iteration runs inside one :func:`jax.pure_callback`: scipy
    builds a CSR from the :class:`EmbeddedSparsity` cache + deduped
    ``K_data`` (same recipe as :func:`spsolve_jax`'s ``_build_csr``),
    :func:`pyamg.smoothed_aggregation_solver` sets up the hierarchy,
    its V-cycle preconditioner is exposed via ``.aspreconditioner()``,
    and :func:`scipy.sparse.linalg.cg` runs to convergence.

    AD via :func:`jax.lax.custom_linear_solve` with ``symmetric=True``;
    the matvec used for the AD path is a JAX-traceable BCSR matmul
    against the same dedup buffer (parallel to :func:`cg_jax` /
    :func:`gmres_jax` / :func:`spsolve_jax`).

    ``pyamg_kwargs`` are forwarded verbatim to
    :func:`pyamg.smoothed_aggregation_solver`; ``None`` selects pyamg's
    defaults. For a CSR ``K`` the default near null space is the
    constant vector — correct for scalar diffusion, suboptimal for
    elasticity (whose true near null space is the rigid-body modes),
    but still converges. No CMAD-side validation of kwargs — pyamg
    raises on bad keys.

    Under :func:`jax.vmap` with K captured outside, ``rhs`` arrives 2D
    at the callback (length-1 leading axes on K squeezed); scipy CG is
    1D-RHS only, so the 2D branch loops columns reusing the same
    hierarchy.

    For non-SPD K this will silently produce wrong results.
    """
    unique_data = jnp.zeros(
        sparsity.num_unique, dtype=K_data.dtype,
    ).at[sparsity.segment_ids].add(K_data[sparsity.perm])

    K_bcsr = BCSR(
        (unique_data, sparsity.col_indices, sparsity.indptr),
        shape=(sparsity.n, sparsity.n),
    )

    col_np = np.asarray(sparsity.col_indices)
    indptr_np = np.asarray(sparsity.indptr)
    n = sparsity.n
    pyamg_kw = pyamg_kwargs or {}

    def matvec(x: JaxArray) -> JaxArray:
        return K_bcsr @ x

    def _build_csr(unique_data_np: np.ndarray) -> scipy.sparse.csr_matrix:
        ud = np.asarray(unique_data_np)
        while ud.ndim > 1 and ud.shape[0] == 1:
            ud = ud[0]
        return scipy.sparse.csr_matrix(
            (ud, col_np, indptr_np), shape=(n, n),
        )

    def _scipy_amg_cg(
            unique_data_np: np.ndarray, b_np: np.ndarray,
    ) -> np.ndarray:
        K_csr = _build_csr(unique_data_np)
        ml = pyamg.smoothed_aggregation_solver(K_csr, **pyamg_kw)
        M = ml.aspreconditioner()
        b_arr = np.asarray(b_np)
        maxiter = max_iters if max_iters is not None else 10 * n
        if b_arr.ndim == 1:
            x, _info = scipy.sparse.linalg.cg(
                K_csr, b_arr, M=M, rtol=rtol, maxiter=maxiter,
            )
            return np.asarray(x)
        batch_shape = b_arr.shape[:-1]
        b_2d = b_arr.reshape(-1, b_arr.shape[-1])
        x_2d = np.empty_like(b_2d)
        for i in range(b_2d.shape[0]):
            x_i, _info = scipy.sparse.linalg.cg(
                K_csr, b_2d[i], M=M, rtol=rtol, maxiter=maxiter,
            )
            x_2d[i] = x_i
        return x_2d.reshape(*batch_shape, b_arr.shape[-1])

    def solve(_unused_matvec, rhs: JaxArray) -> JaxArray:
        return jax.pure_callback(
            _scipy_amg_cg,
            jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            unique_data, rhs,
            vmap_method="expand_dims",
        )

    return lax.custom_linear_solve(
        matvec, b, solve, symmetric=True,
    )


def _embedded_bc_enforce(
        K_bcoo: BCOO, presc_idx: JaxArray,
) -> tuple[JaxArray, JaxArray]:
    """Embedded-BC symmetric form on a :class:`BCOO` tangent.

    Returns ``(K_data, K_ii_presc)``:

    - ``K_data[:nnz_assembled]`` are the assembled COO values with
      prescribed rows AND prescribed columns zeroed (every entry
      with either index in ``presc_idx``), via a mask multiplication
      on ``K_bcoo.data``;
    - ``K_data[nnz_assembled:]`` are the original assembled diagonal
      values ``K_ii`` at the prescribed rows, appended at
      ``(presc_idx, presc_idx)``;
    - ``K_ii_presc`` is the length-``n_presc`` vector of those same
      diagonal values, surfaced for the residual rescale at
      prescribed rows in the Newton driver.

    The implicit ``(rows, cols)`` are
    ``concatenate([K_bcoo.indices[:, 0], presc_idx])`` and
    ``concatenate([K_bcoo.indices[:, 1], presc_idx])``; the cache
    references them statically via ``perm``.

    Net structure is block-diagonal:
    ``[K_ff (assembled) | 0; 0 | diag(K_ii)]`` where ``K_ff`` is the
    free-free block of the assembled tangent and ``diag(K_ii)`` is the
    prescribed self-block carrying the original assembled diagonal
    entries. Keeping the prescribed-row pivot at the assembled local
    stiffness preserves the matrix's local diagonal scale uniformly
    across the boundary — AMG hierarchy construction (prolongation
    smoothing in particular) sees a locally-consistent diagonal,
    avoiding the boundary-region distortion a global scalar pivot
    would introduce. The Newton solution is invariant to the choice
    because the per-row equation
    ``K_ii · dU = -K_ii · (U[presc] - presc_vals)`` produces
    ``dU[presc] = presc_vals - U[presc]`` regardless of ``K_ii``.

    ``K_ii`` is extracted via a JAX-native scatter-add of the
    diagonal-masked COO entries (``K_bcoo.data * (rows == cols)``)
    into a length-``n`` buffer, then indexed at ``presc_idx``. O(nnz
    + n) traceable.

    The symmetric form preserves the IFT-based JVP rule's
    ``K = ∂r/∂U_star`` invariant only when the residual function
    also zeros its dependence on ``U[presc]`` — see the matching
    clamp in :func:`cmad.fem.nonlinear_solver._fe_newton_solve_ad_jvp`'s
    ``r_of_p`` closure. With that pair in place, ``∂r/∂U_star``
    matches this output's row-and-column-zeroed structure exactly.

    When ``K_bcoo`` already has an entry at a prescribed ``(i, i)``,
    the output contains two values at that position: the original
    (zeroed by the row/column mask) and the appended ``K_ii``.
    :class:`EmbeddedSparsity`'s segment-sum dedup at the cache
    boundary folds them into one unique entry with value ``K_ii``.
    """
    rows = K_bcoo.indices[:, 0]
    cols = K_bcoo.indices[:, 1]
    n = K_bcoo.shape[0]
    p_mask = jnp.zeros(n, dtype=bool).at[presc_idx].set(True)

    keep = ~(p_mask[rows] | p_mask[cols])
    data_zeroed = K_bcoo.data * keep

    diag_mask = rows == cols
    K_ii_full = jnp.zeros(n, dtype=K_bcoo.data.dtype).at[rows].add(
        K_bcoo.data * diag_mask,
    )
    K_ii_presc = K_ii_full[presc_idx]

    return jnp.concatenate([data_zeroed, K_ii_presc]), K_ii_presc


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
