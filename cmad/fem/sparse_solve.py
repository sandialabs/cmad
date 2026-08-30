"""Sparse direct solve and embedded-BC enforcement for the FE Newton driver.

The jax native Krylov solvers and the block preconditioner apply the tangent
through the :class:`TangentOperator` interface; :class:`AssembledOperator`
is the assembled sparse matrix behind it. Two further helpers:

- :func:`scipy_lu` solves ``K x = b`` via
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

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import pyamg
import scipy.sparse
import scipy.sparse.linalg
from jax import lax
from jax.experimental.sparse import BCOO, BCSR
from jax.tree_util import register_pytree_node_class
from numpy.typing import NDArray

from cmad.typing import JaxArray

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem


class TangentOperator(Protocol):
    """What the jax native Krylov solvers and the block preconditioner
    apply: the embedded BC tangent as a matvec, its diagonal, and, for a
    problem with several fields, its field blocks (block ``(i, j)`` takes a
    field ``j`` vector to a field ``i`` vector; ``transpose`` applies the
    transpose operator's block)."""

    @property
    def n(self) -> int: ...

    @property
    def num_fields(self) -> int: ...

    @property
    def field_offsets(self) -> tuple[int, ...]: ...

    def matvec(self, x: JaxArray) -> JaxArray: ...

    def diagonal(self) -> JaxArray: ...

    def block_matvec(
            self, i: int, j: int, x: JaxArray, *, transpose: bool,
    ) -> JaxArray: ...

    def block_diagonal(self, i: int) -> JaxArray: ...


class AssembledOperator:
    """The embedded BC tangent as the assembled sparse matrix.

    ``K_data`` (the embedded COO data of :func:`_embedded_bc_enforce`) is
    segment summed onto the cached CSR pattern of ``sparsity`` and applied
    as a BCSR matvec; the field blocks come from ``block_sparsity``
    (``None`` for a single field problem). The scipy callback solvers read
    ``unique_data`` to rebuild a host side CSR.
    """

    def __init__(
            self, K_data: JaxArray, sparsity: EmbeddedSparsity,
            block_sparsity: BlockSparsity | None = None,
    ) -> None:
        self.sparsity = sparsity
        self.block_sparsity = block_sparsity
        self.unique_data = jnp.zeros(
            sparsity.num_unique, dtype=K_data.dtype,
        ).at[sparsity.segment_ids].add(K_data[sparsity.perm])
        self._K_bcsr = BCSR(
            (self.unique_data, sparsity.col_indices, sparsity.indptr),
            shape=(sparsity.n, sparsity.n),
        )
        self._pair_index = (
            {pair: k for k, pair in enumerate(block_sparsity.pairs)}
            if block_sparsity is not None else {}
        )

    @property
    def n(self) -> int:
        return self.sparsity.n

    @property
    def _blocks(self) -> BlockSparsity:
        if self.block_sparsity is None:
            raise ValueError(
                "the operator has no field blocks: the problem has one field",
            )
        return self.block_sparsity

    @property
    def num_fields(self) -> int:
        return self._blocks.num_fields

    @property
    def field_offsets(self) -> tuple[int, ...]:
        return self._blocks.field_offsets

    def matvec(self, x: JaxArray) -> JaxArray:
        return self._K_bcsr @ x

    def diagonal(self) -> JaxArray:
        return self.unique_data[self.sparsity.diag_idx]

    def block_matvec(
            self, i: int, j: int, x: JaxArray, *, transpose: bool = False,
    ) -> JaxArray:
        """Multiply one field block of the operator by a vector.

        The unknowns are grouped by field, so the operator splits into
        blocks: block ``(i, j)`` couples field ``j``'s unknowns to field
        ``i``'s equations. ``transpose=True`` applies the transpose
        operator's ``(i, j)`` block, which is the stored ``(j, i)`` block
        with its rows and columns swapped: the product is a scatter add,
        so swapping the gather and scatter indices transposes it. An
        empty field pair contributes nothing.
        """
        bs = self._blocks
        n_i = bs.field_offsets[i + 1] - bs.field_offsets[i]
        key = (j, i) if transpose else (i, j)
        if key not in self._pair_index:
            return jnp.zeros(n_i, dtype=self.unique_data.dtype)
        k = self._pair_index[key]
        data = self.unique_data[bs.global_data_indices[k]]
        if not transpose:
            return jnp.zeros(n_i, dtype=data.dtype).at[bs.local_rows[k]].add(
                data * x[bs.local_cols[k]],
            )
        return jnp.zeros(n_i, dtype=data.dtype).at[bs.local_cols[k]].add(
            data * x[bs.local_rows[k]],
        )

    def block_diagonal(self, i: int) -> JaxArray:
        """Main diagonal of field ``i``'s own block ``(i, i)``, the same
        on the forward and transpose sweeps."""
        bs = self._blocks
        n_i = bs.field_offsets[i + 1] - bs.field_offsets[i]
        k = self._pair_index[(i, i)]
        data = self.unique_data[bs.global_data_indices[k]]
        rows = bs.local_rows[k]
        cols = bs.local_cols[k]
        return jnp.zeros(n_i, dtype=data.dtype).at[rows].add(
            jnp.where(rows == cols, data, 0.0),
        )


def _jacobi_preconditioner(
        op: TangentOperator,
) -> Callable[[JaxArray], JaxArray]:
    """Division by the operator's diagonal."""
    diag = op.diagonal()

    def precon(x: JaxArray) -> JaxArray:
        return x / diag

    return precon


def _build_scipy_csr(
        unique_data_np: np.ndarray, col_np: np.ndarray,
        indptr_np: np.ndarray, n: int,
) -> scipy.sparse.csr_matrix:
    """Host-side scipy CSR from a scipy-callback solver's operands.

    ``unique_data`` / ``col_indices`` / ``indptr`` arrive as
    stationary :func:`jax.pure_callback` operands, each carrying a
    leading length-1 axis per enclosing :func:`jax.vmap` layer
    (``vmap_method="expand_dims"``); each is flattened back to 1D.
    """
    return scipy.sparse.csr_matrix(
        (
            np.reshape(unique_data_np, -1),
            np.reshape(col_np, -1),
            np.reshape(indptr_np, -1),
        ),
        shape=(n, n),
    )


def _symmetric_diagonal_scaling(
        K_csc: scipy.sparse.csc_matrix,
) -> NDArray[np.floating]:
    """``s`` with ``s_i = 1/sqrt(|K_ii|)``, leaving a negligible diagonal
    unscaled."""
    d: NDArray[np.floating] = np.sqrt(np.abs(K_csc.diagonal()))
    d[d <= np.finfo(d.dtype).eps * d.max()] = 1.0
    return 1.0 / d


def _index_dtype(largest_index: int) -> type:
    """``np.int32`` when ``largest_index`` fits it, else ``np.int64``."""
    return np.int32 if largest_index <= np.iinfo(np.int32).max else np.int64


def fill_reducing_permutation(
        indptr: NDArray[np.integer], col_indices: NDArray[np.integer],
        n: int,
) -> NDArray[np.integer] | None:
    """A symmetric fill reducing permutation for the CSR pattern, or
    ``None`` when ``scikit-sparse`` is absent or declines it."""
    try:
        from sksparse import cholmod  # type: ignore[import-untyped]
    except ImportError:
        return None
    pattern = scipy.sparse.csr_matrix(
        (np.ones(col_indices.shape[0]), col_indices, indptr), shape=(n, n))
    try:
        result = cholmod.nesdis((pattern + pattern.T).tocsc())
    except Exception as exc:
        warnings.warn(
            f"scikit-sparse is installed but produced no fill reducing "
            f"ordering ({type(exc).__name__}); falling back to COLAMD",
            stacklevel=2)
        return None
    perm = np.asarray(result[0] if isinstance(result, tuple) else result)
    if perm.shape != (n,) or not np.array_equal(np.sort(perm), np.arange(n)):
        return None
    return perm


def _scaled_lu_solve(
        K_csc: scipy.sparse.csc_matrix, b_np: np.ndarray,
        perm: NDArray[np.integer] | None = None,
) -> np.ndarray:
    """Solve ``K x = b`` as ``(S K S) y = S b`` with ``x = S y`` and
    ``S = diag(s)``, reordering by ``perm`` when one is given."""
    s = _symmetric_diagonal_scaling(K_csc)
    S = scipy.sparse.diags(s)
    scaled = (S @ K_csc @ S).tocsc()
    rhs = s * np.asarray(b_np)
    if perm is None:
        return np.asarray(s * scipy.sparse.linalg.splu(scaled).solve(rhs))
    lu = scipy.sparse.linalg.splu(
        scaled[perm][:, perm].tocsc(), permc_spec="NATURAL")
    y = np.empty_like(rhs)
    y[perm] = lu.solve(rhs[perm])
    return np.asarray(s * y)


def scipy_lu(
        K_data: JaxArray, sparsity: EmbeddedSparsity, b: JaxArray,
        fill_perm: NDArray[np.integer] | None = None,
) -> JaxArray:
    """Solve ``K x = b`` for sparse ``K`` with full JAX AD support.

    ``K`` is described by ``(K_data, sparsity)``: the embedded-BC COO
    data buffer + the pre-built :class:`EmbeddedSparsity` cache that
    encodes the static CSR structure (sort permutation, dedup
    segment ids, ``indptr`` + ``col_indices``). Mirrors
    :func:`jax_cg`'s signature so the two solver entry points are
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
      :func:`jax_cg`'s matvec; the cache references only the kept
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
    arrive 1D and the callback factors once for the single column.

    A direct ``@custom_jvp`` over the ``pure_callback`` would not
    auto-transpose: ``pure_callback`` has no transpose rule, and
    JAX's auto-transposition tracks operation linearity rather than
    argument semantics. :func:`jax.lax.custom_linear_solve` factors
    that cleanly: ``matvec`` declares the operator,
    ``transpose_solve`` declares the adjoint, and JAX's built-in
    JVP / VJP rules cover any ``(matvec, solve, transpose_solve)``
    triple.
    """
    op = AssembledOperator(K_data, sparsity)
    unique_data, matvec = op.unique_data, op.matvec
    n = sparsity.n

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
            unique_data_np: np.ndarray, col_np: np.ndarray,
            indptr_np: np.ndarray, b_np: np.ndarray,
    ) -> np.ndarray:
        K_csr = _build_scipy_csr(unique_data_np, col_np, indptr_np, n)
        if b_np.ndim == 1:
            return np.asarray(
                _scaled_lu_solve(K_csr.tocsc(), b_np, fill_perm))
        # the batched path is left unscaled: it factors at default pivoting
        return _multi_back_sub(K_csr.tocsc(), b_np)

    def _scipy_transpose_solve(
            unique_data_np: np.ndarray, col_np: np.ndarray,
            indptr_np: np.ndarray, b_np: np.ndarray,
    ) -> np.ndarray:
        K_csr = _build_scipy_csr(unique_data_np, col_np, indptr_np, n)
        if b_np.ndim == 1:
            return np.asarray(
                _scaled_lu_solve(K_csr.T.tocsc(), b_np, fill_perm))
        return _multi_back_sub(K_csr.T.tocsc(), b_np)

    def solve(_unused_matvec, rhs: JaxArray) -> JaxArray:
        return jax.pure_callback(
            _scipy_solve,
            jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            unique_data, sparsity.col_indices, sparsity.indptr, rhs,
            vmap_method="expand_dims",
        )

    def transpose_solve(_unused_vecmat, rhs: JaxArray) -> JaxArray:
        return jax.pure_callback(
            _scipy_transpose_solve,
            jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            unique_data, sparsity.col_indices, sparsity.indptr, rhs,
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

    Used by :func:`jax_cg_with_iters` to expose the counter
    through its return tuple. :func:`jax_cg` defers to
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


def _gmres_loop(
        matvec: Callable[[JaxArray], JaxArray],
        precon: Callable[[JaxArray], JaxArray],
        b: JaxArray,
        rtol: float,
        restart: int,
        max_iters: int | None,
) -> tuple[JaxArray, JaxArray]:
    """Right preconditioned restarted GMRES, returning ``(x, iterations)``.

    Solves ``(A M) y = b`` and returns ``x = M y``, with ``A`` the
    ``matvec`` and ``M`` the ``precon``, so the residual it tests is the
    true one, ``||b - A x||``, against ``rtol ||b||``. The Arnoldi basis is
    orthogonalized by two passes of classical Gram-Schmidt against the
    whole basis (batched projections; the second pass is what keeps the
    basis orthogonal on ill-conditioned operators, where a single pass
    stagnates: Giraud, Langou, Rozloznik and van den Eshof, 2005), the
    residual estimate is carried through Givens rotations
    so a cycle exits the step it converges, and each cycle ends by
    recomputing the true residual. ``restart`` is the Krylov dimension per
    cycle; ``max_iters`` is the cycle count cap, as for
    :func:`scipy.sparse.linalg.gmres` (``None`` selects ``10 *
    b.shape[0]``). A breakdown (new basis vector below roundoff of the
    vector it came from) ends the cycle with the exact solution of that
    Krylov space.

    ``iterations`` is the total Krylov iteration count over all cycles, for
    diagnostics; :func:`_gmres_solve` drops it for
    :func:`jax.lax.custom_linear_solve`.
    """
    n = b.shape[0]
    restart = min(restart, n)
    if max_iters is None:
        max_iters = 10 * n
    dtype = b.dtype
    eps = jnp.finfo(dtype).eps
    b_norm = jnp.linalg.norm(b)
    tol = rtol * b_norm

    def cycle(carry: tuple) -> tuple:
        x, r, r_norm, cycles, done = carry
        # Basis vectors are rows of V so each new one is a contiguous write.
        V = jnp.zeros((restart + 1, n), dtype).at[0].set(r / r_norm)
        # H[k] holds column k of the Hessenberg matrix after the rotations;
        # untouched rows are identity rows so the triangular solve below is
        # well posed however early the cycle stops.
        H = jnp.eye(restart, restart + 1, dtype=dtype)
        cs = jnp.zeros(restart, dtype)
        sn = jnp.zeros(restart, dtype)
        g = jnp.zeros(restart + 1, dtype).at[0].set(r_norm)

        def step_cond(state: tuple) -> JaxArray:
            k, _V, _H, _cs, _sn, _g, err = state
            return (k < restart) & (err > tol)

        def step(state: tuple) -> tuple:
            k, V, H, cs, sn, g, _err = state
            w = matvec(precon(V[k]))
            w_norm_0 = jnp.linalg.norm(w)
            h1 = V @ w
            w = w - h1 @ V
            h2 = V @ w
            w = w - h2 @ V
            h = h1 + h2
            w_norm = jnp.linalg.norm(w)
            breakdown = w_norm <= eps * w_norm_0
            h = h.at[k + 1].set(jnp.where(breakdown, 0.0, w_norm))
            V = V.at[k + 1].set(
                jnp.where(breakdown, 0.0, w / jnp.where(breakdown, 1.0, w_norm)),
            )

            def rotate(j: int, h: JaxArray) -> JaxArray:
                hj = cs[j] * h[j] + sn[j] * h[j + 1]
                hj1 = -sn[j] * h[j] + cs[j] * h[j + 1]
                return h.at[j].set(hj).at[j + 1].set(hj1)

            h = lax.fori_loop(0, k, rotate, h)
            rho = jnp.hypot(h[k], h[k + 1])
            c = jnp.where(rho > 0.0, h[k] / jnp.where(rho > 0.0, rho, 1.0), 1.0)
            s = jnp.where(rho > 0.0, h[k + 1] / jnp.where(rho > 0.0, rho, 1.0), 0.0)
            h = h.at[k].set(rho).at[k + 1].set(0.0)
            g = g.at[k + 1].set(-s * g[k]).at[k].set(c * g[k])
            err = jnp.abs(g[k + 1])
            return (
                k + 1, V, H.at[k].set(h), cs.at[k].set(c), sn.at[k].set(s),
                g, err,
            )

        k, V, H, _cs, _sn, g, _err = lax.while_loop(
            step_cond, step, (jnp.int32(0), V, H, cs, sn, g, r_norm),
        )
        # g[k] is the residual estimate, not a right hand side entry; the
        # identity rows of H past k must see zeros so y vanishes there.
        g_solved = jnp.where(jnp.arange(restart) < k, g[:-1], 0.0)
        y = jax.scipy.linalg.solve_triangular(H[:, :-1].T, g_solved, lower=False)
        x = x + precon(y @ V[:-1])
        r = b - matvec(x)
        return x, r, jnp.linalg.norm(r), cycles + 1, done + k

    def cycle_cond(carry: tuple) -> JaxArray:
        _x, _r, r_norm, cycles, _done = carry
        return (r_norm > tol) & (cycles < max_iters)

    x0 = jnp.zeros_like(b)
    x, _r, _r_norm, _cycles, iterations = lax.while_loop(
        cycle_cond, cycle, (x0, b, b_norm, jnp.int32(0), jnp.int32(0)),
    )
    return x, iterations


def _gmres_solve(
        matvec: Callable[[JaxArray], JaxArray],
        precon: Callable[[JaxArray], JaxArray],
        b: JaxArray,
        rtol: float,
        restart: int,
        max_iters: int | None,
) -> JaxArray:
    """:func:`_gmres_loop` without the iteration count, the shape
    :func:`jax.lax.custom_linear_solve`'s ``solve`` callbacks need."""
    return _gmres_loop(matvec, precon, b, rtol, restart, max_iters)[0]


def _jacobi_cg(
        op: TangentOperator, b: JaxArray, rtol: float, max_iters: int | None,
) -> JaxArray:
    """Jacobi preconditioned CG on ``op``, differentiable through
    :func:`jax.lax.custom_linear_solve` (``symmetric=True``)."""
    precon = _jacobi_preconditioner(op)

    def solve(_unused_matvec, rhs: JaxArray) -> JaxArray:
        x, _info = jax.scipy.sparse.linalg.cg(
            op.matvec, rhs, M=precon, tol=rtol, maxiter=max_iters,
        )
        return x

    return lax.custom_linear_solve(
        op.matvec, b, solve, symmetric=True,
    )


def _jacobi_gmres(
        op: TangentOperator, b: JaxArray, rtol: float, restart: int,
        max_iters: int | None,
) -> JaxArray:
    """Jacobi preconditioned GMRES on ``op`` (:func:`_gmres_loop`),
    differentiable through :func:`jax.lax.custom_linear_solve`; the adjoint
    solve runs on the auto transposed matvec with the same diagonal
    preconditioner."""
    precon = _jacobi_preconditioner(op)

    def solve(matvec_: Callable[[JaxArray], JaxArray],
              rhs: JaxArray) -> JaxArray:
        return _gmres_solve(matvec_, precon, rhs, rtol, restart, max_iters)

    def transpose_solve(vecmat: Callable[[JaxArray], JaxArray],
                        rhs: JaxArray) -> JaxArray:
        return _gmres_solve(vecmat, precon, rhs, rtol, restart, max_iters)

    return lax.custom_linear_solve(
        op.matvec, b, solve, transpose_solve=transpose_solve,
        symmetric=False,
    )


def jax_cg(
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

    Unlike :func:`scipy_lu` (which routes through scipy via
    :func:`jax.pure_callback` with ``vmap_method="sequential"``),
    this CG path is fully JAX-native and composes with
    :func:`jax.vmap` as a single batched :func:`jax.lax.while_loop`
    — a real advantage for Hessian-column batches and any other
    vmap-over-RHS pattern. The batched while_loop iterates until
    all batch elements converge (OR-reduced cond), so the slowest
    column dictates the iteration count for the batch.

    When the iteration count is needed (without AD), call
    :func:`jax_cg_with_iters` instead.
    """
    return _jacobi_cg(AssembledOperator(K_data, sparsity), b, rtol, max_iters)


def jax_cg_with_iters(
        K_data: JaxArray, sparsity: EmbeddedSparsity, b: JaxArray,
        rtol: float = 1e-10, max_iters: int | None = None,
) -> tuple[JaxArray, JaxArray]:
    """CG returning ``(x, iter_count)``.

    Same operator + Jacobi preconditioner as :func:`jax_cg`. Inner
    iteration is :func:`_pcg_loop`
    rather than :func:`jax.scipy.sparse.linalg.cg` so the loop
    counter is surfaced through the return tuple;
    :func:`jax.lax.custom_linear_solve`'s ``solve`` callback can
    return only a single output, so :func:`jax_cg` drops the
    count.

    Doesn't participate in JAX AD (no
    :func:`jax.lax.custom_linear_solve` wrapper). Call sites that
    need to differentiate through the linear solve must use
    :func:`jax_cg`.

    Iter count may differ from :func:`jax.scipy.sparse.linalg.cg`
    by ±1 on convergence-test rounding; acceptable for diagnostic
    purposes.
    """
    op = AssembledOperator(K_data, sparsity)
    return _pcg_loop(op.matvec, b, _jacobi_preconditioner(op), rtol, max_iters)


def jax_gmres(
        K_data: JaxArray, sparsity: EmbeddedSparsity, b: JaxArray,
        rtol: float = 1e-10, max_iters: int | None = None,
        restart: int = 20,
) -> JaxArray:
    """Solve ``K x = b`` for general (possibly non-symmetric) K via GMRES.

    Built on :func:`_gmres_loop`: right preconditioned restarted
    GMRES(``restart``) with a two pass Gram-Schmidt Arnoldi, testing the
    true residual against ``rtol ||b||`` and capping the restart cycles
    at ``max_iters``; no scipy callback (fully jit-traceable).
    The matvec is :class:`jax.experimental.sparse.BCSR` matrix-vector
    multiplication against the pre-built sparsity cache; the cache's
    ``perm`` + ``segment_ids`` gather + dedup ``K_data`` into the
    unique CSR data buffer once per call (same operator construction
    as :func:`jax_cg`).

    AD via :func:`jax.lax.custom_linear_solve` with ``symmetric=False``:
    the forward solve goes through ``solve``; the adjoint / transpose
    solve goes through ``transpose_solve``, which runs GMRES against
    the auto-transposed ``vecmat`` JAX supplies (the linear matvec is
    JVP-transposable; no precomputed K^T cache is required). The
    Jacobi (diagonal) preconditioner is symmetric, so the same
    ``precon`` is reused on the transpose path — appropriate for any
    diagonal preconditioner, the only kind currently available from
    the cache.

    Trade-off vs :func:`jax_cg` on SPD K: CG converges in ``O(√κ)``
    iters with bounded memory, while restarted GMRES needs ``O(κ)``
    matvecs and stores the Krylov basis up to ``restart``. Use
    :func:`jax_cg` when SPD is known (small-strain elasticity with
    self-adjoint kernels); use :func:`jax_gmres` for general K
    (follower loads, non-associative plasticity) and as a comparison
    point for non-symmetric workloads.

    Like :func:`jax_cg`, this path is fully JAX-native and composes
    with :func:`jax.vmap` as a single batched ``while_loop``; the
    slowest column dictates the iteration count for the batch.
    """
    return _jacobi_gmres(
        AssembledOperator(K_data, sparsity), b, rtol, restart, max_iters,
    )


def scipy_amg_cg(
        K_data: JaxArray, sparsity: EmbeddedSparsity, b: JaxArray,
        rtol: float = 1e-10, max_iters: int | None = None,
        *, pyamg_kwargs: dict | None = None,
) -> JaxArray:
    """Solve ``K x = b`` for SPD K via scipy CG preconditioned by
    pyamg's smoothed-aggregation algebraic multigrid.

    The CG iteration runs inside one :func:`jax.pure_callback`: scipy
    builds a CSR via :func:`_build_scipy_csr` from the
    :class:`EmbeddedSparsity` cache + deduped ``K_data``,
    :func:`pyamg.smoothed_aggregation_solver` sets up the hierarchy,
    its V-cycle preconditioner is exposed via ``.aspreconditioner()``,
    and :func:`scipy.sparse.linalg.cg` runs to convergence.

    AD via :func:`jax.lax.custom_linear_solve` with ``symmetric=True``;
    the matvec used for the AD path is a JAX-traceable BCSR matmul
    against the same dedup buffer (parallel to :func:`jax_cg` /
    :func:`jax_gmres` / :func:`scipy_lu`).

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
    op = AssembledOperator(K_data, sparsity)
    unique_data, matvec = op.unique_data, op.matvec
    n = sparsity.n
    pyamg_kw = pyamg_kwargs or {}

    def _scipy_amg_cg(
            unique_data_np: np.ndarray, col_np: np.ndarray,
            indptr_np: np.ndarray, b_np: np.ndarray,
    ) -> np.ndarray:
        K_csr = _build_scipy_csr(unique_data_np, col_np, indptr_np, n)
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
            unique_data, sparsity.col_indices, sparsity.indptr, rhs,
            vmap_method="expand_dims",
        )

    return lax.custom_linear_solve(
        matvec, b, solve, symmetric=True,
    )


def _block_scaling(op: TangentOperator, i: int) -> JaxArray:
    """``s_i = 1 / sqrt(|diag|)`` of field ``i``'s assembled block ``(i, i)``,
    the Jacobi scaling the Chebyshev inner applies to its block operator on
    both sides. The assembled diagonal serves the ``"schur"`` diagonal block
    too, whose own diagonal is never formed. A negligible diagonal entry is
    left unscaled, as in :func:`_symmetric_diagonal_scaling`."""
    d = jnp.sqrt(jnp.abs(op.block_diagonal(i)))
    negligible = d <= jnp.finfo(d.dtype).eps * jnp.max(d)
    return jnp.where(negligible, 1.0, 1.0 / jnp.where(negligible, 1.0, d))


_LANCZOS_STEPS = 15
_CHEBYSHEV_DEFAULT_DEGREE = 3
_CHEBYSHEV_LMIN_FRACTION = 1.0 / 30.0
_CHEBYSHEV_LMAX_SAFETY = 1.1


def _diag_block_matvec(
        op: TangentOperator, i: int, x: JaxArray, *,
        diagonal_block: str, transpose: bool,
) -> JaxArray:
    """Apply the operator for field ``i``'s diagonal block to a field-``i`` vector.

    ``"assembled"`` applies the assembled block ``(i, i)``. ``"schur"`` applies
    the approximate Schur complement
    ``(i, i) - sum_{j != i} (i, j) diag((j, j))^-1 (j, i)``, formed on the fly
    from block matvecs with nothing materialized: it approximates the other
    fields' inverses by their diagonals, pure block arithmetic with no
    assumption about what the fields are.

    ``transpose=True`` threads the flag through every block matvec, which gives
    the transpose of the forward operator, so the adjoint sweep reuses this with
    no separate derivation.
    """
    out = op.block_matvec(i, i, x, transpose=transpose)
    if diagonal_block != "schur":
        return out
    for j in range(op.num_fields):
        if j == i:
            continue
        proj = op.block_matvec(j, i, x, transpose=transpose)
        proj = proj / op.block_diagonal(j)
        out = out - op.block_matvec(i, j, proj, transpose=transpose)
    return out


def _lanczos_dominant_eigenvalue(
        matvec: Callable[[JaxArray], JaxArray], n: int, dtype: np.dtype,
) -> JaxArray:
    """Dominant eigenvalue of ``matvec`` (largest magnitude, sign kept), via Lanczos.

    Runs ``min(_LANCZOS_STEPS, n)`` symmetric Lanczos steps from a deterministic
    start and returns the resulting tridiagonal's eigenvalue of largest
    magnitude. The sign is kept, so a negative definite block (a stabilized
    pressure block) returns a negative estimate. The largest Ritz value
    approaches the dominant eigenvalue from below, so the bracket keeps its
    safety inflation. ``matvec`` is assumed symmetric.
    """
    steps = min(_LANCZOS_STEPS, n)
    q0 = jnp.arange(1, n + 1, dtype=dtype)
    q0 = q0 / jnp.linalg.norm(q0)

    def step(
            j: int,
            carry: tuple[JaxArray, JaxArray, JaxArray, JaxArray, JaxArray],
    ) -> tuple[JaxArray, JaxArray, JaxArray, JaxArray, JaxArray]:
        q, q_prev, beta_prev, alphas, betas = carry
        w = matvec(q) - beta_prev * q_prev
        alpha = jnp.dot(q, w)
        w = w - alpha * q
        beta = jnp.linalg.norm(w)
        q_next = w / jnp.where(beta > 0.0, beta, 1.0)
        return q_next, q, beta, alphas.at[j].set(alpha), betas.at[j].set(beta)

    zeros_k = jnp.zeros(steps, dtype=dtype)
    init = (
        q0, jnp.zeros_like(q0), jnp.asarray(0.0, dtype=dtype), zeros_k, zeros_k,
    )
    _q, _q_prev, _beta, alphas, betas = lax.fori_loop(0, steps, step, init)

    tridiagonal = (
        jnp.diag(alphas)
        + jnp.diag(betas[:steps - 1], 1)
        + jnp.diag(betas[:steps - 1], -1)
    )
    ritz = jnp.linalg.eigvalsh(tridiagonal)
    return ritz[jnp.argmax(jnp.abs(ritz))]


def _chebyshev_apply(
        matvec: Callable[[JaxArray], JaxArray], r: JaxArray,
        lmin: JaxArray, lmax: JaxArray, degree: int,
) -> JaxArray:
    """Approximately solve ``A z = r`` with a Chebyshev iteration of fixed degree.

    ``matvec`` applies ``A``; ``[lmin, lmax]`` brackets ``A``'s spectrum, both
    bounds the same sign, so a negative definite block is handled by passing two
    negative bounds. The result is a polynomial in ``A`` of degree ``degree``
    applied to ``r`` that approaches ``A^-1 r`` as the degree grows. Used as a
    smoother, not run to a tolerance, so it carries no inner convergence test.
    """
    theta = (lmax + lmin) / 2.0
    delta = (lmax - lmin) / 2.0
    sigma = theta / delta
    rho0 = 1.0 / sigma
    z0 = jnp.zeros_like(r)
    d0 = r / theta

    def step(
            _: int, carry: tuple[JaxArray, JaxArray, JaxArray, JaxArray],
    ) -> tuple[JaxArray, JaxArray, JaxArray, JaxArray]:
        z, res, d, rho = carry
        z = z + d
        res = res - matvec(d)
        rho_next = 1.0 / (2.0 * sigma - rho)
        d = rho * rho_next * d + (2.0 * rho_next / delta) * res
        return z, res, d, rho_next

    z, _res, _d, _rho = lax.fori_loop(0, degree, step, (z0, r, d0, rho0))
    return z


def _chebyshev_field_bounds(
        op: TangentOperator, diagonal_block: str,
) -> tuple[tuple[JaxArray, JaxArray], ...]:
    """Spectrum bracket ``(lmin, lmax)`` for each field's scaled diagonal block.

    The Chebyshev inner works on the Jacobi-scaled block
    ``S_i A_i S_i`` (:func:`_block_scaling`), whose spectrum is far better
    clustered than the block's own, so the bracket is taken there. Each
    field's dominant eigenvalue comes from a short Lanczos run on the scaled
    forward diagonal block matvec, sign kept. The dominant end is inflated
    by a safety margin: the largest Ritz value approaches it from below and
    the Chebyshev polynomial grows past the bracket, so that end must not
    sit under the true eigenvalue. The other end is a fixed fraction of it.
    The pair is ordered so ``lmin <= lmax``; both keep the sign, so a
    negative definite block yields two negative bounds. A block and its
    transpose share eigenvalues, so the same bracket serves the transpose
    sweep.
    """
    bounds: list[tuple[JaxArray, JaxArray]] = []
    offs = op.field_offsets
    for i in range(op.num_fields):
        n_i = offs[i + 1] - offs[i]
        s = _block_scaling(op, i)

        def block_matvec(x: JaxArray, i: int = i, s: JaxArray = s) -> JaxArray:
            return s * _diag_block_matvec(
                op, i, s * x, diagonal_block=diagonal_block, transpose=False,
            )

        lam = _lanczos_dominant_eigenvalue(block_matvec, n_i, s.dtype)
        lo = lam * _CHEBYSHEV_LMIN_FRACTION
        hi = lam * _CHEBYSHEV_LMAX_SAFETY
        bounds.append((jnp.minimum(lo, hi), jnp.maximum(lo, hi)))
    return tuple(bounds)


def _block_precon_apply(
        op: TangentOperator, r: JaxArray, *,
        coupling: str, diagonal_block: str, inner: str, transpose: bool,
        chebyshev_degree: int = 0,
        chebyshev_bounds: tuple[tuple[JaxArray, JaxArray], ...] | None = None,
) -> JaxArray:
    """Apply the block preconditioner once: approximately solve ``M z = r``.

    ``M`` approximates the global tangent from its field blocks. Writing
    ``r_i`` for field ``i``'s part of ``r`` and ``solve_i`` for the
    approximate inverse of field ``i``'s own block ``(i, i)``, the sweep
    over fields is:

    - ``"diagonal"`` (block Jacobi): ``z_i = solve_i(r_i)``, each field
      independent.
    - ``"lower"`` (block Gauss-Seidel): ``z_i = solve_i(r_i - sum_{j<i}
      (i,j) @ z_j)``, fields in increasing order.
    - ``"upper"``: the same in decreasing order over ``j > i``.

    ``inner`` sets the approximate inverse of each field's own block ``(i, i)``:
    ``"jacobi"`` divides by the assembled block's diagonal; ``"chebyshev"`` runs
    ``chebyshev_degree`` Chebyshev steps (spectrum bracket per field in
    ``chebyshev_bounds``) on the Jacobi-scaled diagonal block matvec
    ``S_i A_i S_i`` (:func:`_block_scaling`), returning ``S_i`` times the
    result of the scaled right hand side ``S_i r_i``: the block's spectrum
    is clustered by the scaling, so a low degree polynomial approximates its
    inverse far better. ``diagonal_block`` picks the block matvec:
    ``"assembled"`` is the block ``(i, i)`` as stored, ``"schur"`` its
    approximate Schur complement. ``"jacobi"`` pairs only with
    ``"assembled"``; ``"chebyshev"`` pairs with either.

    ``transpose=True`` runs the sweep on the transpose operator for the
    adjoint solve.
    """
    if inner not in ("jacobi", "chebyshev"):
        raise NotImplementedError(
            f"inner={inner!r} not implemented; "
            f"only 'jacobi' and 'chebyshev' are available"
        )
    if diagonal_block not in ("assembled", "schur"):
        raise NotImplementedError(
            f"diagonal_block={diagonal_block!r} not implemented; "
            f"only 'assembled' and 'schur' are available"
        )
    if inner == "jacobi" and diagonal_block != "assembled":
        raise NotImplementedError(
            "inner='jacobi' supports only diagonal_block='assembled'; "
            "use inner='chebyshev' for the schur diagonal block"
        )

    num_fields = op.num_fields
    offs = op.field_offsets
    r_fields = [r[offs[i]:offs[i + 1]] for i in range(num_fields)]
    z_fields: list[JaxArray] = [
        jnp.zeros_like(r_fields[i]) for i in range(num_fields)
    ]

    def apply_block_inverse(i: int, rhs: JaxArray) -> JaxArray:
        if inner == "jacobi":
            return rhs / op.block_diagonal(i)
        assert chebyshev_bounds is not None
        lmin, lmax = chebyshev_bounds[i]
        s = _block_scaling(op, i)

        def block_matvec(x: JaxArray) -> JaxArray:
            return s * _diag_block_matvec(
                op, i, s * x, diagonal_block=diagonal_block,
                transpose=transpose,
            )

        return s * _chebyshev_apply(
            block_matvec, s * rhs, lmin, lmax, chebyshev_degree,
        )

    order = (
        range(num_fields) if coupling != "upper"
        else range(num_fields - 1, -1, -1)
    )
    for i in order:
        rhs = r_fields[i]
        if coupling == "lower":
            for j in range(i):
                rhs = rhs - op.block_matvec(i, j, z_fields[j], transpose=transpose)
        elif coupling == "upper":
            for j in range(i + 1, num_fields):
                rhs = rhs - op.block_matvec(i, j, z_fields[j], transpose=transpose)
        z_fields[i] = apply_block_inverse(i, rhs)
    return jnp.concatenate(z_fields)


def _block_gmres(
        op: TangentOperator, b: JaxArray, *,
        coupling: str, diagonal_block: str, inner: str, degree: int | None,
        rtol: float, max_iters: int | None, restart: int,
) -> JaxArray:
    """GMRES on ``op`` with the block preconditioner
    (:func:`_block_precon_apply` over ``op``'s field blocks), differentiable
    through :func:`jax.lax.custom_linear_solve`; the adjoint solve runs on
    the auto transposed matvec with the transposed sweep."""
    if inner == "chebyshev":
        chebyshev_degree = (
            _CHEBYSHEV_DEFAULT_DEGREE if degree is None else degree
        )
        chebyshev_bounds = _chebyshev_field_bounds(op, diagonal_block)
    else:
        chebyshev_degree = 0
        chebyshev_bounds = None

    def precon_forward(r: JaxArray) -> JaxArray:
        return _block_precon_apply(
            op, r, coupling=coupling, diagonal_block=diagonal_block,
            inner=inner, transpose=False,
            chebyshev_degree=chebyshev_degree, chebyshev_bounds=chebyshev_bounds,
        )

    def precon_transpose(r: JaxArray) -> JaxArray:
        return _block_precon_apply(
            op, r, coupling=coupling, diagonal_block=diagonal_block,
            inner=inner, transpose=True,
            chebyshev_degree=chebyshev_degree, chebyshev_bounds=chebyshev_bounds,
        )

    def solve(matvec_: Callable[[JaxArray], JaxArray],
              rhs: JaxArray) -> JaxArray:
        return _gmres_solve(
            matvec_, precon_forward, rhs, rtol, restart, max_iters,
        )

    def transpose_solve(vecmat: Callable[[JaxArray], JaxArray],
                        rhs: JaxArray) -> JaxArray:
        return _gmres_solve(
            vecmat, precon_transpose, rhs, rtol, restart, max_iters,
        )

    return lax.custom_linear_solve(
        op.matvec, b, solve, transpose_solve=transpose_solve, symmetric=False,
    )


def jax_block_gmres(
        K_data: JaxArray, sparsity: EmbeddedSparsity, b: JaxArray,
        block_sparsity: BlockSparsity, *,
        coupling: str = "lower", diagonal_block: str = "assembled",
        inner: str = "jacobi", degree: int | None = None,
        rtol: float = 1e-10, max_iters: int | None = None,
        restart: int = 20,
) -> JaxArray:
    """Solve ``K x = b`` with GMRES and a block preconditioner.

    GMRES (:func:`_gmres_loop`, right preconditioned, true residual test)
    drives the same global matvec as the other solvers; the
    preconditioner is the field-block sweep of :func:`_block_precon_apply`
    over the partition in ``block_sparsity``. Fully JAX-native, so it
    composes with :func:`jax.vmap` and runs on GPU.

    Differentiation goes through :func:`jax.lax.custom_linear_solve`, with
    the preconditioner confined to the ``solve`` and ``transpose_solve``
    closures, so the derivative rules see only the global matvec --
    gradients and Hessians match the other solvers.

    ``coupling``, ``diagonal_block``, and ``inner`` select the preconditioner.
    ``degree`` matters only when ``inner="chebyshev"`` (it sets the Chebyshev
    step count); ``None`` selects the built-in default, and the jacobi path
    ignores it.
    """
    return _block_gmres(
        AssembledOperator(K_data, sparsity, block_sparsity), b,
        coupling=coupling, diagonal_block=diagonal_block, inner=inner,
        degree=degree, rtol=rtol, max_iters=max_iters, restart=restart,
    )


def _scipy_block_precon(
        unique_data_np: np.ndarray, bs: BlockSparsity,
        near_null_by_field: list[np.ndarray | None] | None, *,
        coupling: str, diagonal_block: str,
        pyamg_kwargs: dict | None, transpose: bool,
) -> scipy.sparse.linalg.LinearOperator:
    """One block-preconditioner sweep as a scipy operator, AMG per block.

    Host-side counterpart of :func:`_block_precon_apply`, for the callback
    solver. Builds each field block as a scipy CSR from the deduped global
    data, sets up one pyamg V-cycle per diagonal block as its approximate
    inverse, and returns a :class:`scipy.sparse.linalg.LinearOperator` that
    applies the block Jacobi / Gauss-Seidel sweep selected by ``coupling``.

    ``diagonal_block="assembled"`` uses the assembled diagonal block ``(i, i)``;
    ``"schur"`` uses an approximate Schur complement of it,
    ``(i,i) - sum_{j!=i} (i,j) diag((j,j))^{-1} (j,i)`` (a sparse triple product
    that approximates the other blocks' inverses by their diagonals), before the
    AMG setup. ``near_null_by_field`` supplies each
    field's near null space to pyamg. ``transpose=True`` builds the operator
    for ``K^T`` (each block transposed).
    """
    num_fields = bs.num_fields
    offs = bs.field_offsets
    n = offs[-1]
    sizes = [offs[i + 1] - offs[i] for i in range(num_fields)]
    pair_index = {pair: k for k, pair in enumerate(bs.pairs)}

    def stored_csr(a: int, c: int) -> scipy.sparse.csr_matrix | None:
        if (a, c) not in pair_index:
            return None
        k = pair_index[(a, c)]
        return scipy.sparse.csr_matrix(
            (unique_data_np[np.asarray(bs.global_data_indices[k])],
             (np.asarray(bs.local_rows[k]), np.asarray(bs.local_cols[k]))),
            shape=(sizes[a], sizes[c]),
        )

    # Operator blocks of K (or K^T): ops[(i, j)].
    ops: dict[tuple[int, int], scipy.sparse.csr_matrix | None] = {}
    for i in range(num_fields):
        for j in range(num_fields):
            block = stored_csr(j, i) if transpose else stored_csr(i, j)
            ops[(i, j)] = (
                block.T.tocsr() if (transpose and block is not None) else block
            )

    amg_apply = []
    for i in range(num_fields):
        diag_op = ops[(i, i)]
        assert diag_op is not None  # every field has a diagonal block
        diag_block = diag_op.tocsr()
        if diagonal_block == "schur":
            for j in range(num_fields):
                if j == i:
                    continue
                op_ij = ops[(i, j)]
                op_ji = ops[(j, i)]
                op_jj = ops[(j, j)]
                if op_ij is None or op_ji is None or op_jj is None:
                    continue
                inv_diag = scipy.sparse.diags(1.0 / op_jj.diagonal())
                diag_block = diag_block - op_ij @ inv_diag @ op_ji
        kwargs = dict(pyamg_kwargs or {})
        near_null = (
            near_null_by_field[i] if near_null_by_field is not None else None
        )
        if near_null is not None and near_null.shape[1] > 0:
            kwargs.setdefault("B", near_null)
        ml = pyamg.smoothed_aggregation_solver(diag_block.tocsr(), **kwargs)
        amg_apply.append(ml.aspreconditioner())

    order = (
        range(num_fields) if coupling != "upper"
        else range(num_fields - 1, -1, -1)
    )

    def apply(r: np.ndarray) -> np.ndarray:
        z = [np.zeros(sizes[i]) for i in range(num_fields)]
        for i in order:
            rhs = np.array(r[offs[i]:offs[i + 1]])
            neighbors = (
                range(i) if coupling == "lower"
                else range(i + 1, num_fields) if coupling == "upper"
                else range(0)
            )
            for j in neighbors:
                op_ij = ops[(i, j)]
                if op_ij is not None:
                    rhs = rhs - op_ij @ z[j]
            z[i] = amg_apply[i](rhs)
        return np.concatenate(z)

    return scipy.sparse.linalg.LinearOperator((n, n), matvec=apply)


def scipy_block_gmres(
        K_data: JaxArray, sparsity: EmbeddedSparsity, b: JaxArray,
        block_sparsity: BlockSparsity,
        near_null_by_field: list[np.ndarray | None] | None = None, *,
        coupling: str = "lower", diagonal_block: str = "schur",
        rtol: float = 1e-10, max_iters: int | None = None,
        restart: int = 20, pyamg_kwargs: dict | None = None,
) -> JaxArray:
    """Solve ``K x = b`` with GMRES and a block preconditioner, using AMG.

    The callback counterpart of :func:`jax_block_gmres`: the outer GMRES and
    the block preconditioner run inside one :func:`jax.pure_callback` (scipy
    + pyamg), with one algebraic-multigrid V-cycle as the approximate inverse
    of each diagonal block (:func:`_scipy_block_precon`). ``diagonal_block``
    and ``coupling`` select the preconditioner; ``near_null_by_field`` gives
    each field's near null space to pyamg.

    AD via :func:`jax.lax.custom_linear_solve` with the same global matvec the
    other solvers use, so the scipy preconditioner stays inside the solve
    callbacks and never enters the derivative rules. The adjoint solve runs
    GMRES on ``K^T`` with the transpose of the block sweep.
    """
    op = AssembledOperator(K_data, sparsity)
    unique_data, matvec = op.unique_data, op.matvec
    n = sparsity.n
    leaves, treedef = jax.tree_util.tree_flatten(block_sparsity)
    maxiter = max_iters if max_iters is not None else n

    def make_callback(transpose: bool) -> Callable[..., np.ndarray]:
        def callback(
                unique_data_np: np.ndarray, col_np: np.ndarray,
                indptr_np: np.ndarray, b_np: np.ndarray,
                *leaves_np: np.ndarray,
        ) -> np.ndarray:
            K_csr = _build_scipy_csr(unique_data_np, col_np, indptr_np, n)
            if transpose:
                K_csr = K_csr.T.tocsr()
            bs_np = jax.tree_util.tree_unflatten(treedef, list(leaves_np))
            precon = _scipy_block_precon(
                np.reshape(unique_data_np, -1), bs_np, near_null_by_field,
                coupling=coupling, diagonal_block=diagonal_block,
                pyamg_kwargs=pyamg_kwargs, transpose=transpose,
            )
            x, _info = scipy.sparse.linalg.gmres(
                K_csr, np.reshape(b_np, -1), M=precon, rtol=rtol,
                maxiter=maxiter, restart=restart,
            )
            return np.asarray(x).reshape(b_np.shape)
        return callback

    def solve(_unused_matvec: Callable[[JaxArray], JaxArray],
              rhs: JaxArray) -> JaxArray:
        return jax.pure_callback(
            make_callback(False),
            jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            unique_data, sparsity.col_indices, sparsity.indptr, rhs, *leaves,
            vmap_method="sequential",
        )

    def transpose_solve(_unused_vecmat: Callable[[JaxArray], JaxArray],
                        rhs: JaxArray) -> JaxArray:
        return jax.pure_callback(
            make_callback(True),
            jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            unique_data, sparsity.col_indices, sparsity.indptr, rhs, *leaves,
            vmap_method="sequential",
        )

    return lax.custom_linear_solve(
        matvec, b, solve, transpose_solve=transpose_solve, symmetric=False,
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

    This output is the tangent ``∂r/∂U_star`` of the embedded-BC
    residual built by :func:`_embedded_residual`, evaluated at the
    converged ``U_star``. That residual carries the
    ``(free, prescribed)`` coupling on its free rows as
    ``K[free, prescribed] · (presc_vals - U[presc])``; differentiating
    the ``-U[presc]`` factor contributes ``-K[free, prescribed]``,
    cancelling the ``+K[free, prescribed]`` from the assembled
    residual, so the prescribed columns of ``∂r/∂U_star`` vanish and
    match this output's row-and-column-zeroed structure. (Terms from
    ``K`` itself depending on ``U`` drop out: the increment
    ``presc_vals - U[presc]`` is zero at ``U_star``.) The IFT-based
    JVP rule in
    :func:`cmad.fem.nonlinear_solver._fe_newton_solve_ad_jvp` relies
    on this pairing.

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


def _embedded_residual(
        R_assembled: JaxArray, K_bcoo: BCOO, U: JaxArray,
        presc_idx: JaxArray, presc_vals: JaxArray,
        K_ii_presc: JaxArray,
) -> JaxArray:
    """Embedded-BC residual paired with :func:`_embedded_bc_enforce`.

    The Newton step solves ``K_emb · dU = -r``. :func:`_embedded_bc_enforce`
    builds ``K_emb`` by zeroing the prescribed rows and columns of the
    assembled tangent — zeroing the columns drops the
    ``(free, prescribed)`` coupling block from the operator. This helper
    puts that coupling back on the right-hand side, so a prescribed-dof
    increment still reaches the free dofs through the tangent:

    - free rows: ``R_assembled[free]`` plus the coupling
      ``K[free, prescribed] · (presc_vals - U[prescribed])``;
    - prescribed rows: ``K_ii_presc · (U[prescribed] - presc_vals)`` —
      the per-row assembled diagonal (from :func:`_embedded_bc_enforce`)
      times the BC mismatch, so the prescribed-row equation
      ``K_ii · dU = -r`` yields ``dU[prescribed] = presc_vals -
      U[prescribed]``.

    The coupling is formed as ``K_bcoo @ bc_increment`` restricted to
    the free rows, where ``bc_increment`` is the increment
    ``presc_vals - U[prescribed]`` scattered to the prescribed positions
    (zero elsewhere); the matvec's prescribed rows are discarded by the
    final overwrite. The coupling vanishes once
    ``U[prescribed] == presc_vals``, so past the first Newton step this
    is the plain assembled residual with the prescribed rows rescaled.
    """
    bc_increment = jnp.zeros_like(U).at[presc_idx].set(
        presc_vals - U[presc_idx],
    )
    r = R_assembled + K_bcoo @ bc_increment
    return r.at[presc_idx].set(
        K_ii_presc * (U[presc_idx] - presc_vals),
    )


@register_pytree_node_class
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
    - ``indptr``: ``(n+1,)`` CSR row pointers over unique entries.
    - ``col_indices``: ``(num_unique,)`` sorted column indices, one
      per unique entry.
    - ``diag_idx``: ``(n,)`` index into ``unique_data`` of each
      row's diagonal entry; lets diagonal-needing consumers (Jacobi
      preconditioner, residual checks) read the diagonal via
      ``unique_data[diag_idx]`` rather than a per-call scatter-add.

    Registered as a JAX pytree: the five arrays are the children, and
    ``num_unique`` (unique-entry count) / ``n`` (matrix size) are
    properties off the array shapes (``col_indices.shape[0]`` and
    ``indptr.shape[0] - 1``) so the registration carries no aux data.
    """
    perm: JaxArray
    segment_ids: JaxArray
    indptr: JaxArray
    col_indices: JaxArray
    diag_idx: JaxArray

    @property
    def num_unique(self) -> int:
        """Unique ``(row, col)`` entry count (``col_indices`` length)."""
        return self.col_indices.shape[0]

    @property
    def n(self) -> int:
        """Matrix size; ``indptr`` holds ``n + 1`` CSR row pointers."""
        return self.indptr.shape[0] - 1

    def tree_flatten(self) -> tuple[tuple[JaxArray, ...], None]:
        children = (
            self.perm, self.segment_ids, self.indptr,
            self.col_indices, self.diag_idx,
        )
        return children, None

    @classmethod
    def tree_unflatten(
            cls, aux_data: None, children: tuple[JaxArray, ...],
    ) -> EmbeddedSparsity:
        perm, segment_ids, indptr, col_indices, diag_idx = children
        return cls(
            perm=perm, segment_ids=segment_ids, indptr=indptr,
            col_indices=col_indices, diag_idx=diag_idx,
        )


def build_embedded_sparsity(
        fe_problem: FEProblem,
) -> EmbeddedSparsity:
    """Pre-compute the :class:`EmbeddedSparsity` for ``fe_problem``.

    Walks the deduped assembled COO from
    :func:`cmad.fem.assembly.assembled_coo_dedup`, filters
    structural zeros (entries whose row or column is prescribed),
    and appends ``(presc_idx, presc_idx)`` positions for the
    ``alpha`` diagonal block that :func:`_embedded_bc_enforce`
    adds at runtime. The lex-sort permutation, dedup segment ids,
    and CSR row pointers are derived from the kept set; ``perm`` /
    ``segment_ids`` index the ``num_unique + n_presc`` embedded-BC
    data buffer that :func:`_embedded_bc_enforce` produces from a
    deduped :class:`jax.experimental.sparse.BCOO`.

    The assembled COO is already deduplicated, so the only
    duplicates the in-solver segment-sum collapses are the
    ``(presc, presc)`` positions where an assembled diagonal entry
    coincides with an appended ``alpha`` position — folded into one
    unique entry with value ``alpha`` (zero from the runtime mask
    plus appended ``alpha``).
    """
    from cmad.fem.assembly import assembled_coo_dedup

    assembled_rows, assembled_cols, _ = assembled_coo_dedup(fe_problem)
    presc_idx = np.asarray(
        fe_problem.dof_map.prescribed_indices, dtype=np.intp,
    )
    n = int(fe_problem.dof_map.num_total_dofs)
    n_assembled = assembled_rows.shape[0]
    n_presc = presc_idx.shape[0]

    is_presc = np.zeros(n, dtype=bool)
    is_presc[presc_idx] = True

    free_free_mask = (
        ~is_presc[assembled_rows] & ~is_presc[assembled_cols]
    )
    ff_positions = np.where(free_free_mask)[0].astype(np.intp)

    appended_positions = np.arange(
        n_assembled, n_assembled + n_presc, dtype=np.intp,
    )
    kept_positions = np.concatenate([ff_positions, appended_positions])

    full_rows = np.concatenate([assembled_rows, presc_idx])
    full_cols = np.concatenate([assembled_cols, presc_idx])
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

    # The matvec is bandwidth bound, so index width is part of its cost:
    # narrower indices move less data per pass. The largest value any of
    # these arrays holds is a position in the pre-dedup COO, which bounds
    # every one of them, so one check covers all five. Both `indptr` and
    # `col_indices` must share a dtype or the CPU lowering falls back to a
    # generic implementation.
    index_dtype = _index_dtype(n_assembled + n_presc)
    return EmbeddedSparsity(
        perm=jnp.asarray(perm, dtype=index_dtype),
        segment_ids=jnp.asarray(segment_ids, dtype=index_dtype),
        indptr=jnp.asarray(indptr, dtype=index_dtype),
        col_indices=jnp.asarray(col_indices, dtype=index_dtype),
        diag_idx=jnp.asarray(diag_idx, dtype=index_dtype),
    )


@register_pytree_node_class
@dataclass(frozen=True)
class BlockSparsity:
    """Field partition of the global tangent's sparsity.

    The unknowns are grouped by field, so the global matrix splits into
    blocks, one per pair of fields. This records, for each field pair that
    holds any entries, where those entries sit in the deduped global data
    and their row and column positions within the block, so the block
    preconditioner can multiply by one field block at a time without
    re-deriving the layout every solve.

    Fields:

    - ``field_offsets``: each field's start in the global vector, with the
      total length last (length one more than the field count).
    - ``pairs``: the ``(i, j)`` field pairs that hold entries.
    - ``global_data_indices``: per pair, the position of each block entry in
      the deduped global data.
    - ``local_rows`` / ``local_cols``: per pair, the row and column index
      of each entry within the block.

    Registered as a JAX pytree: the per-pair arrays are the children;
    ``field_offsets`` and ``pairs`` are static.
    """
    field_offsets: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    global_data_indices: tuple[JaxArray, ...]
    local_rows: tuple[JaxArray, ...]
    local_cols: tuple[JaxArray, ...]

    @property
    def num_fields(self) -> int:
        """Number of fields the global vector is partitioned into."""
        return len(self.field_offsets) - 1

    def tree_flatten(
            self,
    ) -> tuple[
        tuple[tuple[JaxArray, ...], ...],
        tuple[tuple[int, ...], tuple[tuple[int, int], ...]],
    ]:
        children = (self.global_data_indices, self.local_rows, self.local_cols)
        aux_data = (self.field_offsets, self.pairs)
        return children, aux_data

    @classmethod
    def tree_unflatten(
            cls,
            aux_data: tuple[tuple[int, ...], tuple[tuple[int, int], ...]],
            children: tuple[tuple[JaxArray, ...], ...],
    ) -> BlockSparsity:
        field_offsets, pairs = aux_data
        global_data_indices, local_rows, local_cols = children
        return cls(
            field_offsets=field_offsets, pairs=pairs,
            global_data_indices=global_data_indices,
            local_rows=local_rows, local_cols=local_cols,
        )


def build_block_sparsity(
        embedded_sparsity: EmbeddedSparsity,
        block_offsets: NDArray[np.intp],
) -> BlockSparsity:
    """Build the :class:`BlockSparsity` for a field-major DOF layout.

    ``block_offsets`` gives each field's boundaries: field ``i`` owns the
    global indices ``block_offsets[i]`` up to ``block_offsets[i + 1]``.
    Each unique entry of the deduped global sparsity is sorted into its
    field block by the field of its row and the field of its column.
    """
    offsets = np.asarray(block_offsets, dtype=np.intp)
    num_fields = offsets.shape[0] - 1
    n = int(offsets[-1])
    indptr = np.asarray(embedded_sparsity.indptr)
    col_indices = np.asarray(embedded_sparsity.col_indices)
    unique_rows = np.repeat(np.arange(n, dtype=np.intp), np.diff(indptr))
    field_of_row = np.searchsorted(offsets, unique_rows, side="right") - 1
    field_of_col = np.searchsorted(offsets, col_indices, side="right") - 1

    pairs: list[tuple[int, int]] = []
    global_data_indices: list[JaxArray] = []
    local_rows: list[JaxArray] = []
    local_cols: list[JaxArray] = []
    for i in range(num_fields):
        for j in range(num_fields):
            sel = np.where((field_of_row == i) & (field_of_col == j))[0]
            if sel.shape[0] == 0:
                continue
            pairs.append((i, j))
            global_data_indices.append(jnp.asarray(sel.astype(np.intp)))
            local_rows.append(
                jnp.asarray((unique_rows[sel] - int(offsets[i])).astype(np.intp)),
            )
            local_cols.append(
                jnp.asarray((col_indices[sel] - int(offsets[j])).astype(np.intp)),
            )

    return BlockSparsity(
        field_offsets=tuple(int(x) for x in offsets),
        pairs=tuple(pairs),
        global_data_indices=tuple(global_data_indices),
        local_rows=tuple(local_rows),
        local_cols=tuple(local_cols),
    )


def _near_null_by_field(
        near_null_space: NDArray[np.floating] | None,
        block_offsets: NDArray[np.intp],
) -> list[np.ndarray | None] | None:
    """Split the global near null space into one array per field.

    Slices the rows of ``near_null_space`` by field (``block_offsets`` gives
    the field ranges) and drops the all-zero columns, so each field keeps
    only its own modes. Returns one array per field (``None`` for a field
    with no modes), or ``None`` when there is no near null space.
    """
    if near_null_space is None:
        return None
    modes = np.asarray(near_null_space)
    offsets = np.asarray(block_offsets, dtype=np.intp)
    by_field: list[np.ndarray | None] = []
    for i in range(offsets.shape[0] - 1):
        block = modes[int(offsets[i]):int(offsets[i + 1]), :]
        keep = np.any(block != 0.0, axis=0)
        by_field.append(block[:, keep] if keep.any() else None)
    return by_field
