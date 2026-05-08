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

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from jax import lax
from jax.experimental.sparse import BCOO

from cmad.typing import JaxArray


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

    - **Forward-mode JVP.** ``x_dot = solve(b_dot − matvec_dot(x))``
      where ``matvec_dot(x)`` is JAX's JVP of ``matvec`` at ``x``
      along ``K_data_dot`` — exactly the textbook ``K_dot · x``
      sensitivity term, computed by JAX through plain forward-mode
      AD on the COO scatter-add.
    - **Reverse-mode VJP.** ``λ = transpose_solve(x_bar)``;
      ``b_bar = λ``; the ``K_data`` cotangent comes from
      ``jax.vjp(matvec, x)(−λ)`` which evaluates to
      ``−λ[K_rows] · x[K_cols]`` on ``K_data`` — the canonical
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


def _embedded_bc_enforce(
        K_bcoo: BCOO, presc_idx: JaxArray,
) -> tuple[JaxArray, JaxArray, JaxArray]:
    """Embedded-BC asymmetric form on a :class:`BCOO` tangent.

    Returns COO triplets ``(K_data, K_rows, K_cols)`` for ``K`` with:

    - prescribed rows zeroed (off-diagonal entries included), via a
      mask multiplication on ``K_bcoo.data``;
    - identity entries (``1.0``) appended on the prescribed diagonal.

    Asymmetric (rows-only) rather than symmetric (rows-and-columns):
    the symmetric form helps conditioning under iterative solvers
    but is unnecessary for sparse direct, and the asymmetric form
    has the property ``∂r/∂U`` (with ``r`` the embedded-BC residual)
    matches this output exactly — the consistency the IFT-based JVP
    rule needs.

    When ``K_bcoo`` already has an entry at a prescribed ``(i, i)``,
    the output contains two COO entries at that position: the
    original (zeroed by the row-mask) and the appended ``1.0``. Both
    consumer paths handle the duplicate correctly:
    :class:`scipy.sparse.coo_matrix` sums duplicates during
    ``.tocsr()`` conversion inside :func:`_spsolve_callback`, and
    JAX's scatter-add ``matvec`` accumulates them. The effective
    prescribed diagonal is therefore ``1`` either way.
    """
    rows = K_bcoo.indices[:, 0]
    cols = K_bcoo.indices[:, 1]
    n = K_bcoo.shape[0]
    p_mask = jnp.zeros(n, dtype=bool).at[presc_idx].set(True)

    keep = ~p_mask[rows]
    data_zeroed = K_bcoo.data * keep

    add_data = jnp.ones(presc_idx.shape[0], dtype=K_bcoo.data.dtype)

    K_data = jnp.concatenate([data_zeroed, add_data])
    K_rows = jnp.concatenate([rows, presc_idx])
    K_cols = jnp.concatenate([cols, presc_idx])
    return K_data, K_rows, K_cols
