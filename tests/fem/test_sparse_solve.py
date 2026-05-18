"""Unit tests for :mod:`cmad.fem.sparse_solve`.

Exercises :func:`scipy_lu` (forward, JVP, VJP, HVP, jit, matvec /
solve consistency) and :func:`_embedded_bc_enforce` (symmetric
prescribed-row/column + scaled-diagonal structure).

Free of FE machinery — small dense-cast matrices generated inline.
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

from cmad.fem.sparse_solve import (
    EmbeddedSparsity,
    _embedded_bc_enforce,
    jax_cg,
    jax_gmres,
    scipy_amg_cg,
    scipy_lu,
)
from cmad.typing import JaxArray


def _dense_to_cache(K_dense: np.ndarray) -> tuple[JaxArray, EmbeddedSparsity]:
    """Flatten a dense matrix into ``(K_data, EmbeddedSparsity)``.

    ``K_data`` is the row-major flattened K (length ``n²``);
    :class:`EmbeddedSparsity` treats every entry as a distinct unique
    CSR entry (no duplicates filtered). Suitable for testing the AD
    rules and matvec / solve consistency without standing up a real
    FE mesh.
    """
    n = K_dense.shape[0]
    nnz = n * n
    col_indices = np.tile(np.arange(n), n).astype(np.intp)
    indptr = np.arange(0, nnz + 1, n).astype(np.intp)
    diag_idx = (np.arange(n) * (n + 1)).astype(np.intp)
    sparsity = EmbeddedSparsity(
        perm=jnp.asarray(np.arange(nnz, dtype=np.intp)),
        segment_ids=jnp.asarray(np.arange(nnz, dtype=np.intp)),
        indptr=jnp.asarray(indptr),
        col_indices=jnp.asarray(col_indices),
        diag_idx=jnp.asarray(diag_idx),
    )
    K_data = jnp.asarray(K_dense.reshape(-1))
    return K_data, sparsity


def _random_spd(n: int, seed: int = 0) -> np.ndarray:
    """Random SPD matrix via ``A A^T + n I``."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def _random_nonsymm(n: int, seed: int = 1) -> np.ndarray:
    """Diagonal-shifted upper-triangular: non-symmetric, well-conditioned."""
    rng = np.random.default_rng(seed)
    A = np.triu(rng.standard_normal((n, n)))
    return A + n * np.eye(n)


class TestScipyLuForward(unittest.TestCase):
    """Forward-solve correctness."""

    def test_spd_matches_dense(self) -> None:
        n = 6
        K = _random_spd(n, seed=0)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(2).standard_normal(n))

        x = scipy_lu(K_data, sparsity, b)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-10, atol=1e-12)

    def test_nonsymm_matches_dense(self) -> None:
        n = 5
        K = _random_nonsymm(n, seed=1)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(3).standard_normal(n))

        x = scipy_lu(K_data, sparsity, b)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-10, atol=1e-12)


class TestScipyLuConsistency(unittest.TestCase):
    """matvec / solve consistency.

    Guards against a class of wrong-gradient bugs: if the matvec
    closure and the spsolve closure disagree about the operator,
    ``lax.custom_linear_solve``'s JVP / VJP rules silently produce
    incorrect tangents. ``K @ scipy_lu(K, b) ≈ b`` makes the
    inconsistency observable at the operator level.
    """

    def test_matvec_inverse_of_solve(self) -> None:
        n = 6
        K = _random_spd(n, seed=4)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(5).standard_normal(n))

        x = scipy_lu(K_data, sparsity, b)
        b_recovered = jnp.asarray(K) @ x
        np.testing.assert_allclose(np.asarray(b_recovered), np.asarray(b),
                                   rtol=1e-10, atol=1e-12)


def _fd_jvp(f, primals, tangents, eps=1e-6):
    """Central-difference JVP for vector-output ``f``."""
    plus = f(*[p + eps * t for p, t in zip(primals, tangents, strict=True)])
    minus = f(*[p - eps * t for p, t in zip(primals, tangents, strict=True)])
    return (plus - minus) / (2 * eps)


class TestScipyLuJVP(unittest.TestCase):
    """Forward-mode JVP correctness vs central-difference FD."""

    def test_jvp_K_and_b(self) -> None:
        n = 5
        K = _random_spd(n, seed=10)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(11).standard_normal(n))

        rng = np.random.default_rng(12)
        K_data_dot = jnp.asarray(rng.standard_normal(K_data.shape[0]))
        b_dot = jnp.asarray(rng.standard_normal(n))

        def f(K_data_, b_):
            return scipy_lu(K_data_, sparsity, b_)

        _, jvp_out = jax.jvp(f, (K_data, b), (K_data_dot, b_dot))
        jvp_fd = _fd_jvp(f, (K_data, b), (K_data_dot, b_dot), eps=1e-6)

        np.testing.assert_allclose(np.asarray(jvp_out), np.asarray(jvp_fd),
                                   rtol=1e-5, atol=1e-7)


class TestScipyLuVJP(unittest.TestCase):
    """Reverse-mode VJP correctness vs central-difference FD."""

    def test_vjp_K_and_b(self) -> None:
        n = 4
        K = _random_spd(n, seed=20)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(21).standard_normal(n))

        x_bar = jnp.asarray(np.random.default_rng(22).standard_normal(n))

        def f(K_data_, b_):
            return scipy_lu(K_data_, sparsity, b_)

        _, vjp_fn = jax.vjp(f, K_data, b)
        gK, gb = vjp_fn(x_bar)

        def J(K_data_, b_):
            return jnp.dot(x_bar, f(K_data_, b_))

        eps = 1e-6
        gK_fd = np.zeros_like(np.asarray(K_data))
        for i in range(K_data.shape[0]):
            ei = jnp.zeros_like(K_data).at[i].set(eps)
            gK_fd[i] = (J(K_data + ei, b) - J(K_data - ei, b)) / (2 * eps)
        gb_fd = np.zeros_like(np.asarray(b))
        for i in range(b.shape[0]):
            ei = jnp.zeros_like(b).at[i].set(eps)
            gb_fd[i] = (J(K_data, b + ei) - J(K_data, b - ei)) / (2 * eps)

        np.testing.assert_allclose(np.asarray(gK), gK_fd,
                                   rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(np.asarray(gb), gb_fd,
                                   rtol=1e-5, atol=1e-7)


class TestScipyLuHVP(unittest.TestCase):
    """Hessian-vector product correctness via forward-over-reverse.

    Exercises the full AD-composition path end-to-end:
    ``jax.jvp(jax.grad(J), …)`` re-enters ``scipy_lu``'s JVP rule
    with non-zero ``K_data_dot`` — the same path HVPs through the FE
    adjoint will exercise once that lands.
    """

    def test_hvp_K(self) -> None:
        n = 4
        K = _random_spd(n, seed=30)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(31).standard_normal(n))

        v = jnp.asarray(np.random.default_rng(32).standard_normal(K_data.shape[0]))

        def J(K_data_):
            x = scipy_lu(K_data_, sparsity, b)
            return 0.5 * jnp.sum(x ** 2)

        gradJ = jax.grad(J)
        _, hvp_for = jax.jvp(gradJ, (K_data,), (v,))

        eps = 1e-4
        hvp_fd = (gradJ(K_data + eps * v) - gradJ(K_data - eps * v)) / (2 * eps)

        np.testing.assert_allclose(np.asarray(hvp_for), np.asarray(hvp_fd),
                                   rtol=1e-3, atol=1e-5)


class TestScipyLuVmapOverRhs(unittest.TestCase):
    """Vmap over RHS routes through the 2D-RHS callback path.

    Confirms vmap'd :func:`scipy_lu` (single host callback with
    K broadcast to a leading length-1 axis and a batched ``b``,
    amortized LU + multi back-substitute) matches per-element
    invocations outside vmap (multiple callbacks with 1D ``b``
    each).
    """

    def test_vmap_matches_sequential(self) -> None:
        n, batch = 6, 4
        K = _random_spd(n, seed=70)
        K_data, sparsity = _dense_to_cache(K)
        B = jnp.asarray(
            np.random.default_rng(71).standard_normal((batch, n)),
        )

        out_v = jax.vmap(
            lambda b: scipy_lu(K_data, sparsity, b),
        )(B)
        out_seq = jnp.stack([
            scipy_lu(K_data, sparsity, B[i]) for i in range(batch)
        ])

        np.testing.assert_allclose(np.asarray(out_v), np.asarray(out_seq),
                                   rtol=1e-10, atol=1e-12)


class TestScipyLuJit(unittest.TestCase):
    """``jit(scipy_lu)`` round-trip on a concrete small instance."""

    def test_jit_round_trip(self) -> None:
        n = 5
        K = _random_spd(n, seed=40)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(41).standard_normal(n))

        @jax.jit
        def solve_jit(K_data_, b_):
            return scipy_lu(K_data_, sparsity, b_)

        x = solve_jit(K_data, b)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-10, atol=1e-12)


class TestJaxCgForward(unittest.TestCase):
    """Forward-solve correctness for ``jax_cg`` on SPD K.

    Directly tests the CG path on a hand-built ``EmbeddedSparsity``,
    independent of the FE-Newton driver's ``{type: cg}`` dispatch.
    Closes the coverage gap that FE tests default to ``{type:
    direct}``.
    """

    def test_spd_matches_dense(self) -> None:
        n = 10
        K = _random_spd(n, seed=50)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(51).standard_normal(n))

        x = jax_cg(K_data, sparsity, b, rtol=1e-12)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-8, atol=1e-9)


class TestScipyAmgCgForward(unittest.TestCase):
    """Forward-solve correctness for ``scipy_amg_cg`` on SPD K.

    Parallels :class:`TestJaxCgForward`; covers the pyamg-preconditioned
    branch with default settings (no ``pyamg_kwargs``, so pyamg uses
    the constant-vector near null space) and with an explicit
    ``B``-passing path so the ``pyamg_kwargs`` passthrough is exercised.
    """

    def test_spd_matches_dense(self) -> None:
        n = 10
        K = _random_spd(n, seed=80)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(81).standard_normal(n))

        x = scipy_amg_cg(K_data, sparsity, b, rtol=1e-12)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-8, atol=1e-9)

    def test_pyamg_kwargs_B_passthrough(self) -> None:
        """``pyamg_kwargs={'B': ...}`` is forwarded to
        :func:`pyamg.smoothed_aggregation_solver` and produces the
        same solution as the dense reference.
        """
        n = 10
        K = _random_spd(n, seed=82)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(83).standard_normal(n))
        B = np.ones((n, 1), dtype=np.float64)

        x = scipy_amg_cg(
            K_data, sparsity, b, rtol=1e-12,
            pyamg_kwargs={"B": B},
        )
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-8, atol=1e-9)


class TestJaxGmresForward(unittest.TestCase):
    """Forward-solve correctness for ``jax_gmres`` on non-symmetric K.

    Exercises GMRES on its native turf: a non-symmetric K where
    :func:`jax_cg` would fail. Builds K via :func:`_random_nonsymm`
    (upper-triangular + diagonally dominant), wires it through the
    same ``_dense_to_cache`` helper, and compares to a dense solve.
    """

    def test_nonsymm_matches_dense(self) -> None:
        n = 10
        K = _random_nonsymm(n, seed=60)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(61).standard_normal(n))

        x = jax_gmres(K_data, sparsity, b, rtol=1e-12)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-8, atol=1e-9)


class TestEmbeddedBCEnforce(unittest.TestCase):
    """Data-buffer layout of the embedded-BC symmetric form.

    Builds a 4×4 K_bcoo whose data is the row-major flattened dense
    matrix and marks rows/cols 1, 2 as prescribed; asserts the
    returned ``K_data`` matches the hand-written expected layout:
    the assembled segment with prescribed rows AND columns zeroed,
    followed by the appended assembled ``K_ii`` entries at the
    prescribed diagonal. ``K_ii_presc`` (the second return) carries
    those same diagonal values for the residual rescale at
    prescribed rows.
    """

    def test_data_buffer_layout(self) -> None:
        n = 4
        K_dense = np.array([
            [10.0, 1.0, 2.0, 3.0],
            [4.0, 20.0, 5.0, 6.0],
            [7.0, 8.0, 30.0, 9.0],
            [11.0, 12.0, 13.0, 40.0],
        ])
        rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        K_bcoo = BCOO(
            (jnp.asarray(K_dense.reshape(-1)),
             jnp.stack([jnp.asarray(rows.reshape(-1)),
                        jnp.asarray(cols.reshape(-1))], axis=-1)),
            shape=(n, n),
        )

        presc_idx = jnp.asarray([1, 2])
        K_data, K_ii_presc = _embedded_bc_enforce(K_bcoo, presc_idx)

        # Expected layout: assembled segment (row-major n² entries)
        # with rows 1, 2 and cols 1, 2 zeroed, followed by 2 appended
        # entries carrying the original assembled diagonal at the
        # prescribed (1, 1) and (2, 2) positions: K_dense[1,1]=20,
        # K_dense[2,2]=30.
        expected = jnp.asarray([
            10.0, 0.0, 0.0, 3.0,    # row 0: cols 1, 2 zeroed
            0.0, 0.0, 0.0, 0.0,     # row 1 fully zeroed (prescribed)
            0.0, 0.0, 0.0, 0.0,     # row 2 fully zeroed (prescribed)
            11.0, 0.0, 0.0, 40.0,   # row 3: cols 1, 2 zeroed
            20.0, 30.0,             # appended K_ii at presc rows
        ])
        np.testing.assert_allclose(np.asarray(K_data), np.asarray(expected),
                                   rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(K_ii_presc),
                                   np.asarray([20.0, 30.0]),
                                   rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
