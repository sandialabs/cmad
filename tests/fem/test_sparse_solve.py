"""Unit tests for :mod:`cmad.fem.sparse_solve`.

Exercises :func:`spsolve_jax` (forward, JVP, VJP, HVP, jit, matvec /
solve consistency) and :func:`_embedded_bc_enforce` (asymmetric
prescribed-row + identity-diagonal structure).

Free of FE machinery — small dense-cast matrices generated inline.
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

from cmad.fem.sparse_solve import _embedded_bc_enforce, spsolve_jax
from cmad.typing import JaxArray


def _dense_to_coo(K_dense: np.ndarray) -> tuple[JaxArray, JaxArray, JaxArray]:
    """Flatten a dense matrix into JAX COO triplets.

    Includes structural zeros — the tests don't require a true sparse
    pattern, just the COO triplet form spsolve_jax expects.
    """
    n = K_dense.shape[0]
    rows, cols = np.meshgrid(
        np.arange(n), np.arange(n), indexing="ij",
    )
    return (
        jnp.asarray(K_dense.reshape(-1)),
        jnp.asarray(rows.reshape(-1)),
        jnp.asarray(cols.reshape(-1)),
    )


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


class TestSpsolveJaxForward(unittest.TestCase):
    """Forward-solve correctness."""

    def test_spd_matches_dense(self) -> None:
        n = 6
        K = _random_spd(n, seed=0)
        K_data, K_rows, K_cols = _dense_to_coo(K)
        b = jnp.asarray(np.random.default_rng(2).standard_normal(n))

        x = spsolve_jax(K_data, K_rows, K_cols, n, b)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-10, atol=1e-12)

    def test_nonsymm_matches_dense(self) -> None:
        n = 5
        K = _random_nonsymm(n, seed=1)
        K_data, K_rows, K_cols = _dense_to_coo(K)
        b = jnp.asarray(np.random.default_rng(3).standard_normal(n))

        x = spsolve_jax(K_data, K_rows, K_cols, n, b)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-10, atol=1e-12)


class TestSpsolveJaxConsistency(unittest.TestCase):
    """matvec / solve consistency.

    Guards against a class of wrong-gradient bugs: if the matvec
    closure and the spsolve closure disagree about the operator,
    ``lax.custom_linear_solve``'s JVP / VJP rules silently produce
    incorrect tangents. This test makes the inconsistency observable
    at the operator level — ``matvec(spsolve_jax(K, b)) ≈ b``.
    """

    def test_matvec_inverse_of_solve(self) -> None:
        n = 6
        K = _random_spd(n, seed=4)
        K_data, K_rows, K_cols = _dense_to_coo(K)
        b = jnp.asarray(np.random.default_rng(5).standard_normal(n))

        x = spsolve_jax(K_data, K_rows, K_cols, n, b)
        b_recovered = jnp.zeros_like(x).at[K_rows].add(K_data * x[K_cols])
        np.testing.assert_allclose(np.asarray(b_recovered), np.asarray(b),
                                   rtol=1e-10, atol=1e-12)


def _fd_jvp(f, primals, tangents, eps=1e-6):
    """Central-difference JVP for vector-output ``f``."""
    plus = f(*[p + eps * t for p, t in zip(primals, tangents, strict=True)])
    minus = f(*[p - eps * t for p, t in zip(primals, tangents, strict=True)])
    return (plus - minus) / (2 * eps)


class TestSpsolveJaxJVP(unittest.TestCase):
    """Forward-mode JVP correctness vs central-difference FD."""

    def test_jvp_K_and_b(self) -> None:
        n = 5
        K = _random_spd(n, seed=10)
        K_data, K_rows, K_cols = _dense_to_coo(K)
        b = jnp.asarray(np.random.default_rng(11).standard_normal(n))

        rng = np.random.default_rng(12)
        K_data_dot = jnp.asarray(rng.standard_normal(K_data.shape[0]))
        b_dot = jnp.asarray(rng.standard_normal(n))

        def f(K_data_, b_):
            return spsolve_jax(K_data_, K_rows, K_cols, n, b_)

        _, jvp_out = jax.jvp(f, (K_data, b), (K_data_dot, b_dot))
        jvp_fd = _fd_jvp(f, (K_data, b), (K_data_dot, b_dot), eps=1e-6)

        np.testing.assert_allclose(np.asarray(jvp_out), np.asarray(jvp_fd),
                                   rtol=1e-5, atol=1e-7)


class TestSpsolveJaxVJP(unittest.TestCase):
    """Reverse-mode VJP correctness vs central-difference FD."""

    def test_vjp_K_and_b(self) -> None:
        n = 4
        K = _random_spd(n, seed=20)
        K_data, K_rows, K_cols = _dense_to_coo(K)
        b = jnp.asarray(np.random.default_rng(21).standard_normal(n))

        x_bar = jnp.asarray(np.random.default_rng(22).standard_normal(n))

        def f(K_data_, b_):
            return spsolve_jax(K_data_, K_rows, K_cols, n, b_)

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


class TestSpsolveJaxHVP(unittest.TestCase):
    """Hessian-vector product correctness via forward-over-reverse.

    Exercises the full AD-composition path end-to-end:
    ``jax.jvp(jax.grad(J), …)`` re-enters ``spsolve_jax``'s JVP rule
    with non-zero ``K_data_dot`` — the same path HVPs through the FE
    adjoint will exercise once that lands.
    """

    def test_hvp_K(self) -> None:
        n = 4
        K = _random_spd(n, seed=30)
        K_data, K_rows, K_cols = _dense_to_coo(K)
        b = jnp.asarray(np.random.default_rng(31).standard_normal(n))

        v = jnp.asarray(np.random.default_rng(32).standard_normal(K_data.shape[0]))

        def J(K_data_):
            x = spsolve_jax(K_data_, K_rows, K_cols, n, b)
            return 0.5 * jnp.sum(x ** 2)

        gradJ = jax.grad(J)
        _, hvp_for = jax.jvp(gradJ, (K_data,), (v,))

        eps = 1e-4
        hvp_fd = (gradJ(K_data + eps * v) - gradJ(K_data - eps * v)) / (2 * eps)

        np.testing.assert_allclose(np.asarray(hvp_for), np.asarray(hvp_fd),
                                   rtol=1e-3, atol=1e-5)


class TestSpsolveJaxJit(unittest.TestCase):
    """``jit(spsolve_jax)`` round-trip on a concrete small instance."""

    def test_jit_round_trip(self) -> None:
        n = 5
        K = _random_spd(n, seed=40)
        K_data, K_rows, K_cols = _dense_to_coo(K)
        b = jnp.asarray(np.random.default_rng(41).standard_normal(n))

        @jax.jit
        def solve_jit(K_data_, b_):
            return spsolve_jax(K_data_, K_rows, K_cols, n, b_)

        x = solve_jit(K_data, b)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-10, atol=1e-12)


class TestEmbeddedBCEnforce(unittest.TestCase):
    """Structure of the embedded-BC asymmetric form.

    Builds a 4x4 K_bcoo with non-zero off-diagonal entries in a row
    that's then marked prescribed; verifies the output is
    ``(0, …, 0, 1, 0, …, 0)`` on that row (identity-only-on-diagonal),
    with non-prescribed rows unchanged.
    """

    def test_zeroes_off_diagonal_in_prescribed_row(self) -> None:
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
        K_data, K_rows, K_cols = _embedded_bc_enforce(K_bcoo, presc_idx)

        K_csr_like = np.zeros((n, n))
        np.add.at(
            K_csr_like,
            (np.asarray(K_rows), np.asarray(K_cols)),
            np.asarray(K_data),
        )

        expected = K_dense.copy()
        expected[1, :] = 0.0
        expected[2, :] = 0.0
        expected[1, 1] = 1.0
        expected[2, 2] = 1.0
        np.testing.assert_allclose(K_csr_like, expected,
                                   rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
