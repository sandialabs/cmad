"""Tests for :func:`cmad.fem.sparse_solve.cudss_lu`: the ``scipy_lu``
checks of ``test_sparse_solve.py`` on the cuDSS solve. They run only
where CuPy, nvmath with cuDSS, and a CUDA device are present.
"""
import importlib
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cmad.fem.sparse_solve import cudss_lu
from tests.fem.test_sparse_solve import (
    _dense_to_cache,
    _fd_jvp,
    _random_nonsymm,
    _random_spd,
)

SKIP_REASON = "needs nvmath with cuDSS, CuPy, and a CUDA device"


def cudss_available() -> bool:
    """Whether CuPy and nvmath with cuDSS import and CuPy sees a CUDA
    device."""
    try:
        import cupy
        importlib.import_module("nvmath.sparse.advanced")
    except ImportError:
        return False
    try:
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


@unittest.skipUnless(cudss_available(), SKIP_REASON)
class TestCudssLuForward(unittest.TestCase):

    def test_spd_matches_dense(self) -> None:
        n = 6
        K = _random_spd(n, seed=0)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(2).standard_normal(n))

        x = cudss_lu(K_data, sparsity, b)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-10, atol=1e-12)

    def test_nonsymm_matches_dense(self) -> None:
        n = 5
        K = _random_nonsymm(n, seed=1)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(3).standard_normal(n))

        x = cudss_lu(K_data, sparsity, b)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-10, atol=1e-12)


@unittest.skipUnless(cudss_available(), SKIP_REASON)
class TestCudssLuConsistency(unittest.TestCase):
    """``K @ cudss_lu(K, b)`` recovers ``b``: the matvec and the solve
    agree about the operator, which the AD rules rely on."""

    def test_matvec_inverse_of_solve(self) -> None:
        n = 6
        K = _random_spd(n, seed=4)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(5).standard_normal(n))

        x = cudss_lu(K_data, sparsity, b)
        b_recovered = jnp.asarray(K) @ x
        np.testing.assert_allclose(np.asarray(b_recovered), np.asarray(b),
                                   rtol=1e-10, atol=1e-12)


@unittest.skipUnless(cudss_available(), SKIP_REASON)
class TestCudssLuJVP(unittest.TestCase):
    """Forward mode against central differences."""

    def test_jvp_K_and_b(self) -> None:
        n = 5
        K = _random_spd(n, seed=10)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(11).standard_normal(n))

        rng = np.random.default_rng(12)
        K_data_dot = jnp.asarray(rng.standard_normal(K_data.shape[0]))
        b_dot = jnp.asarray(rng.standard_normal(n))

        def f(K_data_, b_):
            return cudss_lu(K_data_, sparsity, b_)

        _, jvp_out = jax.jvp(f, (K_data, b), (K_data_dot, b_dot))
        jvp_fd = _fd_jvp(jax.jit(f), (K_data, b), (K_data_dot, b_dot), eps=1e-6)

        np.testing.assert_allclose(np.asarray(jvp_out), np.asarray(jvp_fd),
                                   rtol=1e-5, atol=1e-7)


@unittest.skipUnless(cudss_available(), SKIP_REASON)
class TestCudssLuVJP(unittest.TestCase):
    """Reverse mode against central differences, through the transposed
    factorization."""

    def test_vjp_K_and_b(self) -> None:
        n = 4
        K = _random_spd(n, seed=20)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(21).standard_normal(n))

        x_bar = jnp.asarray(np.random.default_rng(22).standard_normal(n))

        def f(K_data_, b_):
            return cudss_lu(K_data_, sparsity, b_)

        _, vjp_fn = jax.vjp(f, K_data, b)
        gK, gb = vjp_fn(x_bar)

        @jax.jit
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


@unittest.skipUnless(cudss_available(), SKIP_REASON)
class TestCudssLuHVP(unittest.TestCase):
    """Hessian vector product by forward over reverse against central
    differences of the gradient."""

    def test_hvp_K(self) -> None:
        n = 4
        K = _random_spd(n, seed=30)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(31).standard_normal(n))

        v = jnp.asarray(np.random.default_rng(32).standard_normal(K_data.shape[0]))

        def J(K_data_):
            x = cudss_lu(K_data_, sparsity, b)
            return 0.5 * jnp.sum(x ** 2)

        gradJ = jax.jit(jax.grad(J))
        _, hvp_for = jax.jvp(gradJ, (K_data,), (v,))

        eps = 1e-4
        hvp_fd = (gradJ(K_data + eps * v) - gradJ(K_data - eps * v)) / (2 * eps)

        np.testing.assert_allclose(np.asarray(hvp_for), np.asarray(hvp_fd),
                                   rtol=1e-3, atol=1e-5)


@unittest.skipUnless(cudss_available(), SKIP_REASON)
class TestCudssLuVmapOverRhs(unittest.TestCase):
    """``vmap`` over the right hand side, the one factorization applied
    to every column, against one solve per column."""

    def test_vmap_matches_sequential(self) -> None:
        n, batch = 6, 4
        K = _random_spd(n, seed=70)
        K_data, sparsity = _dense_to_cache(K)
        B = jnp.asarray(
            np.random.default_rng(71).standard_normal((batch, n)),
        )

        out_v = jax.vmap(
            lambda b: cudss_lu(K_data, sparsity, b),
        )(B)
        out_seq = jnp.stack([
            cudss_lu(K_data, sparsity, B[i]) for i in range(batch)
        ])

        np.testing.assert_allclose(np.asarray(out_v), np.asarray(out_seq),
                                   rtol=1e-10, atol=1e-12)


@unittest.skipUnless(cudss_available(), SKIP_REASON)
class TestCudssLuJit(unittest.TestCase):

    def test_jit_round_trip(self) -> None:
        n = 5
        K = _random_spd(n, seed=40)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(41).standard_normal(n))

        @jax.jit
        def solve_jit(K_data_, b_):
            return cudss_lu(K_data_, sparsity, b_)

        x = solve_jit(K_data, b)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
