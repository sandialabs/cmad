"""Tests for :func:`cmad.fem.sparse_solve.cudss_lu`: the ``scipy_lu``
checks of ``test_sparse_solve.py`` on the cuDSS solve. They run only
where CuPy, nvmath with cuDSS, and a CUDA device are present.
"""
import contextlib
import importlib
import io
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.fem.sparse_solve import EmbeddedSparsity, cudss_lu
from cmad.global_residuals.modes import GlobalResidualMode
from tests.fem.test_fem_fd_checks import (
    _J2_FD_PARAM_PATHS,
    _build_fe_problem_2x2x2,
    _compare_ad_vs_fd,
    _compare_hessian_ad_vs_fd,
    _initial_xi_by_block,
    _make_J2_model,
)
from tests.fem.test_sparse_solve import (
    _dense_to_cache,
    _fd_jvp,
    _random_block_matrix,
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


def _pattern_to_cache(
        rows: np.ndarray, cols: np.ndarray, n: int,
) -> EmbeddedSparsity:
    """:class:`EmbeddedSparsity` for an explicit pattern given as sorted
    unique ``(row, col)`` pairs, every row holding its diagonal; the
    ``K_data`` positions follow the pairs. The sparse counterpart of
    ``_dense_to_cache``."""
    nnz = rows.shape[0]
    indptr = np.searchsorted(rows, np.arange(n + 1), side="left")
    return EmbeddedSparsity(
        perm=jnp.asarray(np.arange(nnz, dtype=np.intp)),
        segment_ids=jnp.asarray(np.arange(nnz, dtype=np.intp)),
        indptr=jnp.asarray(indptr.astype(np.intp)),
        col_indices=jnp.asarray(cols.astype(np.intp)),
        diag_idx=jnp.asarray(np.where(rows == cols)[0].astype(np.intp)),
    )


@unittest.skipUnless(cudss_available(), SKIP_REASON)
class TestCudssLuPlanReuse(unittest.TestCase):
    """One jitted solve called three times: new values on the same
    pattern reuse the plan, a different pattern with the same array
    shapes remakes it; every solve against a dense solve."""

    def test_same_and_changed_pattern(self) -> None:
        n = 6
        diagonal = {(i, i) for i in range(n)}
        pattern_a = sorted(diagonal | {(0, 1), (1, 0), (2, 3), (3, 2)})
        pattern_b = sorted(diagonal | {(0, 2), (2, 0), (4, 5), (5, 4)})
        rng = np.random.default_rng(90)

        @jax.jit
        def solve(K_data: jax.Array, sparsity: EmbeddedSparsity,
                  b: jax.Array) -> jax.Array:
            return cudss_lu(K_data, sparsity, b, print_convergence=True)

        for pattern, reused in ((pattern_a, False), (pattern_a, True),
                                (pattern_b, False)):
            rows = np.array([r for r, _ in pattern], dtype=np.intp)
            cols = np.array([c for _, c in pattern], dtype=np.intp)
            values = np.where(rows == cols, float(n), 0.5)
            values = values * rng.uniform(0.5, 1.5, rows.shape[0])
            K = np.zeros((n, n))
            K[rows, cols] = values
            b = rng.standard_normal(n)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                x = solve(
                    jnp.asarray(values), _pattern_to_cache(rows, cols, n),
                    jnp.asarray(b),
                )
                x = np.asarray(x)
            with self.subTest(pattern=pattern, reused=reused):
                np.testing.assert_allclose(
                    x, np.linalg.solve(K, b), rtol=1e-10, atol=1e-12,
                )
                self.assertEqual("plan reused" in printed.getvalue(), reused)


@unittest.skipUnless(cudss_available(), SKIP_REASON)
class TestCudssLuSymmetric(unittest.TestCase):
    """The symmetric path: the lower triangle factored as LDL^T, the
    transpose solve on the same factorization. Forward on a positive
    definite and on an indefinite matrix against dense solves, and the
    VJP against central differences along symmetric perturbations of K
    (the solve reads one triangle, so only symmetric directions have a
    derivative it can be compared on)."""

    def test_spd_matches_dense(self) -> None:
        n = 6
        K = _random_spd(n, seed=95)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(96).standard_normal(n))

        x = cudss_lu(K_data, sparsity, b, symmetric=True)
        x_ref = jnp.linalg.solve(jnp.asarray(K), b)
        np.testing.assert_allclose(np.asarray(x), np.asarray(x_ref),
                                   rtol=1e-10, atol=1e-12)

    def test_indefinite_matches_dense(self) -> None:
        K, _ = _random_block_matrix((4, 2), symmetric=True, seed=97)
        n = K.shape[0]
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(98).standard_normal(n))

        x = cudss_lu(K_data, sparsity, b, symmetric=True)
        np.testing.assert_allclose(np.asarray(x), np.linalg.solve(K, np.asarray(b)),
                                   rtol=1e-10, atol=1e-12)

    def test_nonsymmetric_raises(self) -> None:
        n = 5
        K = _random_nonsymm(n, seed=102)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(103).standard_normal(n))

        with self.assertRaises(Exception) as ctx:
            np.asarray(cudss_lu(K_data, sparsity, b, symmetric=True))
        self.assertIn("not symmetric", str(ctx.exception))

    def test_vjp_K_and_b(self) -> None:
        n = 4
        K = _random_spd(n, seed=99)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(100).standard_normal(n))
        x_bar = jnp.asarray(np.random.default_rng(101).standard_normal(n))

        def f(K_data_, b_):
            return cudss_lu(K_data_, sparsity, b_, symmetric=True)

        _, vjp_fn = jax.vjp(f, K_data, b)
        gK, gb = vjp_fn(x_bar)

        @jax.jit
        def J(K_data_, b_):
            return jnp.dot(x_bar, f(K_data_, b_))

        eps = 1e-6
        gK_np = np.asarray(gK).reshape(n, n)
        for i in range(n):
            for j in range(i, n):
                direction = np.zeros((n, n))
                direction[i, j] = 1.0
                direction[j, i] = 1.0
                d = jnp.asarray(direction.reshape(-1))
                fd = (J(K_data + eps * d, b) - J(K_data - eps * d, b)) / (2 * eps)
                ad = float(np.sum(gK_np * direction))
                np.testing.assert_allclose(float(fd), ad, rtol=1e-5, atol=1e-7)
        gb_fd = np.zeros_like(np.asarray(b))
        for i in range(n):
            ei = jnp.zeros_like(b).at[i].set(eps)
            gb_fd[i] = (J(K_data, b + ei) - J(K_data, b - ei)) / (2 * eps)
        np.testing.assert_allclose(np.asarray(gb), gb_fd, rtol=1e-5, atol=1e-7)


@unittest.skipUnless(cudss_available(), SKIP_REASON)
class TestCudssFeGradient(unittest.TestCase):
    """The COUPLED single step check of ``test_fem_fd_checks.py`` through
    the cudss solve, GENERAL and symmetric: the gradient and the Hessian
    of a QoI of the converged displacement against central differences.
    """

    def _check(self, linear_solver_settings: dict) -> None:
        slope = 2e-3
        t = 1.0
        model = _make_J2_model()
        fe_problem = _build_fe_problem_2x2x2(
            model, GlobalResidualMode.COUPLED, slope,
        )
        params_at = model.parameters.values
        n_dofs = fe_problem.dof_map.num_total_dofs

        def _J(params):
            U_prev = jnp.zeros(n_dofs)
            xi_prev = _initial_xi_by_block(fe_problem)
            U_star, _ = fe_newton_solve(
                fe_problem, {"all": params},
                U_prev=U_prev, xi_prev_by_block=xi_prev, t=t,
                nonlinear_solver_settings={
                    "max iters": 30, "abs tol": 1e-10, "rel tol": 1e-10,
                },
                linear_solver_settings=linear_solver_settings,
            )
            return jnp.sum(U_star ** 2)

        J = jax.jit(_J)
        _compare_ad_vs_fd(
            self, J, jax.jit(jax.grad(_J)), params_at, _J2_FD_PARAM_PATHS,
        )
        _compare_hessian_ad_vs_fd(
            self, J, jax.jit(jax.hessian(_J)), params_at, _J2_FD_PARAM_PATHS,
        )

    def test_general(self) -> None:
        self._check({"type": "cudss"})

    def test_symmetric(self) -> None:
        self._check({"type": "cudss", "symmetric": True})


if __name__ == "__main__":
    unittest.main()
