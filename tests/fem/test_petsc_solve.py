"""Tests for :func:`cmad.fem.sparse_solve.petsc_solve`: the ``scipy_lu``
checks of ``test_sparse_solve.py`` on the PETSc solve for each Krylov
method, the field split on two field matrices, the refill of the held
matrix, the symmetry and definiteness guards, and the FE gradient and
Hessian FD checks. They run only where petsc4py is present.
"""
import contextlib
import importlib
import io
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.fem.sparse_solve import petsc_solve
from cmad.global_residuals.modes import GlobalResidualMode
from tests.fem.test_cudss_lu import _pattern_to_cache
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

SKIP_REASON = "needs petsc4py"
_SETTINGS = {"rtol": 1e-12, "restart": 20}
_TWO_FIELDS = {"field_names": ("u", "p"), "field_block_sizes": (3, 1)}


def petsc_available() -> bool:
    """Whether petsc4py imports."""
    try:
        importlib.import_module("petsc4py.PETSc")
    except ImportError:
        return False
    return True


def _field_rows(offsets: np.ndarray) -> list[np.ndarray]:
    return [
        np.arange(offsets[i], offsets[i + 1]) for i in range(offsets.shape[0] - 1)
    ]


def _dense_adjoint(
        K: np.ndarray, b: np.ndarray, x_bar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The cotangents of ``x = K^-1 b`` for ``x_bar``: ``(gK, gb)`` with
    ``gb = K^-T x_bar`` and ``gK = -gb x^T`` flattened row major."""
    lam = np.linalg.solve(K.T, x_bar)
    x = np.linalg.solve(K, b)
    return -np.outer(lam, x).reshape(-1), lam


@unittest.skipUnless(petsc_available(), SKIP_REASON)
class TestPetscGmres(unittest.TestCase):
    """One field, GAMG on the whole matrix, the adjoint on the transposed
    KSP."""

    def test_spd_matches_dense(self) -> None:
        n = 6
        K = _random_spd(n, seed=0)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(2).standard_normal(n))

        x = petsc_solve(K_data, sparsity, b, krylov="gmres", **_SETTINGS)
        np.testing.assert_allclose(
            np.asarray(x), np.linalg.solve(K, np.asarray(b)),
            rtol=1e-10, atol=1e-12,
        )

    def test_nonsymm_matches_dense(self) -> None:
        n = 5
        K = _random_nonsymm(n, seed=1)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(3).standard_normal(n))

        x = petsc_solve(K_data, sparsity, b, krylov="gmres", **_SETTINGS)
        np.testing.assert_allclose(
            np.asarray(x), np.linalg.solve(K, np.asarray(b)),
            rtol=1e-10, atol=1e-12,
        )

    def test_matvec_inverse_of_solve(self) -> None:
        n = 6
        K = _random_spd(n, seed=4)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(5).standard_normal(n))

        x = petsc_solve(K_data, sparsity, b, krylov="gmres", **_SETTINGS)
        np.testing.assert_allclose(
            np.asarray(jnp.asarray(K) @ x), np.asarray(b),
            rtol=1e-10, atol=1e-12,
        )

    def test_jvp_K_and_b(self) -> None:
        n = 5
        K = _random_spd(n, seed=10)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(11).standard_normal(n))
        rng = np.random.default_rng(12)
        K_data_dot = jnp.asarray(rng.standard_normal(K_data.shape[0]))
        b_dot = jnp.asarray(rng.standard_normal(n))

        def f(K_data_, b_):
            return petsc_solve(K_data_, sparsity, b_, krylov="gmres", **_SETTINGS)

        _, jvp_out = jax.jvp(f, (K_data, b), (K_data_dot, b_dot))
        jvp_fd = _fd_jvp(jax.jit(f), (K_data, b), (K_data_dot, b_dot), eps=1e-6)
        np.testing.assert_allclose(
            np.asarray(jvp_out), np.asarray(jvp_fd), rtol=1e-5, atol=1e-7,
        )

    def test_vjp_K_and_b(self) -> None:
        n = 5
        K = _random_nonsymm(n, seed=20)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(21).standard_normal(n))
        x_bar = jnp.asarray(np.random.default_rng(22).standard_normal(n))

        def f(K_data_, b_):
            return petsc_solve(K_data_, sparsity, b_, krylov="gmres", **_SETTINGS)

        _, vjp_fn = jax.vjp(f, K_data, b)
        gK, gb = vjp_fn(x_bar)
        gK_ref, gb_ref = _dense_adjoint(K, np.asarray(b), np.asarray(x_bar))
        np.testing.assert_allclose(np.asarray(gK), gK_ref, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(np.asarray(gb), gb_ref, rtol=1e-9, atol=1e-11)

    def test_hvp_K(self) -> None:
        n = 4
        K = _random_spd(n, seed=30)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(31).standard_normal(n))
        v = jnp.asarray(np.random.default_rng(32).standard_normal(K_data.shape[0]))

        def J(K_data_):
            x = petsc_solve(K_data_, sparsity, b, krylov="gmres", **_SETTINGS)
            return 0.5 * jnp.sum(x ** 2)

        gradJ = jax.jit(jax.grad(J))
        _, hvp_for = jax.jvp(gradJ, (K_data,), (v,))
        eps = 1e-4
        hvp_fd = (gradJ(K_data + eps * v) - gradJ(K_data - eps * v)) / (2 * eps)
        np.testing.assert_allclose(
            np.asarray(hvp_for), np.asarray(hvp_fd), rtol=1e-3, atol=1e-5,
        )

    def test_vmap_matches_sequential(self) -> None:
        n, batch = 6, 4
        K = _random_spd(n, seed=70)
        K_data, sparsity = _dense_to_cache(K)
        B = jnp.asarray(np.random.default_rng(71).standard_normal((batch, n)))

        def f(b_):
            return petsc_solve(K_data, sparsity, b_, krylov="gmres", **_SETTINGS)

        out_v = jax.vmap(f)(B)
        out_seq = jnp.stack([f(B[i]) for i in range(batch)])
        np.testing.assert_allclose(
            np.asarray(out_v), np.asarray(out_seq), rtol=1e-10, atol=1e-12,
        )

    def test_jit_round_trip(self) -> None:
        n = 5
        K = _random_spd(n, seed=40)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(41).standard_normal(n))

        @jax.jit
        def solve_jit(K_data_, b_):
            return petsc_solve(K_data_, sparsity, b_, krylov="gmres", **_SETTINGS)

        np.testing.assert_allclose(
            np.asarray(solve_jit(K_data, b)), np.linalg.solve(K, np.asarray(b)),
            rtol=1e-10, atol=1e-12,
        )


@unittest.skipUnless(petsc_available(), SKIP_REASON)
class TestPetscGmresTwoFields(unittest.TestCase):
    """The block Gauss-Seidel field split and its transpose on a
    nonsymmetric indefinite two field matrix."""

    def test_matches_dense(self) -> None:
        K, offsets = _random_block_matrix((6, 3), symmetric=False, seed=50)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(51).standard_normal(K.shape[0]))

        x = petsc_solve(
            K_data, sparsity, b, krylov="gmres", field_rows=_field_rows(offsets),
            **_TWO_FIELDS, **_SETTINGS,
        )
        np.testing.assert_allclose(
            np.asarray(x), np.linalg.solve(K, np.asarray(b)),
            rtol=1e-10, atol=1e-12,
        )

    def test_vjp_K_and_b(self) -> None:
        K, offsets = _random_block_matrix((6, 3), symmetric=False, seed=52)
        K_data, sparsity = _dense_to_cache(K)
        n = K.shape[0]
        b = jnp.asarray(np.random.default_rng(53).standard_normal(n))
        x_bar = jnp.asarray(np.random.default_rng(54).standard_normal(n))

        def f(K_data_, b_):
            return petsc_solve(
                K_data_, sparsity, b_, krylov="gmres",
                field_rows=_field_rows(offsets), **_TWO_FIELDS, **_SETTINGS,
            )

        _, vjp_fn = jax.vjp(f, K_data, b)
        gK, gb = vjp_fn(x_bar)
        gK_ref, gb_ref = _dense_adjoint(K, np.asarray(b), np.asarray(x_bar))
        np.testing.assert_allclose(np.asarray(gK), gK_ref, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(np.asarray(gb), gb_ref, rtol=1e-9, atol=1e-11)


@unittest.skipUnless(petsc_available(), SKIP_REASON)
class TestPetscRefill(unittest.TestCase):
    """One jitted solve called three times: new values on the same
    pattern refill the held matrix, a different pattern with the same
    array shapes rebuilds it; every solve against a dense solve."""

    def test_same_and_changed_pattern(self) -> None:
        n = 6
        diagonal = {(i, i) for i in range(n)}
        pattern_a = sorted(diagonal | {(0, 1), (1, 0), (2, 3), (3, 2)})
        pattern_b = sorted(diagonal | {(0, 2), (2, 0), (4, 5), (5, 4)})
        rng = np.random.default_rng(90)

        @jax.jit
        def solve(K_data, sparsity, b):
            return petsc_solve(
                K_data, sparsity, b, krylov="gmres", print_convergence=True,
                **_SETTINGS,
            )

        for pattern, refilled in ((pattern_a, False), (pattern_a, True),
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
                x = np.asarray(solve(
                    jnp.asarray(values), _pattern_to_cache(rows, cols, n),
                    jnp.asarray(b),
                ))
            with self.subTest(pattern=pattern, refilled=refilled):
                np.testing.assert_allclose(
                    x, np.linalg.solve(K, b), rtol=1e-10, atol=1e-12,
                )
                self.assertEqual("matrix refilled" in printed.getvalue(), refilled)
                self.assertEqual("matrix built" in printed.getvalue(), not refilled)


@unittest.skipUnless(petsc_available(), SKIP_REASON)
class TestPetscCg(unittest.TestCase):
    """CG with GAMG on a symmetric positive definite matrix, the adjoint
    on the forward KSP, and the two guards."""

    def test_spd_matches_dense(self) -> None:
        n = 6
        K = _random_spd(n, seed=95)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(96).standard_normal(n))

        x = petsc_solve(K_data, sparsity, b, krylov="cg", **_SETTINGS)
        np.testing.assert_allclose(
            np.asarray(x), np.linalg.solve(K, np.asarray(b)),
            rtol=1e-10, atol=1e-12,
        )

    def test_vjp_K_and_b(self) -> None:
        n = 5
        K = _random_spd(n, seed=99)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(100).standard_normal(n))
        x_bar = jnp.asarray(np.random.default_rng(101).standard_normal(n))

        def f(K_data_, b_):
            return petsc_solve(K_data_, sparsity, b_, krylov="cg", **_SETTINGS)

        _, vjp_fn = jax.vjp(f, K_data, b)
        gK, gb = vjp_fn(x_bar)
        gK_ref, gb_ref = _dense_adjoint(K, np.asarray(b), np.asarray(x_bar))
        np.testing.assert_allclose(np.asarray(gK), gK_ref, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(np.asarray(gb), gb_ref, rtol=1e-9, atol=1e-11)

    def test_hvp_K(self) -> None:
        n = 4
        K = _random_spd(n, seed=104)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(105).standard_normal(n))
        # A symmetric direction: the solve reads K as symmetric, so only
        # symmetric perturbations have the derivative it can be compared on.
        d = np.random.default_rng(106).standard_normal((n, n))
        v = jnp.asarray((d + d.T).reshape(-1))

        def J(K_data_):
            x = petsc_solve(K_data_, sparsity, b, krylov="cg", **_SETTINGS)
            return 0.5 * jnp.sum(x ** 2)

        gradJ = jax.jit(jax.grad(J))
        _, hvp_for = jax.jvp(gradJ, (K_data,), (v,))
        eps = 1e-4
        hvp_fd = (gradJ(K_data + eps * v) - gradJ(K_data - eps * v)) / (2 * eps)
        np.testing.assert_allclose(
            np.asarray(hvp_for), np.asarray(hvp_fd), rtol=1e-3, atol=1e-5,
        )

    def test_nonsymmetric_raises(self) -> None:
        n = 5
        K = _random_nonsymm(n, seed=102)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(103).standard_normal(n))

        with self.assertRaises(Exception) as ctx:
            np.asarray(petsc_solve(K_data, sparsity, b, krylov="cg", **_SETTINGS))
        self.assertIn("not symmetric", str(ctx.exception))

    def test_indefinite_raises(self) -> None:
        # Symmetric with both signs on the diagonal; the exact coarse
        # solve GAMG makes of a matrix this small hands CG a negative
        # ``z . r`` on the first iteration.
        K = np.diag([1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.ones(K.shape[0])

        with self.assertRaises(Exception) as ctx:
            np.asarray(petsc_solve(K_data, sparsity, b, krylov="cg", **_SETTINGS))
        self.assertIn("indefinite", str(ctx.exception))


@unittest.skipUnless(petsc_available(), SKIP_REASON)
class TestPetscMinres(unittest.TestCase):
    """MINRES with the block diagonal Schur split on a symmetric indefinite
    two field matrix, the adjoint on the forward KSP, and the symmetry
    guard."""

    def test_two_fields_matches_dense(self) -> None:
        K, offsets = _random_block_matrix((6, 3), symmetric=True, seed=60)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(61).standard_normal(K.shape[0]))

        x = petsc_solve(
            K_data, sparsity, b, krylov="minres",
            field_rows=_field_rows(offsets), **_TWO_FIELDS, **_SETTINGS,
        )
        np.testing.assert_allclose(
            np.asarray(x), np.linalg.solve(K, np.asarray(b)),
            rtol=1e-10, atol=1e-12,
        )

    def test_two_fields_vjp_K_and_b(self) -> None:
        K, offsets = _random_block_matrix((6, 3), symmetric=True, seed=62)
        K_data, sparsity = _dense_to_cache(K)
        n = K.shape[0]
        b = jnp.asarray(np.random.default_rng(63).standard_normal(n))
        x_bar = jnp.asarray(np.random.default_rng(64).standard_normal(n))

        def f(K_data_, b_):
            return petsc_solve(
                K_data_, sparsity, b_, krylov="minres",
                field_rows=_field_rows(offsets), **_TWO_FIELDS, **_SETTINGS,
            )

        _, vjp_fn = jax.vjp(f, K_data, b)
        gK, gb = vjp_fn(x_bar)
        gK_ref, gb_ref = _dense_adjoint(K, np.asarray(b), np.asarray(x_bar))
        np.testing.assert_allclose(np.asarray(gK), gK_ref, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(np.asarray(gb), gb_ref, rtol=1e-9, atol=1e-11)

    def test_nonsymmetric_raises(self) -> None:
        n = 5
        K = _random_nonsymm(n, seed=65)
        K_data, sparsity = _dense_to_cache(K)
        b = jnp.asarray(np.random.default_rng(66).standard_normal(n))

        with self.assertRaises(Exception) as ctx:
            np.asarray(
                petsc_solve(K_data, sparsity, b, krylov="minres", **_SETTINGS),
            )
        self.assertIn("not symmetric", str(ctx.exception))


@unittest.skipUnless(petsc_available(), SKIP_REASON)
class TestPetscFEGradient(unittest.TestCase):
    """The COUPLED single step check of ``test_fem_fd_checks.py`` through
    the petsc solve, gmres and cg: the gradient and the Hessian of a QoI
    of the converged displacement against central differences."""

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

    def test_gmres(self) -> None:
        self._check({"type": "petsc", "krylov": "gmres", "rtol": 1e-12})

    def test_cg(self) -> None:
        self._check({"type": "petsc", "krylov": "cg", "rtol": 1e-12})


if __name__ == "__main__":
    unittest.main()
