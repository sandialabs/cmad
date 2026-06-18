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
    BlockSparsity,
    EmbeddedSparsity,
    _embedded_bc_enforce,
    _near_null_by_field,
    build_block_sparsity,
    jax_block_gmres,
    jax_cg,
    jax_gmres,
    scipy_amg_cg,
    scipy_block_gmres,
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


def _random_block_matrix(
        field_sizes: tuple[int, ...], *, symmetric: bool, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Indefinite block matrix with the given field sizes.

    Each diagonal block is SPD, with the last one negated so the whole
    matrix is indefinite; the off-diagonal blocks are small. With
    ``symmetric=True`` the cross blocks are transposes of each other,
    otherwise they are independent. Returns ``(K, field_offsets)``.
    """
    rng = np.random.default_rng(seed)
    offsets = np.concatenate([[0], np.cumsum(field_sizes)]).astype(np.intp)
    n = int(offsets[-1])
    num_fields = len(field_sizes)
    K = np.zeros((n, n))
    for a in range(num_fields):
        s, e = int(offsets[a]), int(offsets[a + 1])
        m = e - s
        block = rng.standard_normal((m, m))
        spd = block @ block.T + m * np.eye(m)
        K[s:e, s:e] = spd if a < num_fields - 1 else -spd
    for a in range(num_fields):
        for c in range(a + 1, num_fields):
            sa, ea = int(offsets[a]), int(offsets[a + 1])
            sc, ec = int(offsets[c]), int(offsets[c + 1])
            cross = 0.3 * rng.standard_normal((ea - sa, ec - sc))
            K[sa:ea, sc:ec] = cross
            K[sc:ec, sa:ea] = (
                cross.T if symmetric
                else 0.3 * rng.standard_normal((ec - sc, ea - sa))
            )
    return K, offsets


def _dense_to_block_cache(
        K_dense: np.ndarray, field_offsets: np.ndarray,
) -> tuple[JaxArray, EmbeddedSparsity, BlockSparsity]:
    """``_dense_to_cache`` plus the matching :class:`BlockSparsity`."""
    K_data, sparsity = _dense_to_cache(K_dense)
    block_sparsity = build_block_sparsity(
        sparsity, np.asarray(field_offsets, dtype=np.intp),
    )
    return K_data, sparsity, block_sparsity


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


class TestBuildBlockSparsity(unittest.TestCase):
    """:func:`build_block_sparsity` partitions the global sparsity by field."""

    def test_partition_recovers_matrix(self) -> None:
        K, offsets = _random_block_matrix((4, 2), symmetric=False, seed=100)
        K_data, _sparsity, bs = _dense_to_block_cache(K, offsets)

        self.assertEqual(bs.num_fields, 2)
        self.assertEqual(set(bs.pairs), {(0, 0), (0, 1), (1, 0), (1, 1)})

        # Reassemble each field block from its recorded entries and confirm
        # the result matches the original dense matrix.
        unique_data = np.asarray(K_data)
        recovered = np.zeros_like(K)
        for k, (i, j) in enumerate(bs.pairs):
            si, sj = int(offsets[i]), int(offsets[j])
            data = unique_data[np.asarray(bs.global_data_indices[k])]
            rows = np.asarray(bs.local_rows[k])
            cols = np.asarray(bs.local_cols[k])
            recovered[si + rows, sj + cols] = data
        np.testing.assert_allclose(recovered, K, rtol=1e-12, atol=1e-14)


class TestJaxBlockGmresForward(unittest.TestCase):
    """Forward-solve correctness for ``jax_block_gmres`` vs a dense solve.

    Covers two and three fields, symmetric and non-symmetric block
    matrices, and all three couplings -- every case through the one
    ``symmetric=False`` path.
    """

    def test_matches_dense(self) -> None:
        cases = [
            ((4, 2), True), ((4, 2), False),
            ((3, 2, 2), True), ((3, 2, 2), False),
        ]
        for seed, (field_sizes, symmetric) in enumerate(cases):
            K, offsets = _random_block_matrix(
                field_sizes, symmetric=symmetric, seed=200 + seed,
            )
            n = K.shape[0]
            K_data, sparsity, bs = _dense_to_block_cache(K, offsets)
            b = jnp.asarray(
                np.random.default_rng(300 + seed).standard_normal(n),
            )
            x_ref = np.linalg.solve(K, np.asarray(b))
            for coupling in ("diagonal", "lower", "upper"):
                with self.subTest(field_sizes=field_sizes,
                                  symmetric=symmetric, coupling=coupling):
                    x = jax_block_gmres(
                        K_data, sparsity, b, bs, coupling=coupling,
                        rtol=1e-12, max_iters=4, restart=n,
                    )
                    np.testing.assert_allclose(
                        np.asarray(x), x_ref, rtol=1e-7, atol=1e-9,
                    )


class TestJaxBlockGmresJVP(unittest.TestCase):
    """Forward-mode JVP vs central-difference FD (non-symmetric blocks)."""

    def test_jvp_K_and_b(self) -> None:
        K, offsets = _random_block_matrix((4, 2), symmetric=False, seed=400)
        n = K.shape[0]
        K_data, sparsity, bs = _dense_to_block_cache(K, offsets)
        b = jnp.asarray(np.random.default_rng(401).standard_normal(n))

        rng = np.random.default_rng(402)
        K_data_dot = jnp.asarray(rng.standard_normal(K_data.shape[0]))
        b_dot = jnp.asarray(rng.standard_normal(n))

        def f(K_data_: JaxArray, b_: JaxArray) -> JaxArray:
            return jax_block_gmres(
                K_data_, sparsity, b_, bs, coupling="lower",
                rtol=1e-12, max_iters=4, restart=n,
            )

        _, jvp_out = jax.jvp(f, (K_data, b), (K_data_dot, b_dot))
        jvp_fd = _fd_jvp(f, (K_data, b), (K_data_dot, b_dot), eps=1e-6)
        np.testing.assert_allclose(np.asarray(jvp_out), np.asarray(jvp_fd),
                                   rtol=1e-5, atol=1e-7)


class TestJaxBlockGmresVJP(unittest.TestCase):
    """Reverse-mode VJP vs central-difference FD (non-symmetric blocks).

    Exercises the ``transpose_solve`` path -- the block sweep on the
    transpose operator -- that the adjoint gradient relies on.
    """

    def test_vjp_K_and_b(self) -> None:
        K, offsets = _random_block_matrix((4, 2), symmetric=False, seed=500)
        n = K.shape[0]
        K_data, sparsity, bs = _dense_to_block_cache(K, offsets)
        b = jnp.asarray(np.random.default_rng(501).standard_normal(n))
        x_bar = jnp.asarray(np.random.default_rng(502).standard_normal(n))

        def f(K_data_: JaxArray, b_: JaxArray) -> JaxArray:
            return jax_block_gmres(
                K_data_, sparsity, b_, bs, coupling="lower",
                rtol=1e-12, max_iters=4, restart=n,
            )

        _, vjp_fn = jax.vjp(f, K_data, b)
        gK, gb = vjp_fn(x_bar)

        def J(K_data_: JaxArray, b_: JaxArray) -> JaxArray:
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

        np.testing.assert_allclose(np.asarray(gK), gK_fd, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(np.asarray(gb), gb_fd, rtol=1e-5, atol=1e-7)


class TestJaxBlockGmresHVP(unittest.TestCase):
    """Hessian-vector product via forward-over-reverse (non-symmetric)."""

    def test_hvp_K(self) -> None:
        K, offsets = _random_block_matrix((4, 2), symmetric=False, seed=600)
        n = K.shape[0]
        K_data, sparsity, bs = _dense_to_block_cache(K, offsets)
        b = jnp.asarray(np.random.default_rng(601).standard_normal(n))
        v = jnp.asarray(
            np.random.default_rng(602).standard_normal(K_data.shape[0]),
        )

        def J(K_data_: JaxArray) -> JaxArray:
            x = jax_block_gmres(
                K_data_, sparsity, b, bs, coupling="lower",
                rtol=1e-12, max_iters=4, restart=n,
            )
            return 0.5 * jnp.sum(x ** 2)

        gradJ = jax.grad(J)
        _, hvp_for = jax.jvp(gradJ, (K_data,), (v,))
        eps = 1e-4
        hvp_fd = (gradJ(K_data + eps * v) - gradJ(K_data - eps * v)) / (2 * eps)
        np.testing.assert_allclose(np.asarray(hvp_for), np.asarray(hvp_fd),
                                   rtol=1e-3, atol=1e-5)


class TestScipyBlockGmresForward(unittest.TestCase):
    """Forward solve for ``scipy_block_gmres`` (AMG inner) vs a dense solve.

    Covers the assembled and schur diagonal blocks on a non-symmetric block
    matrix, with pyamg's default near null space (``near_null_by_field`` None).
    """

    def test_matches_dense(self) -> None:
        K, offsets = _random_block_matrix((8, 4), symmetric=False, seed=700)
        n = K.shape[0]
        K_data, sparsity, bs = _dense_to_block_cache(K, offsets)
        b = jnp.asarray(np.random.default_rng(701).standard_normal(n))
        x_ref = np.linalg.solve(K, np.asarray(b))
        for diagonal_block in ("assembled", "schur"):
            with self.subTest(diagonal_block=diagonal_block):
                x = scipy_block_gmres(
                    K_data, sparsity, b, bs, None,
                    coupling="lower", diagonal_block=diagonal_block,
                    rtol=1e-12, restart=n,
                )
                np.testing.assert_allclose(
                    np.asarray(x), x_ref, rtol=1e-7, atol=1e-9,
                )


class TestScipyBlockGmresVJP(unittest.TestCase):
    """Reverse-mode VJP for ``scipy_block_gmres`` vs FD.

    Exercises the AMG transpose path (the block sweep on the transpose
    operator) the adjoint gradient relies on.
    """

    def test_vjp_K_and_b(self) -> None:
        K, offsets = _random_block_matrix((4, 2), symmetric=False, seed=710)
        n = K.shape[0]
        K_data, sparsity, bs = _dense_to_block_cache(K, offsets)
        b = jnp.asarray(np.random.default_rng(711).standard_normal(n))
        x_bar = jnp.asarray(np.random.default_rng(712).standard_normal(n))

        def f(K_data_: JaxArray, b_: JaxArray) -> JaxArray:
            return scipy_block_gmres(
                K_data_, sparsity, b_, bs, None,
                coupling="lower", diagonal_block="schur",
                rtol=1e-12, restart=n,
            )

        _, vjp_fn = jax.vjp(f, K_data, b)
        gK, gb = vjp_fn(x_bar)

        def J(K_data_: JaxArray, b_: JaxArray) -> JaxArray:
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

        np.testing.assert_allclose(np.asarray(gK), gK_fd, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(np.asarray(gb), gb_fd, rtol=1e-5, atol=1e-7)


class TestNearNullByField(unittest.TestCase):
    """``_near_null_by_field`` splits the global modes per field."""

    def test_block_structured_modes(self) -> None:
        offsets = np.array([0, 6, 9], dtype=np.intp)
        modes = np.zeros((9, 3))
        modes[:6, 0] = 1.0
        modes[:6, 1] = np.arange(6)
        modes[6:, 2] = 1.0
        by_field = _near_null_by_field(modes, offsets)
        assert by_field is not None
        self.assertEqual(by_field[0].shape, (6, 2))
        self.assertEqual(by_field[1].shape, (3, 1))
        self.assertIsNone(_near_null_by_field(None, offsets))


if __name__ == "__main__":
    unittest.main()
