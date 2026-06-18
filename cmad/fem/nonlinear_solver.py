"""Global Newton driver for the FE forward problem."""
from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from numpy.typing import NDArray

from cmad.fem.assembly import assemble_global
from cmad.fem.fe_problem import FEProblem
from cmad.fem.kernel_arrays import FEKernelArrays
from cmad.fem.sparse_solve import (
    _bcsr_operator,
    _embedded_bc_enforce,
    _embedded_residual,
    _near_null_by_field,
    jax_block_gmres,
    jax_cg,
    jax_gmres,
    scipy_amg_cg,
    scipy_block_gmres,
    scipy_lu,
)
from cmad.typing import JaxArray, Params, Scalar
from cmad.util.line_search import DEFAULT_LINE_SEARCH_SETTINGS, line_search

_DEFAULT_NONLINEAR_SOLVER_SETTINGS: dict[str, Any] = {
    "max iters": 20,
    "abs tol": 1.0e-10,
    "rel tol": 1.0e-10,
    "print convergence": False,
    "line search": DEFAULT_LINE_SEARCH_SETTINGS,
}
_DEFAULT_LINEAR_SOLVER_SETTINGS: dict[str, Any] = {
    "type": "direct",
    "rtol": 1.0e-10,
    "max iters": None,
    "restart": 20,
    "preconditioner": {"type": "jacobi"},
}


class _FrozenDict(tuple):
    """Hashable dict-like wrapper for ``custom_jvp`` ``nondiff_argnums``.

    Subclasses :class:`tuple` and carries ``(key, value)`` pairs in
    sorted-key order. The subclass marker disambiguates a frozen dict
    from a frozen list at :func:`_thaw` time.
    """


def _freeze(value: Any) -> Any:
    """Recursively convert a dict tree (with nested dicts / lists) to a
    hashable structure for ``custom_jvp`` ``nondiff_argnums``.

    Dicts become :class:`_FrozenDict` (a sorted, hashable tuple
    subclass); lists become plain tuples; everything else passes
    through unchanged.
    """
    if isinstance(value, Mapping):
        return _FrozenDict(
            (k, _freeze(v)) for k, v in sorted(value.items())
        )
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Inverse of :func:`_freeze`. Restores plain Python dict / list trees."""
    if isinstance(value, _FrozenDict):
        return {k: _thaw(v) for k, v in value}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _solve_linear(
        K: JaxArray,
        fe_problem: FEProblem,
        fe_arrays: FEKernelArrays,
        rhs: JaxArray,
        linear_solver_settings: dict[str, Any],
) -> JaxArray:
    """Dispatch on ``settings['type']`` to direct / CG / GMRES, with the
    iterative arms picking a preconditioner from
    ``settings['preconditioner']``: Jacobi or pyamg for CG, Jacobi or a
    block preconditioner for GMRES (:func:`jax_block_gmres` with a Jacobi
    or Chebyshev inner solve, :func:`scipy_block_gmres` with an AMG inner
    solve).

    :attr:`FEProblem.near_null_space` is auto-merged into pyamg
    ``kwargs`` as ``B`` when present and the caller hasn't already set
    it.
    """
    sparsity = fe_arrays.embedded_sparsity
    kind = linear_solver_settings["type"]
    if kind == "direct":
        return scipy_lu(K, sparsity, rhs)

    precon_spec = linear_solver_settings.get(
        "preconditioner", {"type": "jacobi"},
    )
    precon = precon_spec["type"]

    if kind == "cg":
        if precon == "jacobi":
            return jax_cg(
                K, sparsity, rhs,
                rtol=linear_solver_settings["rtol"],
                max_iters=linear_solver_settings["max iters"],
            )
        if precon == "pyamg":
            kwargs = dict(precon_spec.get("kwargs") or {})
            if "B" not in kwargs and fe_problem.near_null_space is not None:
                kwargs["B"] = fe_problem.near_null_space
            return scipy_amg_cg(
                K, sparsity, rhs,
                rtol=linear_solver_settings["rtol"],
                max_iters=linear_solver_settings["max iters"],
                pyamg_kwargs=kwargs,
            )
        raise ValueError(
            f"unknown preconditioner type {precon!r} for cg; "
            f"expected 'jacobi' or 'pyamg'"
        )
    if kind == "gmres":
        if precon == "jacobi":
            return jax_gmres(
                K, sparsity, rhs,
                rtol=linear_solver_settings["rtol"],
                max_iters=linear_solver_settings["max iters"],
                restart=linear_solver_settings["restart"],
            )
        if precon == "block":
            block_sparsity = fe_arrays.block_sparsity
            if block_sparsity is None:
                raise ValueError(
                    "block preconditioner requires a problem with more "
                    "than one residual block"
                )
            coupling = precon_spec.get("coupling", "lower")
            diagonal_block = precon_spec.get("diagonal_block", "assembled")
            inner = precon_spec.get("inner", "jacobi")
            if inner in ("jacobi", "chebyshev"):
                return jax_block_gmres(
                    K, sparsity, rhs, block_sparsity,
                    coupling=coupling, diagonal_block=diagonal_block,
                    inner=inner, degree=precon_spec.get("degree"),
                    rtol=linear_solver_settings["rtol"],
                    max_iters=linear_solver_settings["max iters"],
                    restart=linear_solver_settings["restart"],
                )
            if inner == "amg":
                near_null = _near_null_by_field(
                    fe_problem.near_null_space,
                    fe_problem.dof_map.block_offsets,
                )
                return scipy_block_gmres(
                    K, sparsity, rhs, block_sparsity, near_null,
                    coupling=coupling, diagonal_block=diagonal_block,
                    rtol=linear_solver_settings["rtol"],
                    max_iters=linear_solver_settings["max iters"],
                    restart=linear_solver_settings["restart"],
                )
            raise ValueError(
                f"unknown inner solve {inner!r} for the block "
                f"preconditioner; expected 'jacobi', 'chebyshev', or 'amg'"
            )
        if precon == "pyamg":
            raise NotImplementedError(
                "pyamg preconditioner with gmres is not implemented; "
                "use type='cg' with preconditioner.type='pyamg' for SPD K"
            )
        raise ValueError(
            f"unknown preconditioner type {precon!r} for gmres; "
            f"expected 'jacobi' or 'block'"
        )
    raise ValueError(
        f"unknown linear solver type {kind!r}; "
        f"expected 'direct', 'cg', or 'gmres'"
    )


def _fe_newton_primal(
        fe_problem: FEProblem,
        fe_arrays: FEKernelArrays,
        params_by_block: Mapping[str, Params],
        U_prev: JaxArray,
        xi_prev_by_block: Mapping[str, JaxArray],
        t: Scalar,
        nonlinear_solver_settings: dict[str, Any],
        linear_solver_settings: dict[str, Any],
) -> tuple[JaxArray, dict[str, JaxArray]]:
    """Forward Newton iteration: ``lax.while_loop`` + linear-solver dispatch.

    Each step assembles ``(K_bcoo, R)`` via :func:`assemble_global`,
    builds the embedded-BC tangent ``K`` via
    :func:`_embedded_bc_enforce` and the matching residual ``r`` via
    :func:`_embedded_residual`, and solves ``K · dU = -r`` via the
    linear solver named in ``linear_solver_settings['type']`` (one of
    ``direct``, ``cg``, ``gmres``). ``cond`` checks the residual norm
    against the absolute and relative tolerances.

    Returns ``(U_star, xi_star)``: the converged displacement and the
    solved state at it.
    """
    max_iters = nonlinear_solver_settings["max iters"]
    abs_tol = nonlinear_solver_settings["abs tol"]
    rel_tol = nonlinear_solver_settings["rel tol"]
    print_global_convergence = nonlinear_solver_settings["print convergence"]
    ls_settings = {
        **DEFAULT_LINE_SEARCH_SETTINGS,
        **nonlinear_solver_settings.get("line search", {}),
    }
    ls_max_evals = ls_settings["max evals"]

    dof_map = fe_problem.dof_map
    sparsity = fe_arrays.embedded_sparsity
    presc_idx = fe_arrays.prescribed_indices
    presc_vals = jnp.asarray(
        dof_map.evaluate_prescribed_values(fe_arrays.dbc_arrays, t),
    )

    U_init = U_prev

    def _assemble_enforced(U):
        K_bcoo, R_assembled, xi = assemble_global(
            fe_problem, fe_arrays, params_by_block,
            U, U_prev, t,
            xi_prev_by_block=xi_prev_by_block,
        )
        K, K_ii_presc = _embedded_bc_enforce(K_bcoo, presc_idx)
        r = _embedded_residual(
            R_assembled, K_bcoo, U, presc_idx, presc_vals, K_ii_presc,
        )
        return r, K, xi

    r_init, K_init, xi_init = _assemble_enforced(U_init)
    R0 = jnp.maximum(jnp.linalg.norm(r_init), abs_tol)

    def _print_line(k, r):
        if print_global_convergence:
            R_norm = jnp.linalg.norm(r)
            jax.debug.print(" > ({k}) Newton iteration", k=k)
            jax.debug.print(
                " > absolute ||R|| = {abs_r:.6e}", abs_r=R_norm,
            )
            jax.debug.print(
                " > relative ||R|| = {rel_r:.6e}", rel_r=R_norm / R0,
            )

    _print_line(1, r_init)

    def cond(state):
        i, r, _, _, _ = state
        R_norm = jnp.linalg.norm(r)
        return (i < max_iters) & (R_norm >= abs_tol) & (
            R_norm >= rel_tol * R0
        )

    def body(state):
        i, r, K, U, xi = state
        dU = _solve_linear(
            K, fe_problem, fe_arrays, -r, linear_solver_settings,
        )
        if ls_max_evals > 0:
            r_norm_sq = r @ r

            def eval_fn(alpha):
                r_trial, K_trial, xi_trial = _assemble_enforced(U + alpha * dU)
                _, matvec = _bcsr_operator(K_trial, sparsity)
                slope = r_trial @ matvec(dU)
                phi = 0.5 * (r_trial @ r_trial)
                return phi, slope, (r_trial, K_trial, xi_trial)

            alpha, (r_new, K_new, xi_new) = line_search(
                eval_fn, 0.5 * r_norm_sq, -r_norm_sq, ls_settings, (r, K, xi),
            )
            U_new = U + alpha * dU
        else:
            U_new = U + dU
            r_new, K_new, xi_new = _assemble_enforced(U_new)
        _print_line(i + 2, r_new)
        return (i + 1, r_new, K_new, U_new, xi_new)

    _, _, _, U_star, xi_star = lax.while_loop(
        cond, body, (0, r_init, K_init, U_init, xi_init),
    )
    return U_star, xi_star


def fe_newton_solve(
        fe_problem: FEProblem,
        params_by_block: Mapping[str, Params],
        U_prev: NDArray[np.floating] | JaxArray,
        xi_prev_by_block: Mapping[str, NDArray[np.floating] | JaxArray]
        | None = None,
        t: float = 0.0,
        nonlinear_solver_settings: dict[str, Any] | None = None,
        linear_solver_settings: dict[str, Any] | None = None,
) -> tuple[JaxArray, dict[str, JaxArray]]:
    """Quasi-static global Newton driver for the FE forward problem.

    Nonlinear convention: ``K = dR/dU`` is the tangent stiffness and
    ``R(U) = R_int(U) - F_ext`` is the residual (body force folded
    into ``R`` by the assembly — no separate ``F`` vector). Each
    Newton step solves ``K · dU = -r`` for the embedded-BC residual
    ``r`` (built by
    :func:`cmad.fem.sparse_solve._embedded_residual`) defined as
    ``r[free] = R(U)[free] + K[free, prescribed] ·
    (prescribed_vals(t) - U[prescribed])`` and
    ``r[prescribed] = K_ii · (U[prescribed] - prescribed_vals(t))``,
    where ``K_ii`` is the assembled diagonal at the prescribed row
    (per-row, surfaced by :func:`_embedded_bc_enforce`). The
    ``K[free, prescribed]`` term carries the coupling that the
    symmetric column-zeroing drops from ``K``, so a prescribed-dof
    increment reaches the interior through the tangent; it vanishes
    once ``U[prescribed] == prescribed_vals(t)``.

    Forward iteration is :func:`jax.lax.while_loop` over Newton
    steps. Each body call assembles ``(K_bcoo, R)`` once, builds the
    embedded-BC tangent via
    :func:`cmad.fem.sparse_solve._embedded_bc_enforce` (symmetric
    form: prescribed rows AND columns zeroed, original assembled
    ``K_ii`` on the prescribed diagonal — block-diagonal
    ``K_ff | diag(K_ii)``) and the matching residual via
    :func:`cmad.fem.sparse_solve._embedded_residual`, and solves
    ``K · dU = -r`` via the linear solver chosen by
    ``linear_solver_settings['type']``: ``direct`` (sparse direct
    via :func:`scipy.sparse.linalg.spsolve` through
    :func:`jax.pure_callback`), ``cg`` (JAX-native CG), or ``gmres``
    (JAX-native restarted GMRES).

    AD over the converged ``(U_star, xi_star)`` is provided by an
    inner :func:`jax.custom_jvp` rule. The JVP rule is the IFT
    linear sensitivity equation
    ``K · U_star_dot = -∂r/∂(p) · p_dot`` solved through the same
    linear-solver dispatch: the ``K``-side cotangent flows
    automatically via the underlying solver's VJP rule. JAX
    auto-transposes the JVP for :func:`jax.grad`; HVPs via
    forward-over-reverse re-invoke the JVP rule with non-zero
    ``K_data_dot``. ``custom_jvp`` (rather than ``custom_vjp``)
    keeps forward-mode AD available for HVPs and
    :func:`jax.hessian`.

    Initial iterate is ``U_prev`` — the previous step's converged
    solution used directly as the quasi-static warm start, including
    its prescribed dofs. The current targets enter through the
    ``K[free, prescribed]`` coupling term in ``r``: the first Newton
    step moves the boundary to ``prescribed_vals(t)`` and the
    interior in response together. Same driver handles homogeneous
    and non-homogeneous Dirichlet.

    ``params_by_block`` is required and threads explicitly through
    the assembly call chain — pass tracer-leaved per-block params
    for AD callers, or build via
    :func:`cmad.fem.assembly.params_by_block_from_models` for
    imperative callers.

    ``xi_prev_by_block`` is the previous time-step's converged xi
    keyed by COUPLED block; required when the FE problem has any
    COUPLED block, ignored otherwise. ``xi_prev`` stays fixed
    across global Newton iterations; the per-IP local Newton inside
    the COUPLED kernel re-solves for ``xi(U_iter, xi_prev)`` every
    iteration. Returned ``xi_star`` is the converged state at
    ``U_star``. Empty dict for CLOSED_FORM-only problems. A
    missing COUPLED-block entry surfaces as a
    ``ValueError`` from
    :func:`cmad.fem.assembly.assemble_element_block` on the first
    body iteration.

    ``nonlinear_solver_settings`` is a dict with keys
    ``max iters`` / ``abs tol`` / ``rel tol`` / ``print convergence`` /
    ``line search``; omitted keys fall back to
    :data:`_DEFAULT_NONLINEAR_SOLVER_SETTINGS`. ``line search`` is a dict
    (the keys of
    :data:`cmad.util.line_search.DEFAULT_LINE_SEARCH_SETTINGS`); its
    ``max evals = 0`` disables the search and takes the full Newton step.
    ``linear_solver_settings`` is a dict with keys
    ``type`` / ``rtol`` / ``max iters`` / ``restart`` / ``preconditioner``
    (``restart`` consumed only by ``gmres``; ``preconditioner`` ignored
    when ``type='direct'``). ``preconditioner`` is itself a dict with
    a required ``type`` (``'jacobi'``, ``'pyamg'``, or ``'block'``). pyamg
    takes an optional freeform ``kwargs`` dict forwarded to
    :func:`pyamg.smoothed_aggregation_solver`; block takes ``coupling``
    (``'diagonal'`` / ``'lower'`` / ``'upper'``), ``diagonal_block``
    (``'assembled'`` / ``'schur'``), ``inner`` (``'jacobi'`` / ``'chebyshev'``
    / ``'amg'``), and ``degree`` (the Chebyshev step count). Omitted keys fall
    back to :data:`_DEFAULT_LINEAR_SOLVER_SETTINGS`.

    Returns ``(U_star, xi_star)``. Outputs are JAX arrays.
    """
    nls = {
        **_DEFAULT_NONLINEAR_SOLVER_SETTINGS,
        **(nonlinear_solver_settings or {}),
    }
    lss = {
        **_DEFAULT_LINEAR_SOLVER_SETTINGS,
        **(linear_solver_settings or {}),
    }
    U_prev_jax = jnp.asarray(U_prev, dtype=jnp.float64)
    xi_prev_jax: dict[str, JaxArray] = (
        {k: jnp.asarray(v) for k, v in xi_prev_by_block.items()}
        if xi_prev_by_block is not None else {}
    )
    return _fe_newton_solve_ad(
        fe_problem, fe_problem.kernel_arrays, params_by_block,
        U_prev_jax, xi_prev_jax, t, _freeze(nls), _freeze(lss),
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 6, 7))
def _fe_newton_solve_ad(
        fe_problem: FEProblem,
        fe_arrays: FEKernelArrays,
        params_by_block: Mapping[str, Params],
        U_prev: JaxArray,
        xi_prev_by_block: dict[str, JaxArray],
        t: Scalar,
        nonlinear_solver_settings_frozen: tuple[tuple[str, Any], ...],
        linear_solver_settings_frozen: tuple[tuple[str, Any], ...],
) -> tuple[JaxArray, dict[str, JaxArray]]:
    """AD-decorated inner driver. JaxArray inputs only.

    Splitting the public ``fe_newton_solve`` from this inner form
    keeps the boundary ``np.ndarray → jnp.ndarray`` conversion
    outside the ``custom_jvp``-tracked function body, so the diff
    args are uniformly typed for the JVP rule. ``t`` stays in the
    diff set: when the driver runs inside a :func:`jax.lax.scan`
    over the time schedule, the per-step ``t`` is a tracer (one
    slice of the scan's traced input), and ``nondiff_argnums``
    requires hashable Python values. The JVP rule threads a
    ``t_dot`` tangent that no current consumer populates. The two
    settings dicts are passed as :func:`_freeze`'d tuples so they
    are hashable for ``custom_jvp``'s nondiff-arg cache.
    """
    nls = _thaw(nonlinear_solver_settings_frozen)
    lss = _thaw(linear_solver_settings_frozen)
    U_star, xi_star = _fe_newton_primal(
        fe_problem, fe_arrays, params_by_block, U_prev, xi_prev_by_block,
        t, nls, lss,
    )
    return U_star, xi_star


@_fe_newton_solve_ad.defjvp
def _fe_newton_solve_ad_jvp(
        fe_problem: FEProblem,
        nonlinear_solver_settings_frozen: tuple[tuple[str, Any], ...],
        linear_solver_settings_frozen: tuple[tuple[str, Any], ...],
        primals, tangents,
):
    """IFT linear-sensitivity JVP for :func:`_fe_newton_solve_ad`.

    For ``r(U, p) = 0`` with ``p = (params, U_prev, xi_prev, t)``,
    ``U_star_dot = -K^{-1} · (∂r/∂p · p_dot)`` with ``K = ∂r/∂U`` at
    ``U_star``. ``∂r/∂p · p_dot`` is computed by
    ``jax.jvp(r, p, p_dot)`` at fixed ``U_star``; ``K`` is the
    ``_embedded_bc_enforce``-applied assembled tangent at ``U_star``;
    the linear solve goes through :func:`_solve_linear` so the ``K``
    cotangent flows automatically when JAX auto-transposes the rule.
    ``xi_star_dot`` follows from chain rule: the assembly's xi
    output is differentiated jointly w.r.t. ``U_star`` (with tangent
    ``U_star_dot``) and w.r.t. ``p`` (with tangent ``p_dot``).
    ``t_dot`` is threaded as ceremony — no current consumer
    populates a non-zero ``t_dot`` — but ``t`` itself stays in the
    primals tuple because under :func:`jax.lax.scan` it is a
    tracer and cannot ride in ``nondiff_argnums``.
    """
    fe_arrays, params_by_block, U_prev, xi_prev_by_block, t = primals
    p_dot = tangents[1:]  # tangents[0]: fe_arrays tangent, unused

    lss = _thaw(linear_solver_settings_frozen)

    U_star, xi_star = _fe_newton_solve_ad(
        fe_problem, fe_arrays, params_by_block, U_prev, xi_prev_by_block,
        t, nonlinear_solver_settings_frozen,
        linear_solver_settings_frozen,
    )

    presc_idx = fe_arrays.prescribed_indices

    # Trailing-underscore params (params_ <-> params_by_block, Up_ <->
    # U_prev, xp_ <-> xi_prev_by_block, t_ <-> t) are this helper's explicit
    # jvp-differentiated inputs; U_star is captured, held fixed by the IFT.
    def r_of_p(params_, Up_, xp_, t_):
        pv = jnp.asarray(
            fe_problem.dof_map.evaluate_prescribed_values(
                fe_arrays.dbc_arrays, t_,
            ),
        )
        K_bcoo_local, R_local, _ = assemble_global(
            fe_problem, fe_arrays, params_,
            U_star, Up_, t_,
            xi_prev_by_block=xp_,
        )
        _, K_ii_presc_local = _embedded_bc_enforce(
            K_bcoo_local, presc_idx,
        )
        return _embedded_residual(
            R_local, K_bcoo_local, U_star, presc_idx, pv,
            K_ii_presc_local,
        )

    _, Rp_dot = jax.jvp(
        r_of_p,
        (params_by_block, U_prev, xi_prev_by_block, t),
        p_dot,
    )

    K_bcoo, _, _ = assemble_global(
        fe_problem, fe_arrays, params_by_block,
        U_star, U_prev, t,
        xi_prev_by_block=xi_prev_by_block,
    )
    K, _ = _embedded_bc_enforce(K_bcoo, presc_idx)

    U_star_dot = _solve_linear(
        K, fe_problem, fe_arrays, -Rp_dot, lss,
    )

    def xi_of_U_p(U_, params_, Up_, xp_, t_):
        _, _, xi_local = assemble_global(
            fe_problem, fe_arrays, params_,
            U_, Up_, t_,
            xi_prev_by_block=xp_,
        )
        return xi_local

    _, xi_star_dot = jax.jvp(
        xi_of_U_p,
        (U_star, params_by_block, U_prev, xi_prev_by_block, t),
        (U_star_dot, *p_dot),
    )

    primals_out = (U_star, xi_star)
    tangents_out = (U_star_dot, xi_star_dot)
    return primals_out, tangents_out
