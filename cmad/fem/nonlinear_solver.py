"""Global Newton driver for the FE forward problem."""
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from numpy.typing import NDArray

from cmad.fem.assembly import assemble_element_tangent, assemble_global
from cmad.fem.fe_problem import FEProblem
from cmad.fem.kernel_arrays import FEKernelArrays
from cmad.fem.sharding import place_element_leaves
from cmad.fem.sparse_solve import (
    AssembledOperator,
    ElementOperator,
    TangentOperator,
    _block_gmres,
    _embedded_bc_enforce,
    _embedded_residual,
    _jacobi_cg,
    _jacobi_gmres,
    _near_null_by_field,
    scipy_amg_cg,
    scipy_block_gmres,
    scipy_lu,
)
from cmad.models.global_fields import StepTime
from cmad.typing import JaxArray, Params
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
    "operator": "assembled",
}
_OPERATORS = ("assembled", "element")


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


def _operator_kind(linear_solver_settings: dict[str, Any]) -> str:
    """``settings['operator']``: ``'assembled'`` (the default) or
    ``'element'``."""
    operator = linear_solver_settings.get("operator", "assembled")
    if operator not in _OPERATORS:
        raise ValueError(
            f"unknown operator {operator!r}; expected one of {_OPERATORS}",
        )
    return str(operator)


def _tangent_operator(
        K: Any, fe_problem: FEProblem, fe_arrays: FEKernelArrays,
        operator: str,
) -> TangentOperator:
    """The operator the line search slope and the jax native solvers apply:
    :class:`AssembledOperator` on the enforced COO data when ``operator``
    is ``'assembled'``, :class:`ElementOperator` on the per element blocks
    when it is ``'element'``."""
    if operator == "assembled":
        return AssembledOperator(
            K, fe_arrays.embedded_sparsity, fe_arrays.block_sparsity,
            fe_problem.device_mesh,
        )
    return ElementOperator(
        K, fe_arrays.r_scatter_eq_by_block, fe_problem.field_idx_per_block,
        fe_problem.dof_map.block_offsets, fe_arrays.prescribed_indices,
        fe_problem.dof_map.num_total_dofs, fe_problem.device_mesh,
    )


def _assemble_tangent_and_residual(
        fe_problem: FEProblem,
        fe_arrays: FEKernelArrays,
        params_by_block: Mapping[str, Params],
        U: JaxArray,
        U_prev: JaxArray,
        step_time: StepTime,
        xi_prev_by_block: Mapping[str, JaxArray],
        presc_vals: JaxArray,
        operator: str,
) -> tuple[JaxArray, Any, dict[str, JaxArray]]:
    """Assemble at ``U`` and return ``(r, K, xi)``: the embedded residual,
    the tangent, and the solved state. ``operator`` picks the tangent's
    representation: ``'assembled'`` gives the enforced COO data of
    :func:`_embedded_bc_enforce`, ``'element'`` the per element blocks of
    :func:`assemble_element_tangent`; :func:`_tangent_operator` builds the
    operator from either."""
    presc_idx = fe_arrays.prescribed_indices
    if operator == "assembled":
        K_bcoo, R, xi = assemble_global(
            fe_problem, fe_arrays, params_by_block, U, U_prev, step_time,
            xi_prev_by_block=xi_prev_by_block,
        )
        K, K_ii_presc = _embedded_bc_enforce(K_bcoo, presc_idx)
        r = _embedded_residual(
            R, lambda v: K_bcoo @ v, U, presc_idx, presc_vals, K_ii_presc,
        )
        return r, K, xi
    K_elem, R, xi = assemble_element_tangent(
        fe_problem, fe_arrays, params_by_block, U, U_prev, step_time,
        xi_prev_by_block=xi_prev_by_block,
    )
    op = ElementOperator(
        K_elem, fe_arrays.r_scatter_eq_by_block, fe_problem.field_idx_per_block,
        fe_problem.dof_map.block_offsets, presc_idx,
        fe_problem.dof_map.num_total_dofs,
    )
    r = _embedded_residual(
        R, op.raw_matvec, U, presc_idx, presc_vals, op.diagonal()[presc_idx],
    )
    return r, K_elem, xi


def _solve_linear(
        K: Any,
        fe_problem: FEProblem,
        fe_arrays: FEKernelArrays,
        rhs: JaxArray,
        linear_solver_settings: dict[str, Any],
) -> JaxArray:
    """Dispatch on ``settings['type']`` to direct / CG / GMRES, with the
    iterative arms picking a preconditioner from
    ``settings['preconditioner']``: Jacobi or pyamg for CG, Jacobi or a
    block preconditioner for GMRES (:func:`_block_gmres` with a Jacobi
    or Chebyshev inner solve, :func:`scipy_block_gmres` with an AMG inner
    solve). The jax native solvers apply ``K`` through
    :func:`_tangent_operator`; the direct, pyamg and AMG solvers need the
    assembled representation.

    :attr:`FEProblem.near_null_space` is auto-merged into pyamg
    ``kwargs`` as ``B`` when present and the caller hasn't already set
    it.
    """
    sparsity = fe_arrays.embedded_sparsity
    kind = linear_solver_settings["type"]
    operator = _operator_kind(linear_solver_settings)

    def require_assembled(solver: str) -> None:
        if operator != "assembled":
            raise ValueError(
                f"linear solver {solver} needs operator 'assembled'; the "
                f"element operator serves cg + jacobi, gmres + jacobi and "
                f"gmres + block with a jacobi or chebyshev inner solve"
            )

    def tangent_operator() -> TangentOperator:
        return _tangent_operator(K, fe_problem, fe_arrays, operator)

    if kind == "direct":
        require_assembled("'direct'")
        return scipy_lu(K, sparsity, rhs, fe_problem.fill_permutation)

    precon_spec = linear_solver_settings.get(
        "preconditioner", {"type": "jacobi"},
    )
    precon = precon_spec["type"]

    if kind == "cg":
        if precon == "jacobi":
            return _jacobi_cg(
                tangent_operator(), rhs,
                rtol=linear_solver_settings["rtol"],
                max_iters=linear_solver_settings["max iters"],
            )
        if precon == "pyamg":
            require_assembled("'cg' with the pyamg preconditioner")
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
            return _jacobi_gmres(
                tangent_operator(), rhs,
                rtol=linear_solver_settings["rtol"],
                restart=linear_solver_settings["restart"],
                max_iters=linear_solver_settings["max iters"],
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
                return _block_gmres(
                    tangent_operator(), rhs,
                    coupling=coupling, diagonal_block=diagonal_block,
                    inner=inner, degree=precon_spec.get("degree"),
                    rtol=linear_solver_settings["rtol"],
                    max_iters=linear_solver_settings["max iters"],
                    restart=linear_solver_settings["restart"],
                )
            if inner == "amg":
                require_assembled("'gmres' with the block amg preconditioner")
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


@dataclass(frozen=True)
class NewtonStatus:
    """How a global Newton step ended.

    ``converged`` is :func:`newton_converged` applied to the two norms;
    the norms come along so a caller can report how far off a failed step
    was, in both absolute and relative terms.
    """
    converged: JaxArray
    residual_norm: JaxArray
    residual_norm_0: JaxArray

    def relative_norm(self) -> JaxArray:
        return self.residual_norm / self.residual_norm_0


def newton_converged(
        residual_norm: JaxArray,
        residual_norm_0: JaxArray,
        abs_tol: float,
        rel_tol: float,
) -> JaxArray:
    """Whether a global Newton residual meets either tolerance.

    The single place the criterion is written. The forward loop tests it
    to decide when to stop, and a caller tests it on the returned norms to
    tell a converged step from one that hit the iteration limit, so the two
    cannot drift apart.

    ``residual_norm_0`` is the norm the step started from, floored at
    ``abs_tol`` by the solver, so on a step that begins already converged
    the relative test is measured against that floor rather than the true
    initial norm.
    """
    return jnp.logical_or(
        residual_norm < abs_tol, residual_norm < rel_tol * residual_norm_0,
    )


def _fe_newton_primal(
        fe_problem: FEProblem,
        fe_arrays: FEKernelArrays,
        params_by_block: Mapping[str, Params],
        U_prev: JaxArray,
        xi_prev_by_block: Mapping[str, JaxArray],
        step_time: StepTime,
        nonlinear_solver_settings: dict[str, Any],
        linear_solver_settings: dict[str, Any],
) -> tuple[JaxArray, dict[str, JaxArray], JaxArray, JaxArray]:
    """Forward Newton iteration: ``lax.while_loop`` + linear-solver dispatch.

    Each step assembles ``(K_bcoo, R)`` via :func:`assemble_global`,
    builds the embedded-BC tangent ``K`` via
    :func:`_embedded_bc_enforce` and the matching residual ``r`` via
    :func:`_embedded_residual`, and solves ``K · dU = -r`` via the
    linear solver named in ``linear_solver_settings['type']`` (one of
    ``direct``, ``cg``, ``gmres``). ``cond`` checks the residual norm
    against the absolute and relative tolerances.

    Returns ``(U_star, xi_star, residual_norm, residual_norm_0)``: the
    converged displacement, the solved state at it, the residual norm the
    loop exited on, and the norm it started from. The loop stops once the
    residual meets the absolute tolerance ``abs tol`` or falls to
    ``rel tol`` of where it started, and otherwise when it hits the
    iteration limit. Both norms come back because the exit alone does not say
    which of those happened; :func:`newton_converged` is the shared test.
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
    presc_vals = jnp.asarray(
        dof_map.evaluate_prescribed_values(fe_arrays.dbc_arrays, step_time.t),
    )
    operator = _operator_kind(linear_solver_settings)

    U_init = U_prev

    def _assemble_enforced(U):
        return _assemble_tangent_and_residual(
            fe_problem, fe_arrays, params_by_block, U, U_prev, step_time,
            xi_prev_by_block, presc_vals, operator,
        )

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
        converged = newton_converged(
            jnp.linalg.norm(r), R0, abs_tol, rel_tol,
        )
        return (i < max_iters) & jnp.logical_not(converged)

    def body(state):
        i, r, K, U, xi = state
        dU = _solve_linear(
            K, fe_problem, fe_arrays, -r, linear_solver_settings,
        )
        if ls_max_evals > 0:
            r_norm_sq = r @ r

            def eval_fn(alpha):
                r_trial, K_trial, xi_trial = _assemble_enforced(U + alpha * dU)
                slope = r_trial @ _tangent_operator(
                    K_trial, fe_problem, fe_arrays, operator,
                ).matvec(dU)
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

    _, r_star, _, U_star, xi_star = lax.while_loop(
        cond, body, (0, r_init, K_init, U_init, xi_init),
    )
    return U_star, xi_star, jnp.linalg.norm(r_star), R0


def fe_newton_solve(
        fe_problem: FEProblem,
        params_by_block: Mapping[str, Params],
        U_prev: NDArray[np.floating] | JaxArray,
        xi_prev_by_block: Mapping[str, NDArray[np.floating] | JaxArray]
        | None = None,
        t: float = 0.0,
        t_prev: float | None = None,
        nonlinear_solver_settings: dict[str, Any] | None = None,
        linear_solver_settings: dict[str, Any] | None = None,
        return_status: bool = False,
) -> tuple[JaxArray, dict[str, JaxArray]] | tuple[
        JaxArray, dict[str, JaxArray], "NewtonStatus"]:
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

    Returns ``(U_star, xi_star)``, or with ``return_status`` a third
    entry, a :class:`NewtonStatus` saying whether the step met a
    tolerance or hit the iteration limit. Without it a caller cannot tell
    those apart, and an abandoned step looks like a solved one. Outputs
    are JAX arrays.
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
        place_element_leaves(
            {k: jnp.asarray(v) for k, v in xi_prev_by_block.items()},
            fe_problem.device_mesh,
        )
        if xi_prev_by_block is not None else {}
    )
    step_time = StepTime(t, t if t_prev is None else t_prev)
    U_star, xi_star, residual_norm, residual_norm_0 = _fe_newton_solve_ad(
        fe_problem, fe_problem.kernel_arrays, params_by_block,
        U_prev_jax, xi_prev_jax, step_time, _freeze(nls), _freeze(lss),
    )
    if not return_status:
        return U_star, xi_star
    return U_star, xi_star, NewtonStatus(
        converged=newton_converged(
            residual_norm, residual_norm_0, nls["abs tol"], nls["rel tol"],
        ),
        residual_norm=residual_norm,
        residual_norm_0=residual_norm_0,
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 6, 7))
def _fe_newton_solve_ad(
        fe_problem: FEProblem,
        fe_arrays: FEKernelArrays,
        params_by_block: Mapping[str, Params],
        U_prev: JaxArray,
        xi_prev_by_block: dict[str, JaxArray],
        step_time: StepTime,
        nonlinear_solver_settings_frozen: tuple[tuple[str, Any], ...],
        linear_solver_settings_frozen: tuple[tuple[str, Any], ...],
) -> tuple[JaxArray, dict[str, JaxArray], JaxArray, JaxArray]:
    """AD-decorated inner driver. JaxArray inputs only.

    Splitting the public ``fe_newton_solve`` from this inner form
    keeps the boundary ``np.ndarray → jnp.ndarray`` conversion
    outside the ``custom_jvp``-tracked function body, so the diff
    args are uniformly typed for the JVP rule. ``step_time`` stays in
    the diff set: when the driver runs inside a :func:`jax.lax.scan`
    over the time schedule, its ``t`` / ``t_prev`` are tracers (slices
    of the scan's traced input), and ``nondiff_argnums`` requires
    hashable Python values. The JVP rule accepts a ``step_time``
    tangent that no current consumer populates. The two
    settings dicts are passed as :func:`_freeze`'d tuples so they
    are hashable for ``custom_jvp``'s nondiff-arg cache.
    """
    nls = _thaw(nonlinear_solver_settings_frozen)
    lss = _thaw(linear_solver_settings_frozen)
    return _fe_newton_primal(
        fe_problem, fe_arrays, params_by_block, U_prev, xi_prev_by_block,
        step_time, nls, lss,
    )


@_fe_newton_solve_ad.defjvp
def _fe_newton_solve_ad_jvp(
        fe_problem: FEProblem,
        nonlinear_solver_settings_frozen: tuple[tuple[str, Any], ...],
        linear_solver_settings_frozen: tuple[tuple[str, Any], ...],
        primals, tangents,
):
    """IFT linear-sensitivity JVP for :func:`_fe_newton_solve_ad`.

    For ``r(U, p) = 0`` with ``p = (params, U_prev, xi_prev, step_time)``,
    ``U_star_dot = -K^{-1} · (∂r/∂p · p_dot)`` with ``K = ∂r/∂U`` at
    ``U_star``. ``∂r/∂p · p_dot`` is computed by
    ``jax.jvp(r, p, p_dot)`` at fixed ``U_star``; ``K`` is the
    embedded tangent at ``U_star`` in the representation the settings
    select (:func:`_assemble_tangent_and_residual`);
    the linear solve goes through :func:`_solve_linear` so the ``K``
    cotangent flows automatically when JAX auto-transposes the rule.
    ``xi_star_dot`` follows from chain rule: the assembly's xi
    output is differentiated jointly w.r.t. ``U_star`` (with tangent
    ``U_star_dot``) and w.r.t. ``p`` (with tangent ``p_dot``).
    The ``step_time`` tangent is present only as ceremony — no current
    consumer populates a non-zero one — but ``step_time`` itself stays
    in the primals tuple because under :func:`jax.lax.scan` it is
    traced (its ``t`` is a scanned input slice and its ``t_prev`` a
    carried value) and cannot ride in ``nondiff_argnums``.
    """
    fe_arrays, params_by_block, U_prev, xi_prev_by_block, step_time = primals
    p_dot = tangents[1:]  # tangents[0]: fe_arrays tangent, unused

    lss = _thaw(linear_solver_settings_frozen)

    U_star, xi_star, residual_norm, residual_norm_0 = _fe_newton_solve_ad(
        fe_problem, fe_arrays, params_by_block, U_prev, xi_prev_by_block,
        step_time, nonlinear_solver_settings_frozen,
        linear_solver_settings_frozen,
    )

    operator = _operator_kind(lss)

    def prescribed_values(step_time_):
        return jnp.asarray(
            fe_problem.dof_map.evaluate_prescribed_values(
                fe_arrays.dbc_arrays, step_time_.t,
            ),
        )

    # Trailing-underscore params (params_ <-> params_by_block, Up_ <->
    # U_prev, xp_ <-> xi_prev_by_block, step_time_ <-> step_time) are this
    # helper's explicit jvp-differentiated inputs; U_star is captured,
    # held fixed by the IFT.
    def r_of_p(params_, Up_, xp_, step_time_):
        r, _, _ = _assemble_tangent_and_residual(
            fe_problem, fe_arrays, params_, U_star, Up_, step_time_, xp_,
            prescribed_values(step_time_), operator,
        )
        return r

    _, Rp_dot = jax.jvp(
        r_of_p,
        (params_by_block, U_prev, xi_prev_by_block, step_time),
        p_dot,
    )

    _, K, _ = _assemble_tangent_and_residual(
        fe_problem, fe_arrays, params_by_block, U_star, U_prev, step_time,
        xi_prev_by_block, prescribed_values(step_time), operator,
    )

    U_star_dot = _solve_linear(
        K, fe_problem, fe_arrays, -Rp_dot, lss,
    )

    def xi_of_U_p(U_, params_, Up_, xp_, step_time_):
        _, _, xi_local = assemble_global(
            fe_problem, fe_arrays, params_,
            U_, Up_, step_time_,
            xi_prev_by_block=xp_,
        )
        return xi_local

    _, xi_star_dot = jax.jvp(
        xi_of_U_p,
        (U_star, params_by_block, U_prev, xi_prev_by_block, step_time),
        (U_star_dot, *p_dot),
    )

    # The residual norm reports how the forward loop exited, so it is a
    # diagnostic rather than a solution quantity and carries no tangent.
    primals_out = (U_star, xi_star, residual_norm, residual_norm_0)
    tangents_out = (
        U_star_dot, xi_star_dot,
        jnp.zeros_like(residual_norm), jnp.zeros_like(residual_norm_0),
    )
    return primals_out, tangents_out
