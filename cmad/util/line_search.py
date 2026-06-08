"""Backtracking Armijo line search with two-point Hermite cubic interpolation.

Step-length control shared by CMAD's two damped Newton solves: the FE
global Newton (:mod:`cmad.fem.nonlinear_solver`) and the local
constitutive Newton (:mod:`cmad.models.nonlinear_solver`). For a step
``dx`` from a base iterate, the merit is
``phi(alpha) = 1/2 ||r(x + alpha*dx)||^2``; the search accepts the largest
trial step meeting the Armijo sufficient-decrease condition
``phi(alpha) <= phi(0) + c1*alpha*phi'(0)``, and takes each rejected step
from the minimizer of the cubic that matches ``phi`` and ``phi'`` at the
base point and at the current trial (a two-point Hermite cubic), limited
to a fraction of the current step.

The caller's ``eval_fn(alpha)`` returns ``(phi, slope, aux)`` at a trial
step. How the slope ``phi'(alpha)`` is formed is the caller's choice: the
FE global Newton reads it from a sparse matvec with the tangent it
assembles for the merit (``r . (K . dx)``); the local constitutive
Newton, whose residual carries no boundary coupling, takes it from a
forward-mode ``jax.jvp`` of the residual. ``aux`` is an extra value the
caller wants returned at the accepted step: the FE solve returns the
``(r, K, xi)`` it assembled there so the next Newton iteration reuses it
instead of re-assembling. The search returns ``(alpha, aux)``.

Branch-free throughout (``lax.while_loop`` plus ``jnp.where``) so it runs
inside the traced Newton solves. It sits within those solves' own
``custom_jvp`` implicit-function-theorem wrappers, so the search iterates
are not differentiated through; only the converged step feeds AD.
"""
from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
from jax import debug, lax, tree_util

from cmad.typing import PyTree, Scalar

DEFAULT_LINE_SEARCH_SETTINGS: dict[str, Any] = {
    "max evals": 4,
    "sufficient decrease": 1.0e-4,
    "min backtrack factor": 0.5,
    "max backtrack factor": 0.9,
    "print": False,
}


def cubic_min(
        phi_0: Scalar,
        dphi_0: Scalar,
        a: Scalar,
        phi: Scalar,
        slope_a: Scalar,
) -> Scalar:
    """Minimizer of the cubic matching ``(phi_0, dphi_0)`` at ``alpha = 0``
    and ``(phi, slope_a)`` at ``alpha = a``.

    Standard two-point Hermite cubic interpolation. Falls back to
    ``0.5 * a`` when there is no real interior minimizer (negative
    radicand) or the denominator is degenerate; the caller limits the
    result to a fraction of ``a``.
    """
    d1 = dphi_0 + slope_a - 3.0 * (phi_0 - phi) / (0.0 - a)
    radicand = d1 * d1 - dphi_0 * slope_a
    d2 = jnp.sqrt(jnp.maximum(radicand, 0.0))
    denom = slope_a - dphi_0 + 2.0 * d2
    safe_denom = jnp.where(denom == 0.0, 1.0, denom)
    candidate = a - a * (slope_a + d2 - d1) / safe_denom
    fallback = jnp.logical_or(radicand < 0.0, denom == 0.0)
    return jnp.where(fallback, 0.5 * a, candidate)


def _select(pred: Scalar, on_true: PyTree, on_false: PyTree) -> PyTree:
    """``jnp.where(pred, on_true, on_false)`` applied across a pytree."""
    return tree_util.tree_map(
        lambda x, y: jnp.where(pred, x, y), on_true, on_false,
    )


def line_search(
        eval_fn: Callable[[Scalar], tuple[Scalar, Scalar, PyTree]],
        phi_0: Scalar,
        dphi_0: Scalar,
        settings: Mapping[str, Any],
        init_aux: PyTree,
) -> tuple[Scalar, PyTree]:
    """Backtracking Armijo line search; returns ``(alpha, aux)``.

    ``eval_fn(alpha)`` advances to the trial step and returns
    ``(phi, slope, aux)`` with ``phi = 1/2 ||r(alpha)||^2``,
    ``slope = phi'(alpha)``, and ``aux`` an extra value the caller wants
    back (the assembly at the trial). ``phi_0`` / ``dphi_0`` are the merit
    and its slope at the base point (``dphi_0 = -||r_0||^2 < 0`` for a
    Newton step). ``settings`` carries the keys of
    :data:`DEFAULT_LINE_SEARCH_SETTINGS`; ``max evals = 0`` returns the
    full step without probing. ``init_aux`` is the starting value for the
    carried ``aux`` (the base-point assembly works) and is what comes back
    when no trial is evaluated.

    Starts at the full step ``alpha = 1`` and accepts the first trial
    meeting sufficient decrease; otherwise contracts via
    :func:`cubic_min` limited to ``[min_factor*alpha, max_factor*alpha]``.
    A non-finite trial merit (a diverged probe) halves the step instead.
    If no trial is accepted within ``max evals``, returns the lowest-merit
    step tried. The returned ``aux`` is the value ``eval_fn`` produced at
    the returned step, so the caller reuses it rather than recomputing.
    """
    max_evals = settings["max evals"]
    c1 = settings["sufficient decrease"]
    backtrack_min = settings["min backtrack factor"]
    backtrack_max = settings["max backtrack factor"]
    print_ls = settings["print"]

    armijo_slope = c1 * dphi_0

    def cond(carry):
        n, _alpha, accepted, _aux, _b_alpha, _b_phi, _b_aux = carry
        return jnp.logical_and(n < max_evals, jnp.logical_not(accepted))

    def body(carry):
        n, alpha, _accepted, _aux, best_alpha, best_phi, best_aux = carry
        phi, slope, aux = eval_fn(alpha)
        finite = jnp.isfinite(phi)

        improved = jnp.logical_and(finite, phi < best_phi)
        best_alpha = jnp.where(improved, alpha, best_alpha)
        best_phi = jnp.where(improved, phi, best_phi)
        best_aux = _select(improved, aux, best_aux)

        accepted = jnp.logical_and(
            finite, phi <= phi_0 + alpha * armijo_slope,
        )

        alpha_cubic = jnp.clip(
            cubic_min(phi_0, dphi_0, alpha, phi, slope),
            backtrack_min * alpha,
            backtrack_max * alpha,
        )
        # accepted: hold alpha (the loop exits); diverged: halve; else cubic.
        alpha_next = jnp.where(
            accepted, alpha,
            jnp.where(finite, alpha_cubic, 0.5 * alpha),
        )
        return (
            n + 1, alpha_next, accepted, aux,
            best_alpha, best_phi, best_aux,
        )

    init = (
        jnp.asarray(0),
        jnp.asarray(1.0),
        jnp.asarray(False),
        init_aux,
        jnp.asarray(1.0),
        jnp.asarray(jnp.inf),
        init_aux,
    )
    n, alpha, accepted, aux, best_alpha, _b_phi, best_aux = lax.while_loop(
        cond, body, init,
    )
    result_alpha = jnp.where(accepted, alpha, best_alpha)
    result_aux = _select(accepted, aux, best_aux)

    if print_ls:
        debug.print(
            " > line search: alpha = {a:.3e} ({n} evals)",
            a=result_alpha, n=n,
        )
    return result_alpha, result_aux
