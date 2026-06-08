"""Tests for the Armijo line search (``cmad/util/line_search.py``)."""
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit

from cmad.util.line_search import (
    DEFAULT_LINE_SEARCH_SETTINGS,
    cubic_min,
    line_search,
    quad_min,
)


def test_cubic_min_recovers_cubic_minimizer():
    """cubic_min returns the interior minimizer of the interpolated cubic.

    ``p(t) = t^3 + 0.9 t^2 - 1.2 t`` has ``p'(t) = 3 t^2 + 1.8 t - 1.2``,
    whose positive root (the local minimum) is ``t = 0.4``. The two-point
    Hermite interpolant of a cubic is that same cubic, so the value and
    slope at the base point and at ``a > 0.4`` recover 0.4.
    """
    def p(t):
        return t**3 + 0.9 * t**2 - 1.2 * t

    def dp(t):
        return 3.0 * t**2 + 1.8 * t - 1.2

    a = 1.0
    alpha = cubic_min(p(0.0), dp(0.0), a, p(a), dp(a))
    assert float(alpha) == pytest.approx(0.4, abs=1e-12)


def test_quad_min_recovers_quadratic_minimizer():
    """quad_min returns the interior minimizer of the interpolated quadratic.

    ``q(t) = (t - 0.4)^2`` has its minimum at ``t = 0.4``. The value and
    slope at the base point plus the value at ``a`` recover 0.4, since the
    quadratic through those points is q itself.
    """
    def q(t):
        return (t - 0.4) ** 2

    def dq(t):
        return 2.0 * (t - 0.4)

    a = 1.0
    alpha = quad_min(q(0.0), dq(0.0), a, q(a))
    assert float(alpha) == pytest.approx(0.4, abs=1e-12)


def test_quadratic_path_backtracks():
    """eval_fn returning slope=None selects the quadratic step model.

    An overshooting full step is still damped to alpha < 1 with sufficient
    decrease, using only the merit values (no trial slope).
    """
    def phi(alpha):
        r = 1.0 - 3.0 * alpha
        return 0.5 * r * r

    def eval_fn(alpha):
        r = 1.0 - 3.0 * alpha
        return phi(alpha), None, r

    phi_0, dphi_0 = phi(0.0), -3.0
    c1 = DEFAULT_LINE_SEARCH_SETTINGS["sufficient decrease"]
    alpha, aux = line_search(
        eval_fn, phi_0, dphi_0, DEFAULT_LINE_SEARCH_SETTINGS, 1.0,
    )
    alpha = float(alpha)

    assert 0.0 < alpha < 1.0
    assert phi(alpha) < phi_0
    assert phi(alpha) <= phi_0 + c1 * alpha * dphi_0  # Armijo
    assert float(aux) == pytest.approx(1.0 - 3.0 * alpha)


def test_full_step_accepted():
    """A clean Newton step (the full step reduces the merit) is accepted at 1."""
    # r(a) = 1 - a predicts the root at the full step; phi(a) = 0.5 (1-a)^2.
    def eval_fn(alpha):
        r = 1.0 - alpha
        return 0.5 * r * r, (1.0 - alpha) * (-1.0), r

    alpha, aux = line_search(eval_fn, 0.5, -1.0, DEFAULT_LINE_SEARCH_SETTINGS, 1.0)
    assert float(alpha) == pytest.approx(1.0)
    assert float(aux) == pytest.approx(0.0)  # residual at the accepted step


def test_backtracks_on_overshoot():
    """An overshooting full step is damped to alpha < 1 with sufficient decrease."""
    # r(a) = 1 - 3a (root at 1/3); the full step triples the residual.
    def phi(alpha):
        r = 1.0 - 3.0 * alpha
        return 0.5 * r * r

    def eval_fn(alpha):
        r = 1.0 - 3.0 * alpha
        return phi(alpha), r * (-3.0), r

    phi_0, dphi_0 = phi(0.0), -3.0
    c1 = DEFAULT_LINE_SEARCH_SETTINGS["sufficient decrease"]
    alpha, aux = line_search(
        eval_fn, phi_0, dphi_0, DEFAULT_LINE_SEARCH_SETTINGS, 1.0,
    )
    alpha = float(alpha)

    assert 0.0 < alpha < 1.0
    assert phi(alpha) < phi_0
    assert phi(alpha) <= phi_0 + c1 * alpha * dphi_0  # Armijo
    assert float(aux) == pytest.approx(1.0 - 3.0 * alpha)  # residual at accepted alpha


def test_returned_aux_is_a_pytree():
    """aux can be an arbitrary pytree; the accepted trial's value comes back."""
    def eval_fn(alpha):
        r = 1.0 - alpha
        return 0.5 * r * r, (1.0 - alpha) * (-1.0), {"r": r, "twice": 2.0 * r}

    init_aux = {"r": 1.0, "twice": 2.0}
    alpha, aux = line_search(
        eval_fn, 0.5, -1.0, DEFAULT_LINE_SEARCH_SETTINGS, init_aux,
    )
    assert float(alpha) == pytest.approx(1.0)
    assert float(aux["r"]) == pytest.approx(0.0)
    assert float(aux["twice"]) == pytest.approx(0.0)


def test_disabled_returns_full_step():
    """max evals = 0 takes the full step and returns init_aux unprobed."""
    def eval_fn(alpha):
        r = 1.0 - 3.0 * alpha  # would be rejected if probed, but it is not
        return 0.5 * r * r, r * (-3.0), r

    settings = {**DEFAULT_LINE_SEARCH_SETTINGS, "max evals": 0}
    alpha, aux = line_search(eval_fn, 0.5, -3.0, settings, 7.0)
    assert float(alpha) == 1.0
    assert float(aux) == 7.0


def test_traces_under_jit():
    """The search runs under jit and matches the eager result.

    It has to survive the jitted Newton solves, so the branch-free form is
    exercised through a jit boundary here.
    """
    def eval_fn(alpha):
        r = 1.0 - 3.0 * alpha
        return 0.5 * r * r, r * (-3.0), r

    def run(phi_0, dphi_0):
        return line_search(
            eval_fn, phi_0, dphi_0, DEFAULT_LINE_SEARCH_SETTINGS, 1.0,
        )

    eager_alpha, eager_aux = run(0.5, -3.0)
    jit_alpha, jit_aux = jit(run)(0.5, -3.0)
    assert float(jit_alpha) == pytest.approx(float(eager_alpha))
    assert float(jit_aux) == pytest.approx(float(eager_aux))


def test_nonfinite_probe_contracts():
    """A non-finite trial merit halves the step rather than accepting it.

    The full step returns NaN (a diverged probe); smaller steps are finite
    and meet sufficient decrease, so the search moves into the finite region.
    """
    def eval_fn(alpha):
        phi = jnp.where(alpha > 0.75, jnp.nan, 0.5 * (1.0 - alpha) ** 2)
        r = 1.0 - alpha
        return phi, (1.0 - alpha) * (-1.0), r

    alpha, aux = line_search(eval_fn, 0.5, -1.0, DEFAULT_LINE_SEARCH_SETTINGS, 1.0)
    alpha = float(alpha)
    assert 0.0 < alpha <= 0.75
    assert np.isfinite(0.5 * (1.0 - alpha) ** 2)
    assert float(aux) == pytest.approx(1.0 - alpha)
