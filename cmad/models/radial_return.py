"""Scalar solve for the plastic multiplier increment using radial return."""
from collections.abc import Callable

import jax.numpy as jnp
from jax import grad
from jax.lax import while_loop

from cmad.models.deformation_types import DefType
from cmad.typing import JaxArray, Scalar

_MAX_ITERS = 10
_INITIAL_GUESSES = ("elastic predictor", "radial return")
_RADIAL_RETURN_DEF_TYPES = (DefType.FULL_3D, DefType.PLANE_STRAIN)

# iteration count, Δγ, g(Δγ), and the same pair at the previous iteration
_Carry = tuple[JaxArray, JaxArray, JaxArray, JaxArray, JaxArray]


def resolve_initial_guess(
        initial_guess: str | None, def_type: int, model_name: str,
) -> str:
    """The initial guess a model uses. When none is given: a radial return
    for FULL_3D or PLANE_STRAIN, the elastic predictor otherwise."""
    if initial_guess is None:
        if def_type in _RADIAL_RETURN_DEF_TYPES:
            return "radial return"
        return "elastic predictor"
    if initial_guess not in _INITIAL_GUESSES:
        raise ValueError(
            f"{model_name}: unknown initial guess '{initial_guess}'; "
            f"known: {list(_INITIAL_GUESSES)}")
    if initial_guess == "radial return" \
            and def_type not in _RADIAL_RETURN_DEF_TYPES:
        raise ValueError(
            f"{model_name}: the radial return initial guess needs full_3d "
            "or plane_strain; in plane stress and uniaxial stress the trial "
            "state depends on the stretch unknowns")
    return initial_guess


def plastic_multiplier_increment(
        g: Callable[[JaxArray], JaxArray], threshold: Scalar,
) -> JaxArray:
    """Root ``Δγ`` of the scalar yield equation ``g(Δγ) = 0`` by a Newton
    iteration from zero, run while ``g`` is above ``threshold`` and still
    decreasing. Zero when the iteration never lowers ``g(0)``."""
    g_prime = grad(g)

    def cond_fun(carry: _Carry) -> JaxArray:
        k, _, g_x, _, g_prev = carry
        still_decreasing = (k == 0) | (g_x < g_prev)
        return (k < _MAX_ITERS) & (g_x > threshold) & still_decreasing

    def body_fun(carry: _Carry) -> _Carry:
        k, x, g_x, _, _ = carry
        x_new = x - g_x / g_prime(x)
        return k + 1, x_new, g(x_new), x, g_x

    zero = jnp.zeros(())
    g_trial = g(zero)
    _, x, g_x, x_prev, g_prev = while_loop(
        cond_fun, body_fun,
        (jnp.zeros((), dtype=jnp.int32), zero, g_trial, zero, g_trial))

    return jnp.where(g_x < g_prev, x, x_prev)
