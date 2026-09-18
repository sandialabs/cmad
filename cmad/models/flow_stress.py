"""The yield function of the plastic models' "yield surface" block.

The block's equation in the plastic branch is ``yield_function(phi, alpha,
alpha_dot, T, params) = 0`` with ``phi`` the effective stress, ``alpha`` the
hardening variable, ``alpha_dot`` its backward Euler rate over the step,
``T`` the temperature at the point (``None`` when the point carries no
temperature field), and ``params`` the ``flow stress`` subtree. The
function is in stress units, positive where the state lies outside the
yield surface, zero at the converged plastic state, and at most zero at
zero stress, which :func:`cmad.models.paths.yield_threshold` relies on.

A relation written as a flow stress gives ``phi - sigma_flow(alpha,
alpha_dot, T)``, the consistency condition with a rate-dependent yield
surface. A viscoplastic relation gives the plastic multiplier as a
function of the stress, and is written here solved for the stress; one
that cannot be inverted would be written as the rate mismatch in stress
units in the same slot. One relation per material, chosen once at
construction from the keys under ``flow stress``; the relations are real
valued, so a complex step model works with the rate-independent relation
only.

The rate-dependent relations are verified on the small strain models and
the hypoelastic model, Peric on be_bar as well. be_bar is not compatible
with Johnson-Cook.
"""
from collections.abc import Callable
from functools import partial
from typing import Any

import jax.numpy as jnp

from cmad.models.hardening import combined_hardening_fun, get_hardening_funs
from cmad.typing import JaxArray, Scalar

RATE_INDEPENDENT_KEYS = frozenset({"initial yield", "hardening"})
JOHNSON_COOK_KEYS = frozenset({"johnson_cook"})
PERIC_KEYS = frozenset({"peric"})

POWER_LAW_OFFSET = 1e-10


def offset_power(x: JaxArray | float, p: JaxArray | float) -> JaxArray:
    """``(x + POWER_LAW_OFFSET) ** p``, a power with a finite slope at zero."""
    return jnp.power(x + POWER_LAW_OFFSET, p)


def rate_independent_flow_stress(
        alpha: JaxArray, alpha_dot: JaxArray, T: Scalar | None,
        params: dict[str, Any],
        hardening_funs: dict[str, Callable[..., JaxArray]],
) -> JaxArray:
    """``Y`` plus the hardening entries; ``alpha_dot`` and ``T`` are unused."""
    return params["initial yield"]["Y"] + combined_hardening_fun(
        alpha, params["hardening"], hardening_funs)


def johnson_cook_flow_stress(
        alpha: JaxArray, alpha_dot: JaxArray, T: Scalar | None,
        params: dict[str, Any],
) -> JaxArray:
    """Johnson and Cook 1983: ``(A + B alpha^n) (1 + C ln(alpha_dot /
    reference rate)) (1 - T*^m)``.

    ``T*`` is the homologous temperature ``(T - T_ref) / (T_melt - T_ref)``,
    clipped to ``[0, 1]``; a point with no temperature field is taken to be
    at ``T_ref``, so its thermal term is one. Below the reference rate,
    where the logarithm would be negative, the rate term is set to one.
    Both powers are evaluated at ``x + POWER_LAW_OFFSET`` so their slope at
    zero is finite, which raises the initial flow stress by ``B alpha_0^n``.
    """
    p = params["johnson_cook"]
    strain_hardening = p["A"] + p["B"] * offset_power(alpha, p["n"])
    rate_dependence = 1.0 + p["C"] * jnp.log(
        jnp.maximum(alpha_dot / p["reference rate"], 1.0))
    if T is None:
        T_star: JaxArray | float = 0.0
    else:
        T_ref = p["reference temperature"]
        T_star = jnp.clip(
            (T - T_ref) / (p["melt temperature"] - T_ref), 0.0, 1.0)
    thermal_softening = jnp.maximum(1.0 - offset_power(T_star, p["m"]), 0.0)
    return strain_hardening * rate_dependence * thermal_softening


def peric_flow_stress(
        alpha: JaxArray, alpha_dot: JaxArray, T: Scalar | None,
        params: dict[str, Any],
        hardening_funs: dict[str, Callable[..., JaxArray]],
) -> JaxArray:
    """Peric 1993: ``alpha_dot = (1 / eta) [(phi / sigma_y)^(1 / epsilon) -
    1]`` solved for the stress, ``sigma_y (1 + eta alpha_dot)^epsilon``, with
    ``sigma_y`` the rate-independent flow stress of the same subtree.
    """
    p = params["peric"]
    sigma_y = rate_independent_flow_stress(
        alpha, alpha_dot, T, p, hardening_funs)
    return sigma_y * jnp.power(1.0 + p["eta"] * alpha_dot, p["epsilon"])


def consistency_yield_function(
        flow_stress_fun: Callable[..., JaxArray],
) -> Callable[..., JaxArray]:
    """The yield function ``phi - sigma_flow`` of a relation written as a
    flow stress.
    """
    def yield_function(
            phi: JaxArray, alpha: JaxArray, alpha_dot: JaxArray,
            T: Scalar | None, params: dict[str, Any],
    ) -> JaxArray:
        return phi - flow_stress_fun(alpha, alpha_dot, T, params)

    return yield_function


def make_yield_function(
        flow_params: dict[str, Any],
        hardening_funs: dict[str, Callable[..., JaxArray]] | None = None,
) -> Callable[..., JaxArray]:
    """The yield function of the one relation the ``flow stress`` subtree
    names.
    """
    if hardening_funs is None:
        hardening_funs = get_hardening_funs()
    keys = frozenset(flow_params)
    if keys == RATE_INDEPENDENT_KEYS:
        return consistency_yield_function(partial(
            rate_independent_flow_stress, hardening_funs=hardening_funs))
    if keys == JOHNSON_COOK_KEYS:
        return consistency_yield_function(johnson_cook_flow_stress)
    if keys == PERIC_KEYS:
        return consistency_yield_function(partial(
            peric_flow_stress, hardening_funs=hardening_funs))
    raise ValueError(
        f"flow stress: keys {sorted(keys)} name no relation; known: the "
        f"pair {sorted(RATE_INDEPENDENT_KEYS)}, {sorted(JOHNSON_COOK_KEYS)}, "
        f"or {sorted(PERIC_KEYS)}",
    )
