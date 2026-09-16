"""Rate dependence (viscoplastic) laws for the plastic multiplier.

Rate independent flow closes a return map with the consistency condition
``f = 0``. A viscoplastic law instead gives the plastic multiplier
explicitly in terms of the overstress, so that equation is replaced by
one relating the increment to ``f``, and the stress is free to sit
outside the yield surface by an amount set by the loading rate.

The laws here return that replacement residual rather than the multiplier
itself: written multiplied through by the viscosity, the zero viscosity
limit is the rate independent ``f = 0``, and the residual keeps the units
and scaling of the one it replaces.

Selected by the deck, which nests the law under its name the way
``hardening`` does::

    plastic:
      flow stress:
        rate_dependence:
          perzyna:
            eta: 100.0
"""
from collections.abc import Callable
from typing import Any

from cmad.typing import JaxArray, Scalar


def perzyna(
        delta_gamma: JaxArray, yield_fun: JaxArray, dt: Scalar,
        scale_factor: Scalar, flow_stress_params: dict[str, Any],
) -> JaxArray:
    """Perzyna overstress residual ``(eta / dt) * delta_gamma - f``.

    Perzyna takes the plastic multiplier rate proportional to the
    overstress, ``gamma_dot = f / eta``, which over a step of size ``dt``
    is ``delta_gamma = dt / eta * f``. That is returned here multiplied
    through by ``eta / dt``, so ``eta -> 0`` degenerates to the rate
    independent residual ``f = 0`` instead of dividing by zero.

    ``eta`` is a viscosity in stress * time: the overstress the material
    carries per unit equivalent plastic strain rate.

    ``yield_fun`` is the model's yield function already divided by
    ``scale_factor`` (models scale their residuals by twice the shear
    modulus), so the viscous term carries the same division and the two
    are comparable.

    Takes the whole flow stress subtree and reaches into its own, the way
    the models read ``flow stress.initial yield.Y``, so callers hold only
    the function and not the name it is registered under.

    No Macaulay bracket on ``f``: the models select this residual only on
    the branch where the elastic trial state lies outside the yield
    surface, which is exactly where the Perzyna solution has
    ``delta_gamma > 0``.
    """
    eta = flow_stress_params["rate_dependence"]["perzyna"]["eta"]

    return eta / dt * delta_gamma / scale_factor - yield_fun


def get_rate_dependence_funs() -> dict[str, Callable[..., JaxArray]]:
    return {"perzyna": perzyna}


def resolve_rate_dependence(
        flow_stress_params: dict[str, Any],
        rate_dependence_funs: dict[str, Callable[..., JaxArray]] | None = None,
) -> tuple[str, Callable[..., JaxArray]] | None:
    """The ``(name, fun)`` of the flow stress subtree's rate dependence law,
    or ``None`` when there is none and the flow is rate independent.

    Resolution happens once, at model construction: the returned function
    is a static argument of the residual, so which law is in play is fixed
    at trace time and only its parameter values are traced.
    """
    section = flow_stress_params.get("rate_dependence")
    if not section:
        return None

    if rate_dependence_funs is None:
        rate_dependence_funs = get_rate_dependence_funs()

    if len(section) != 1:
        raise ValueError(
            "plastic.flow stress.rate_dependence takes exactly one law, "
            f"got {sorted(section)}",
        )

    name = next(iter(section))
    if name not in rate_dependence_funs:
        raise ValueError(
            f"unknown rate dependence law '{name}'; known: "
            f"{sorted(rate_dependence_funs)}",
        )

    return name, rate_dependence_funs[name]
