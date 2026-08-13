from typing import Any

import jax.numpy as jnp

from cmad.models.elastic_stress import two_mu_scale_factor
from cmad.typing import JaxArray, Scalar


def yield_threshold(yield_tol: float, params: dict[str, Any]) -> Scalar:
    """Threshold on the yield function for :func:`cond_residual`.

    ``yield_tol`` is relative to the initial yield stress, so one number
    means the same thing across materials and unit systems. The models
    divide the yield function by ``two_mu_scale_factor``, so the threshold
    carries that factor too.
    """
    Y = params["plastic"]["flow stress"]["initial yield"]["Y"]
    return yield_tol * Y / two_mu_scale_factor(params)


def cond_residual(
        trial_yield_fun: JaxArray, C_e: JaxArray, C_p: JaxArray,
        tol: Scalar,
) -> JaxArray:
    """Select ``C_p`` (plastic-branch residual) when the elastic trial lies
    outside the yield surface, otherwise ``C_e`` (elastic).

    Taken from the trial state, so it is fixed for the whole stress update.

    ``tol`` sits above zero rather than below: a trial just inside the
    surface would need a negative plastic increment to reach it, so calling
    it plastic admits an inadmissible root.

    Implemented with ``jnp.where`` (smooth pointwise select) rather
    than ``lax.cond``: both ``C_e`` and ``C_p`` are pure value
    expressions evaluated unconditionally upstream, so there is no
    branch-pruning benefit from ``lax.cond`` here, and ``jnp.where``
    auto-transposes cleanly under arbitrarily deep AD nesting (whereas
    ``lax.cond``'s transposition introduces an internal
    ``stop_gradient`` that cannot be transposed when this routine is
    composed inside an outer implicit solver's reverse-mode rule).
    """
    return jnp.where(trial_yield_fun > tol, C_p, C_e)
