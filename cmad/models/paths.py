import jax.numpy as jnp

from cmad.typing import JaxArray


def cond_residual(
        f: JaxArray, C_e: JaxArray, C_p: JaxArray, tol: float,
) -> JaxArray:
    """Select ``C_p`` (plastic-branch residual) when the yield function
    indicates yielding, otherwise ``C_e`` (elastic).

    Plastic if ``f > tol`` or ``|f| < tol``; elastic otherwise. The
    ``|f| < tol`` band keeps the residual on the plastic branch within
    a small neighbourhood of the yield surface (numerical robustness
    around ``f ≈ 0``); outside that band the sign of ``f`` decides.

    Implemented with ``jnp.where`` (smooth pointwise select) rather
    than ``lax.cond``: both ``C_e`` and ``C_p`` are pure value
    expressions evaluated unconditionally upstream, so there is no
    branch-pruning benefit from ``lax.cond`` here, and ``jnp.where``
    auto-transposes cleanly under arbitrarily deep AD nesting (whereas
    ``lax.cond``'s transposition introduces an internal
    ``stop_gradient`` that cannot be transposed when this routine is
    composed inside an outer implicit solver's reverse-mode rule).
    """
    is_plastic = jnp.logical_or(f > tol, jnp.abs(f) < tol)
    return jnp.where(is_plastic, C_p, C_e)
