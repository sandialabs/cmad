from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from cmad.typing import JaxArray


def voce_hardening(alpha: JaxArray, voce_params: dict[str, Any]) -> JaxArray:
    S = voce_params["S"]
    D = voce_params["D"]

    return S * (1. - jnp.exp(-D * alpha))


def voce_modulus_hardening(
        alpha: JaxArray, voce_params: dict[str, Any],
) -> JaxArray:
    """Voce saturation parameterized by the initial hardening modulus.

    ``H`` is the slope at zero plastic strain and ``D`` the saturation
    rate, so the saturation increment is ``H / D``. The same curve as
    :func:`voce_hardening` at ``S = H / D``.
    """
    H = voce_params["H"]
    D = voce_params["D"]

    return H / D * (1. - jnp.exp(-D * alpha))


def linear_hardening(alpha: JaxArray, linear_params: dict[str, Any]) -> JaxArray:
    K = linear_params["K"]

    return K * alpha


def get_hardening_funs() -> dict[str, Callable[..., JaxArray]]:
    hardening_funs = {"voce": voce_hardening,
                      "voce_modulus": voce_modulus_hardening,
                      "linear": linear_hardening}
    return hardening_funs


def combined_hardening_fun(
        alpha: JaxArray, params: dict[str, Any],
        hardening_funs: dict[str, Callable[..., JaxArray]],
) -> JaxArray:
    H = jnp.array([hardening_funs[htype](alpha, hparams)
                   for htype, hparams in params.items()])

    return jnp.sum(H)
