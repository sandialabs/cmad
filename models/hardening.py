import numpy as np

import jax.numpy as jnp

from functools import partial


def voce_hardening(alpha, voce_params):
    S = voce_params["S"]
    D = voce_params["D"]

    return S * (1. - jnp.exp(-D * alpha))


def linear_hardening(alpha, linear_params):
    K = linear_params["K"]

    return K * alpha


def get_hardening_funs():
    hardening_funs = {"voce": voce_hardening, "linear": linear_hardening}
    return hardening_funs


def combined_hardening_fun(alpha, params, hardening_funs):
    H = jnp.array([hardening_funs[htype](alpha, hparams)
                   for htype, hparams in params.items()])

    return jnp.sum(H)
