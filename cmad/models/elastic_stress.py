"""
These all use 3D tensors
"""
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from cmad.models.elastic_constants import ElasticConstants
from cmad.typing import JaxArray, Scalar


# form used by elastic-plastic models
def isotropic_linear_elastic_stress(
        elastic_strain: JaxArray, params: dict[str, Any],
) -> JaxArray:
    elastic_constants = ElasticConstants.from_params(params["elastic"])
    return (
        elastic_constants.lmbda * jnp.trace(elastic_strain) * jnp.eye(3)
        + 2. * elastic_constants.mu * elastic_strain
    )


# alternative form used by elasticity-only models
def isotropic_linear_elastic_cauchy_stress(
        F: JaxArray, params: dict[str, Any],
) -> JaxArray:
    I = jnp.eye(3)
    grad_u = F - I
    epsilon = 0.5 * (grad_u + grad_u.T)
    trace_epsilon = jnp.trace(epsilon)
    dev_epsilon = epsilon - trace_epsilon / 3. * I

    elastic_constants = ElasticConstants.from_params(params["elastic"])
    return (
        elastic_constants.kappa * trace_epsilon * I
        + 2. * elastic_constants.mu * dev_epsilon
    )


def compressible_neohookean_cauchy_stress(
        F: JaxArray, params: dict[str, Any],
) -> JaxArray:
    J = jnp.linalg.det(F)
    Jm23 = jnp.cbrt(J)**-2

    I = jnp.eye(3)
    bbar = Jm23 * (F @ F.T)
    dev_bbar = bbar - jnp.trace(bbar) / 3. * I

    elastic_constants = ElasticConstants.from_params(params["elastic"])
    return J**-1 * (
        0.5 * elastic_constants.kappa * (J**2 - 1.) * I
        + elastic_constants.mu * dev_bbar
    )


def conventional_elastic_stress_fun(
        elastic_stress_type: str,
) -> Callable[..., JaxArray]:
    if elastic_stress_type == "isotropic_linear":
        return isotropic_linear_elastic_cauchy_stress
    elif elastic_stress_type == "neohookean":
        return compressible_neohookean_cauchy_stress
    else:
        raise NotImplementedError(
            f"unknown elastic_stress type: '{elastic_stress_type}'",
        )


def two_mu_scale_factor(params: dict[str, Any]) -> Scalar:
    return 2. * ElasticConstants.from_params(params["elastic"]).mu
