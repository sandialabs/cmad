"""
These all use 3D tensors
"""
import jax.numpy as jnp

from cmad.models.elastic_constants import compute_lambda, compute_mu
from cmad.parameters.parameters import unpack_elastic_params


# form used by elastic-plastic models
def isotropic_linear_elastic_stress(elastic_strain, params):
    E, nu = unpack_elastic_params(params)
    lame_lambda = compute_lambda(E, nu)
    lame_mu = compute_mu(E, nu)

    cauchy = lame_lambda * jnp.trace(elastic_strain) * jnp.eye(3) \
        + 2. * lame_mu * elastic_strain

    return cauchy


# alternative form used by elasticity-only models
def isotropic_linear_elastic_cauchy_stress(F, u, params):
    I = jnp.eye(3)
    grad_u = F - I
    epsilon = 0.5 * (grad_u + grad_u.T)
    trace_epsilon = jnp.trace(epsilon)
    dev_epsilon = epsilon - trace_epsilon / 3. * I

    kappa = params["elastic"]["kappa"]
    mu = params["elastic"]["mu"]

    return kappa * trace_epsilon * I + 2. * mu * dev_epsilon


def compressible_neohookean_cauchy_stress(F, u, params):
    J = jnp.linalg.det(F)
    Jm23 = jnp.cbrt(J)**-2

    I = jnp.eye(3)
    bbar = Jm23 * (F @ F.T)
    dev_bbar = bbar - jnp.trace(bbar) / 3. * I

    kappa = params["elastic"]["kappa"]
    mu = params["elastic"]["mu"]

    return J**-1 * (0.5 * kappa * (J**2 - 1.) * I + mu * dev_bbar)


def two_mu_scale_factor(params):
    elastic_params_keys = params["elastic"].keys()
    if "mu" in elastic_params_keys:
        return 2. * params["elastic"]["mu"]
    elif "E" in elastic_params_keys and "nu" in elastic_params_keys:
        E = params["elastic"]["E"]
        nu = params["elastic"]["nu"]
        return 2. * compute_mu(E, nu)
    else:
        raise NotImplementedError
