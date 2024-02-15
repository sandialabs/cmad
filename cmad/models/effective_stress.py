"""
These all assume 3D cauchy stress inputs
"""
import jax.numpy as jnp

from cmad.models.elastic_constants import compute_mu
from cmad.parameters.parameters import unpack_elastic_params


def effective_stress_fun(effective_stress_type):
    if effective_stress_type == "J2":
        return J2_effective_stress
    elif effective_stress_type == "hill":
        return hill_effective_stress
    elif effective_stress_type == "hosford":
        return hosford_effective_stress
    else:
        raise NotImplementedError


def J2_effective_stress(cauchy, params):
    hydro_cauchy = jnp.trace(cauchy) / 3.
    s = cauchy - hydro_cauchy * jnp.eye(3)
    snorm = jnp.linalg.norm(s)
    phi = jnp.sqrt(3. / 2.) * snorm
    return phi


def hill_effective_stress(cauchy, params):
    hill_coeffs = params["effective stress"]["hill"]
    F, G, H = hill_coeffs["F"], hill_coeffs["G"], hill_coeffs["H"]
    L, M, N = hill_coeffs["L"], hill_coeffs["M"], hill_coeffs["N"]

    phi = jnp.sqrt(F * (cauchy[1, 1] - cauchy[2, 2])**2
                   + G * (cauchy[2, 2] - cauchy[0, 0])**2
                   + H * (cauchy[0, 0] - cauchy[1, 1])**2
                   + L * (cauchy[2, 1]**2 + cauchy[1, 2]**2)
                   + M * (cauchy[2, 0]**2 + cauchy[0, 2]**2)
                   + N * (cauchy[1, 0]**2 + cauchy[0, 1]**2))

    return phi


# only working for diagonal cauchy stress now
def hosford_effective_stress(cauchy, params):
    vm_stress = J2_effective_stress(cauchy, params)
    a = params["effective stress"]["hosford"]["a"]
    scaled_cauchy = cauchy / vm_stress
    cauchy_diff_01 = jnp.abs(scaled_cauchy[0, 0] - scaled_cauchy[1, 1])**a
    cauchy_diff_12 = jnp.abs(scaled_cauchy[1, 1] - scaled_cauchy[2, 2])**a
    cauchy_diff_20 = jnp.abs(scaled_cauchy[2, 2] - scaled_cauchy[0, 0])**a
    phi = vm_stress \
        * (0.5 * (cauchy_diff_01 + cauchy_diff_12 + cauchy_diff_20))**(a**-1)
    return phi
