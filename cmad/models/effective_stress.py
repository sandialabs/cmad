"""
These all assume 3D cauchy stress inputs
"""
import jax.numpy as jnp

from functools import partial
from jax import grad
from jax.lax import cond, fori_loop

from cmad.models.elastic_constants import compute_mu
from cmad.parameters.parameters import unpack_elastic_params
from cmad.solver.nonlinear_solver import make_newton_solve
from cmad.util.jax_eigen_decomposition import jax_compute_eigenvalues
from cmad.verification.functions import jax_barlat_yield


def conventional_effective_stress_fun(effective_stress_type):
    if effective_stress_type == "J2":
        return J2_effective_stress
    elif effective_stress_type == "hill":
        return hill_effective_stress
    elif effective_stress_type == "barlat":
        return barlat_effective_stress
    elif effective_stress_type == "hosford":
        return hosford_effective_stress
    else:
        raise NotImplementedError


def J2_effective_stress(cauchy, params):
    hydro_cauchy = jnp.trace(cauchy) / 3.
    s = cauchy - hydro_cauchy * jnp.eye(3)
    snorm = jnp.sqrt(jnp.sum(s * s))
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


def flatten_barlat_params(params):
    barlat_coeffs = params["effective stress"]["barlat"]

    sp_12, sp_13 = barlat_coeffs["sp_12"], barlat_coeffs["sp_13"]
    sp_21, sp_23 = barlat_coeffs["sp_21"], barlat_coeffs["sp_23"]
    sp_31, sp_32 = barlat_coeffs["sp_31"], barlat_coeffs["sp_32"]
    sp_44, sp_55, sp_66 = barlat_coeffs["sp_44"], barlat_coeffs["sp_55"], \
        barlat_coeffs["sp_66"]

    dp_12, dp_13 = barlat_coeffs["dp_12"], barlat_coeffs["dp_13"]
    dp_21, dp_23 = barlat_coeffs["dp_21"], barlat_coeffs["dp_23"]
    dp_31, dp_32 = barlat_coeffs["dp_31"], barlat_coeffs["dp_32"]
    dp_44, dp_55, dp_66 = barlat_coeffs["dp_44"], barlat_coeffs["dp_55"], \
        barlat_coeffs["dp_66"]

    a = barlat_coeffs["a"]

    flat_barlat_coeffs = jnp.array([
        sp_12, sp_13, sp_21, sp_23, sp_31, sp_32, sp_44, sp_55, sp_66,
        dp_12, dp_13, dp_21, dp_23, dp_31, dp_32, dp_44, dp_55, dp_66,
        a
    ])

    return flat_barlat_coeffs


def barlat_effective_stress(cauchy, params):
    flat_barlat_coeffs = flatten_barlat_params(params)

    return jax_barlat_yield(cauchy, flat_barlat_coeffs)


def beta_initial_guess(cauchy, equivalent_stress, tol=1e-14):
    phi_J2 = J2_effective_stress(cauchy, None)
    is_J2_near_zero = jnp.isclose(phi_J2, 0., tol, tol)
    initial_guess = equivalent_stress / phi_J2
    return cond(is_J2_near_zero, lambda x: -1., lambda x: x, initial_guess)


def beta_make_newton_solve(effective_stress_fun, equivalent_stress):
    def residual(beta, initial_guess, cauchy, params):
        scaled_input_effective_stress = \
            effective_stress_fun(beta * cauchy, params)
        return scaled_input_effective_stress / equivalent_stress - 1.

    return make_newton_solve(residual, 1.)


def make_safe_update_fun(initial_guess, cauchy, params, update_fun):
    return cond(initial_guess < 0., lambda *args: 1., update_fun,
                initial_guess, cauchy, params)


def scaled_hybrid_hill_effective_stress(cauchy, params, nn_fun, safe_update):
    Y = params["flow stress"]["initial yield"]["Y"]
    beta = safe_update(beta_initial_guess(cauchy, Y), cauchy, params)
    jax_print("beta = {x}", x=beta)
    scaled_cauchy = beta * cauchy
    return hybrid_hill_effective_stress(scaled_cauchy, params, nn_fun) / beta


def scaled_effective_stress(cauchy, params,
        effective_stress_fun, update_fun, tol=1e-14):

    def beta_effective_stress(cauchy, params, beta):
        return effective_stress_fun(beta * cauchy, params) / beta

    phi_J2 = J2_effective_stress(cauchy, None)
    is_J2_near_zero = jnp.isclose(phi_J2, 0., tol, tol)
    initial_guess = params["flow stress"]["initial yield"]["Y"] / phi_J2
    beta = update_fun(initial_guess, cauchy, params)

    return cond(is_J2_near_zero, lambda *args: phi_J2, beta_effective_stress,
        cauchy, params, beta)


def hybrid_hill_effective_stress(cauchy, params, nn_fun):
    phi_hill = hill_effective_stress(cauchy, params)
    hydro_cauchy = jnp.trace(cauchy) / 3.
    dev_cauchy = cauchy - hydro_cauchy * jnp.eye(3)
    # needed for non-symmetric effective stress functions
    s = 0.5 * (dev_cauchy + dev_cauchy.T)
    flat_s = jnp.array([s[0, 0], s[1, 1], s[2, 2],
        s[0, 1], s[0, 2], s[1, 2]])
    phi_discrepancy = nn_fun(flat_s,
        params["effective stress"]["neural network"])

    return phi_hill + phi_discrepancy[0]
    #return phi_discrepancy[0] # NN only fit


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
