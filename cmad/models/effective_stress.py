"""
These all assume 3D cauchy stress inputs
"""
import jax.numpy as jnp

from functools import partial
from jax import grad
from jax.debug import print as jax_print
from jax.lax import cond, fori_loop

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


def scaled_input_phi(beta, cauchy, params, phi_function):
    return phi_function(beta * cauchy, params)


def beta_newton(ii, carry,
        scaled_phi_fun, scaled_phi_fun_grad):

    beta, params, cauchy = carry
    Y = params["flow stress"]["initial yield"]["Y"]

    res = scaled_phi_fun(beta, cauchy, params) / Y - 1.
    Jac = scaled_phi_fun_grad(beta, cauchy, params) / Y

    res_norm = jnp.abs(res)
    #jax_print("Y = {x}", x=Y)
    #jax_print("beta = {x}", x=beta)
    #jax_print("res norm = {x}", x=res_norm)
    #jax_print("cauchy = {x}", x=cauchy)

    delta_beta = -res / Jac
    beta += delta_beta

    return beta, params, cauchy


def trivial_branch(cauchy, params):
    return 0.


def hybrid_hill_scaled_output(cauchy, params, nn_fun, max_iters):

    phi_fun = partial(hybrid_hill_effective_stress, nn_fun=nn_fun)
    scaled_phi_fun = partial(scaled_input_phi, phi_function=phi_fun)
    scaled_phi_fun_grad = grad(scaled_phi_fun)
    Y = params["flow stress"]["initial yield"]["Y"]
    pt_beta_newton = partial(beta_newton,
        scaled_phi_fun=scaled_phi_fun, scaled_phi_fun_grad=scaled_phi_fun_grad)
    # could consider using a better initial guess
    beta = fori_loop(0, max_iters, pt_beta_newton, (1., params, cauchy))[0]

    return Y / beta


def branching_hybrid_hill_effective_stress(cauchy, params, nn_fun, max_iters):
    pt_hybrid_hill_scaled_output = partial(hybrid_hill_scaled_output,
        nn_fun=nn_fun, max_iters=max_iters)
    phi = cond(jnp.linalg.norm(cauchy) != 0.,
        pt_hybrid_hill_scaled_output, trivial_branch,
        cauchy, params)

    return phi


def hybrid_hill_effective_stress(cauchy, params, nn_fun):
    phi_hill = hill_effective_stress(cauchy, params)
    hydro_cauchy = jnp.trace(cauchy) / 3.
    s = cauchy - hydro_cauchy * jnp.eye(3)
    flat_s = jnp.array([s[0, 0], s[1, 1], s[2, 2],
        s[0, 1], s[0, 2], s[1, 2]])
    phi_discrepancy = nn_fun(flat_s,
        params["effective stress"]["neural network"])

    return phi_hill + phi_discrepancy[0]
    #return phi_discrepancy[0] # NN only fit

# branch 1: return 0 if jnp.linalg.norm(cauchy) == 0.
# branch 2: return Y / beta otherwise
#  - find beta by solving the NL problem for a fixed # of iterations


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
