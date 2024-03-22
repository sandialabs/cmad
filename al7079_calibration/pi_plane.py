import numpy as np
import matplotlib.pyplot as plt
import pickle

import jax.numpy as jnp

from jax import grad, jit

from cmad.neural_networks.input_convex_neural_network \
    import InputConvexNeuralNetwork
from cmad.verification.functions import (hill_yield, hill_yield_normal,
    jax_hill_yield, jax_hill_yield_normal)
from functools import partial


# branch import
from cmad.models.effective_stress import (hill_effective_stress,
    hybrid_hill_effective_stress)

# local import
from al7079 import (slab_data, calibration_weights,
    calibrated_hill_coefficients)


def scaled_input_phi(beta, cauchy, phi_function):
    return phi_function(beta * cauchy)

def newton_solve(cauchy, Y, phi_fun, phi_fun_grad,
                 max_iters=10, abs_tol=1e-14, rel_tol=1e-14):

    beta = Y
    converged = False
    ii = 0
    res_norm_0 = 1.

    while ii < max_iters and not converged:
        res = float(phi_fun(beta, cauchy)) / Y - 1.
        res_norm = np.abs(res)

        if ii == 0:
            res_norm_0 = res_norm
            res_norm_rel = 1.
        else:
            res_norm_rel = res_norm / res_norm_0

        if res_norm_rel < rel_tol or res_norm < abs_tol:
            converged = True
            break

        Jac = float(phi_fun_grad(beta, cauchy)) / Y
        delta_beta = -res / Jac
        beta += delta_beta
        #print(f"ii = {ii}, delta_beta = {delta_beta}")

        ii += 1

    return beta


Y = 525.
hill_coeffs = calibrated_hill_coefficients()
hill_effective_stress = partial(hill_yield, hill_params=hill_coeffs)
jax_hill_effective_stress = partial(jax_hill_yield, hill_params=hill_coeffs)

num_angles = 720
psi_angles = np.linspace(0., 2. * np.pi, num_angles, endpoint=False)
theta_angles = psi_angles - np.pi / 6.  # angle coordinate in pi plane

#effective_stresses = np.array([hill_effective_stress(np.diag(ps)) \
#    for ps in principal_stresses])
#scaling_factors = Y / effective_stresses
#pi_coords = scaling_factors[:, np.newaxis] \
#    * np.column_stack((np.cos(theta_angles), np.sin(theta_angles)))

principal_stresses = 2. / 3. * np.cos(np.column_stack((psi_angles,
    psi_angles - 2. * np.pi / 3., psi_angles + 2. * np.pi / 3.)))


layer_widths = [6, 6, 1]
phi_disc_nn = InputConvexNeuralNetwork(layer_widths, input_scale=0.01,
    output_scale=100.)
nn_params = pickle.load(open("nn_params.p", "rb"))
hybrid_effective_stress = partial(hybrid_hill_effective_stress,
    params=nn_params["plastic"],
    nn_fun=phi_disc_nn.evaluate)

#pt_scaled_input_phi = jit(partial(scaled_input_phi,
#    phi_function=jax_hill_effective_stress))
pt_scaled_input_phi = jit(partial(scaled_input_phi,
    phi_function=hybrid_effective_stress))
pt_scaled_input_phi_grad = jit(grad(pt_scaled_input_phi))
newton = partial(newton_solve, phi_fun=pt_scaled_input_phi,
    phi_fun_grad=pt_scaled_input_phi_grad)
scaling_factors_newton = np.array([newton(np.diag(ps), Y) \
    for ps in principal_stresses])
pi_coords = scaling_factors_newton[:, np.newaxis] \
    * np.column_stack((np.cos(theta_angles), np.sin(theta_angles)))

num_axis_points = 1000
axis_scale_factor = 1.5 * Y
axis_parametric_coord = np.linspace(-1., 1., num_axis_points) \
    * axis_scale_factor
s1_axis = np.column_stack((np.sqrt(3.) / 2. * axis_parametric_coord,
    -0.5 * axis_parametric_coord))
s2_axis = np.column_stack((0. * axis_parametric_coord,
    axis_parametric_coord))
s3_axis = np.column_stack((-s1_axis[:, 0], s1_axis[:, 1]))

fig, ax = plt.subplots(figsize=(11, 8))
plt.plot(s1_axis[:, 0], s1_axis[:, 1], color="black", zorder=0)
plt.plot(s2_axis[:, 0], s2_axis[:, 1], color="black", zorder=0)
plt.plot(s3_axis[:, 0], s3_axis[:, 1], color="black", zorder=0)
plt.scatter(pi_coords[:, 0], pi_coords[:, 1], zorder=1, color="blue")
plt.scatter(Y * np.cos(theta_angles[0]), Y * np.sin(theta_angles[0]),
    color="red", zorder=2)
ax.axis("equal")
