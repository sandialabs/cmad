import numpy as np
import pickle

from matplotlib import pyplot as plt

from functools import partial
from jax import jit

from cmad.calibrations.al7079.support import (
    slab_data,
    calibration_weights,
    calibrated_hill_coefficients,
    params_hill_voce,
    params_icnn_hybrid_hill_voce
)
from cmad.calibrations.al7079.support import (
    calibrated_barlat_coefficients,
    calibrated_hill_coefficients
)
from cmad.models.effective_stress import(
    beta_initial_guess,
    beta_make_newton_solve,
    J2_effective_stress,
    hybrid_hill_effective_stress,
    hill_effective_stress,
    make_safe_update_fun,
    scaled_effective_stress,
)
from cmad.neural_networks.input_convex_neural_network \
    import InputConvexNeuralNetwork
from cmad.parameters.parameters import Parameters
from cmad.util.dev_plane_transformations import (
    compute_forward_and_backward_matrices,
    compute_matrix_from_projection,
    setup_dev_plane_plot
)
from cmad.verification.functions import (
    jax_barlat_yield,
    jax_hill_yield
)


def compute_phi_and_dev_coords(effective_stress_fun, betas,
        cauchys, dev_principal_stresses):
    phi = np.array([effective_stress_fun(beta * cauchy)
        for beta, cauchy in zip(betas, cauchys)
    ])
    dev_coords = np.vstack([
        beta * F @ dev_projection_values
        for beta, dev_projection_values in zip(betas, dev_principal_stresses)
    ])

    return phi, dev_coords


plt.close("all")
use_scaling = True
num_angles = 720

alpha_angles, alpha_sigma_c_values, alpha_ratio_c_values, R_alphas = \
    slab_data("alpha")
beta_angles, beta_sigma_c_values, beta_ratio_c_values, R_betas = \
    slab_data("beta")
gamma_angles, gamma_sigma_c_values, gamma_ratio_c_values, R_gammas = \
    slab_data("gamma")
angles = np.r_[alpha_angles, beta_angles, gamma_angles]
specimen_indices= np.arange(len(angles))
sigma_c_values = np.r_[alpha_sigma_c_values, beta_sigma_c_values, \
     gamma_sigma_c_values]
R_matrices = R_alphas + R_betas + R_gammas
Y = alpha_sigma_c_values[0]
yield_surface_tol = 1e-10

jit_beta_initial_guess = jit(beta_initial_guess)

# J2
jit_J2_effective_stress = jit(partial(J2_effective_stress,
    params=None))

# Hill
hill_coeffs = calibrated_hill_coefficients()
hill_safe_update = jit(partial(make_safe_update_fun,
    update_fun=beta_make_newton_solve(jax_hill_yield, Y)
))
jit_hill_effective_stress = jit(partial(jax_hill_yield,
    hill_params=hill_coeffs
))

# Barlat
barlat_coeffs = calibrated_barlat_coefficients()
barlat_safe_update = jit(partial(make_safe_update_fun,
    update_fun=beta_make_newton_solve(jax_barlat_yield, Y)
))
jit_barlat_effective_stress = jit(partial(jax_barlat_yield,
    barlat_params=barlat_coeffs
))

# ICNN + Hill
nn_props = pickle.load(open("nn_props_16.p", "rb"))
hybrid_model_params = nn_props["params"]
input_scaler = nn_props["input scaler"]
output_scaler = nn_props["output scaler"]
layer_widths = nn_props["layer widths"]
phi_disc_nn = InputConvexNeuralNetwork(layer_widths,
    input_scaler=input_scaler,
    output_scaler=output_scaler)
hybrid_effective_stress = partial(hybrid_hill_effective_stress,
    nn_fun=phi_disc_nn.evaluate)
jit_hybrid_effective_stress = jit(hybrid_effective_stress)

hybrid_update_fun = beta_make_newton_solve(hybrid_effective_stress, Y)
nn_safe_update = jit(partial(make_safe_update_fun,
    update_fun=hybrid_update_fun
))
scaled_hybrid_effective_stress = jit(partial(scaled_effective_stress,
    effective_stress_fun=hybrid_effective_stress,
    update_fun=hybrid_update_fun
))

#specimen_indices = [0]
for specimen_idx in specimen_indices:
    sigma_c = sigma_c_values[specimen_idx]
    R = R_matrices[specimen_idx].T

    dev_plane_angles = np.mod(np.linspace(0., 2. * np.pi, num_angles,
        endpoint=False) - np.pi / 6., 2. * np.pi)
    dev_plane_radii = Y * np.ones(num_angles)
    dev_plane_coords = np.c_[
        dev_plane_radii * np.cos(dev_plane_angles),
        dev_plane_radii * np.sin(dev_plane_angles)
    ]
    F, B = compute_forward_and_backward_matrices(use_scaling)
    dev_principal_stresses = (B @ dev_plane_coords.T).T

    fig, ax = setup_dev_plane_plot(axis_scale_factor=1.5 * Y)

    cauchys = [compute_matrix_from_projection(
        dev_projected_stresses, R)
        for dev_projected_stresses in dev_principal_stresses
    ]

    # J2
    J2_betas = np.array([jit_beta_initial_guess(
        cauchy, Y) for cauchy in cauchys
    ])
    J2_phi, J2_dev_coords = compute_phi_and_dev_coords(jit_J2_effective_stress,
        J2_betas, cauchys, dev_principal_stresses)
    assert np.linalg.norm(J2_phi - Y) / num_angles < yield_surface_tol

    # Hill
    hill_betas = np.array([hill_safe_update(jit_beta_initial_guess(
        cauchy, Y), cauchy, hill_coeffs)
        for cauchy in cauchys
    ])
    hill_phi, hill_dev_coords = compute_phi_and_dev_coords(jit_hill_effective_stress,
        hill_betas, cauchys, dev_principal_stresses)
    assert np.linalg.norm(hill_phi - Y) / num_angles < yield_surface_tol

    # Barlat
    barlat_betas = np.array([barlat_safe_update(jit_beta_initial_guess(
        cauchy, Y), cauchy, calibrated_barlat_coefficients())
        for cauchy in cauchys
    ])
    barlat_phi, barlat_dev_coords = compute_phi_and_dev_coords(jit_barlat_effective_stress,
        barlat_betas, cauchys, dev_principal_stresses)
    assert np.linalg.norm(barlat_phi - Y) / num_angles < yield_surface_tol

    # ICNN + Hill
    nn_betas = np.array([nn_safe_update(jit_beta_initial_guess(
        cauchy, Y), cauchy, hybrid_model_params["plastic"])
        for cauchy in cauchys
    ])
    nn_phi = np.array([jit_hybrid_effective_stress(beta * cauchy,
        hybrid_model_params["plastic"])
        for beta, cauchy in zip(nn_betas, cauchys)
    ])
    assert np.linalg.norm(nn_phi - Y) / num_angles < yield_surface_tol

    # not supposed to be equal to Y --- don't panic
    scaled_nn_phi = np.array([scaled_hybrid_effective_stress(cauchy,
        hybrid_model_params["plastic"])
        for beta, cauchy in zip(nn_betas, cauchys)
    ])
    nn_dev_coords = np.vstack([
        beta * F @ dev_projection_values
        for beta, dev_projection_values in zip(nn_betas, dev_principal_stresses)
    ])

    # check that this is zero (safe)
    zero_cauchy = np.zeros((3, 3))
    scaled_nn_phi_zero = scaled_hybrid_effective_stress(zero_cauchy,
        hybrid_model_params["plastic"])

    ax.scatter(dev_plane_coords[:, 0], dev_plane_coords[:, 1], color="black",
        zorder=1, marker="*", s=150, label="Unit Circle")
    ax.scatter(J2_dev_coords[:, 0], J2_dev_coords[:, 1], color="red",
        zorder=2, label="J2")
    ax.scatter(hill_dev_coords[:, 0], hill_dev_coords[:, 1], color="green",
        zorder=3, label="Hill")
    ax.scatter(barlat_dev_coords[:, 0], barlat_dev_coords[:, 1], color="purple",
        zorder=4, label="Barlat")
    ax.scatter(nn_dev_coords[:, 0], nn_dev_coords[:, 1], color="cyan",
        zorder=5, label="ICNN")
    ax.legend(loc="upper right", fontsize=16)
    fig.savefig(f"dev_plane_specimen_{specimen_idx}.pdf")
