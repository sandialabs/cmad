import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

from cmad.calibrations.al7079.support import (
    slab_data,
    calibration_weights,
    calibrated_barlat_coefficients,
    calibrated_hill_coefficients
)
from cmad.verification.functions import (
    jax_barlat_yield,
    jax_hill_yield,
    hill_yield,
    hill_yield_normal,
)

from functools import partial
from jax import jit, grad
from scipy.optimize import fmin_l_bfgs_b


def hill_analytic_compute_yield_and_normal(R_matrices, sigma_values, params):
    unit_sigma = np.zeros((3, 3))
    unit_sigma[1, 1] = 1.
    yield_values = np.zeros_like(sigma_values)
    ratio_values = np.zeros_like(sigma_values)

    for idx, (R, sigma_c) in enumerate(zip(R_matrices, sigma_values)):
        sigma = sigma_c * unit_sigma
        sigma_mat = R.T @ sigma @ R
        phi = hill_yield(sigma_mat, params)
        normal_mat = hill_yield_normal(sigma_mat, params)
        normal = R @ normal_mat @ R.T
        yield_values[idx] = phi
        ratio_values[idx] = normal[2, 2] / normal[0, 0]

    return yield_values, ratio_values


def jax_compute_yield_and_normal(R, sigma_c, params,
        jax_yield_fun):
    unit_sigma = jnp.array([[0., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.]])
    sigma = sigma_c * unit_sigma
    sigma_mat = R.T @ sigma @ R
    phi = jax_yield_fun(sigma_mat, params)
    normal_fun = grad(jax_yield_fun)
    normal_mat = normal_fun(sigma_mat, params)
    normal = R @ normal_mat @ R.T
    ratio = normal[2, 2] / normal[0, 0]
    return jnp.atleast_2d([phi, ratio]).T


def compute_all_yields_and_normals(params, rotations, sigma_values,
        jax_compute_yield_and_normal_fun):
    vals = jnp.hstack([jax_compute_yield_and_normal_fun(R, sigma_c, params)
        for R, sigma_c in zip(rotations, sigma_values)])
    return vals


def objective(params, rotations, sigma_values, ratio_values, Y, weights,
        compute_all_yields_and_normals_fun):
    w_sigma, w_ratio = weights
    vals = compute_all_yields_and_normals_fun(params, rotations, sigma_values)
    obj_sigma = w_sigma * jnp.sum(jnp.array([(sigma / Y - 1.)**2 \
        for sigma in vals[0, :]]))
    obj_ratio = w_ratio * jnp.sum(jnp.array([(pred_ratio / meas_ratio - 1.)**2 \
        for pred_ratio, meas_ratio in zip(vals[1, :], ratio_values)]))
    return obj_sigma + obj_ratio


def obj_and_grad(params, objective_fun):
    obj_fun_grad = grad(objective_fun)
    J = objective_fun(params)
    dJ_dp = obj_fun_grad(params)

    return J, dJ_dp


plt.close("all")

alpha_angles, alpha_sigma_c_values, alpha_ratio_c_values, R_alphas = \
    slab_data("alpha")
beta_angles, beta_sigma_c_values, beta_ratio_c_values, R_betas = \
    slab_data("beta")
gamma_angles, gamma_sigma_c_values, gamma_ratio_c_values, R_gammas = \
    slab_data("gamma")

angles = np.r_[alpha_angles, beta_angles, gamma_angles]
spec_num = np.arange(len(angles))
sigma_c_values = np.r_[alpha_sigma_c_values, beta_sigma_c_values, \
    gamma_sigma_c_values]
ratio_c_values = np.r_[alpha_ratio_c_values, beta_ratio_c_values, \
    gamma_ratio_c_values]
R_matrices = R_alphas + R_betas + R_gammas
Y = alpha_sigma_c_values[0]

# hill fit
hill_fit_params = calibrated_hill_coefficients()
jax_hill_compute_yield_and_normal=partial(jax_compute_yield_and_normal,
    jax_yield_fun=jax_hill_yield)
hill_compute_all_yields_and_normals = partial(compute_all_yields_and_normals,
    jax_compute_yield_and_normal_fun=jax_hill_compute_yield_and_normal)
hill_obj_fun = partial(objective, rotations=R_matrices,
    sigma_values=sigma_c_values, ratio_values=ratio_c_values, Y=Y,
    weights=calibration_weights(),
    compute_all_yields_and_normals_fun=hill_compute_all_yields_and_normals)
opt_obj_and_grad = jit(partial(obj_and_grad, objective_fun=hill_obj_fun))
bounds = np.c_[np.ones_like(hill_fit_params) * 0.1,
    np.ones_like(hill_fit_params) * 10.]
opt_params, fun_val, cvg_dict = fmin_l_bfgs_b(opt_obj_and_grad, hill_fit_params,
    bounds=bounds)

jit_hill_compute_all_yields_and_normals = \
    jit(hill_compute_all_yields_and_normals)
orig_fit_vals = jit_hill_compute_all_yields_and_normals(hill_fit_params, R_matrices,
    sigma_c_values)
pred_vals = jit_hill_compute_all_yields_and_normals(opt_params, R_matrices,
    sigma_c_values)

fig1, ax1 = plt.subplots(figsize=(22, 8), ncols=2)

ax1[0].plot(spec_num, Y * np.ones_like(angles), color="black", label="$Y$",
    zorder=0)
ax1[0].scatter(spec_num, orig_fit_vals[0, :], color="white", edgecolor="red",
    zorder=1, label="Paper fit")
ax1[0].scatter(spec_num, pred_vals[0, :], color="white", edgecolor="blue",
    zorder=2, label="BFGS fit")
ax1[0].set_xlabel("Specimen Number", fontsize=22)
ax1[0].set_ylabel("Predicted Effective Stress", fontsize=22)
ax1[0].set_title("Hill Yield Predictions", fontsize=22)
ax1[0].legend(loc="best", fontsize=20)

ax1[1].scatter(spec_num, ratio_c_values, color="black", marker="x",
    label="Experiment", zorder=0, s=50)
ax1[1].scatter(spec_num, orig_fit_vals[1, :], color="white", edgecolor="red",
    zorder=1, label="Paper fit")
ax1[1].scatter(spec_num, pred_vals[1, :], color="white", edgecolor="blue",
    zorder=2, label="BFGS fit")
ax1[1].set_xlabel("Specimen Number", fontsize=22)
ax1[1].set_ylabel("Predicted Lankford Ratio", fontsize=22)
ax1[1].set_title("Hill Lankford Ratio Predictions", fontsize=22)
ax1[1].legend(loc="best", fontsize=20)

# use analytical expressions for hill effective stress and its normal
alpha_yield_values, alpha_ratio_values = \
    hill_analytic_compute_yield_and_normal(R_alphas, alpha_sigma_c_values,
    hill_fit_params)
beta_yield_values, beta_ratio_values = \
    hill_analytic_compute_yield_and_normal(R_betas, beta_sigma_c_values,
    hill_fit_params)
gamma_yield_values, gamma_ratio_values = \
    hill_analytic_compute_yield_and_normal(R_gammas, gamma_sigma_c_values,
    hill_fit_params)

analytical_orig_fit_vals = np.vstack((
    np.r_[alpha_yield_values, beta_yield_values, gamma_yield_values],
    np.r_[alpha_ratio_values, beta_ratio_values, gamma_ratio_values])
)
assert np.linalg.norm(orig_fit_vals - analytical_orig_fit_vals) < 1e-12

# barlat fit
barlat_fit_params = calibrated_barlat_coefficients()
jax_barlat_compute_yield_and_normal=partial(jax_compute_yield_and_normal,
    jax_yield_fun=jax_barlat_yield)
barlat_compute_all_yields_and_normals = partial(compute_all_yields_and_normals,
    jax_compute_yield_and_normal_fun=jax_barlat_compute_yield_and_normal)
barlat_obj_fun = partial(objective, rotations=R_matrices,
    sigma_values=sigma_c_values, ratio_values=ratio_c_values, Y=Y,
    weights=calibration_weights(),
    compute_all_yields_and_normals_fun=barlat_compute_all_yields_and_normals)
opt_obj_and_grad = jit(partial(obj_and_grad, objective_fun=barlat_obj_fun))
opt_params, fun_val, cvg_dict = fmin_l_bfgs_b(opt_obj_and_grad,
    barlat_fit_params)

jit_barlat_compute_all_yields_and_normals = \
    jit(barlat_compute_all_yields_and_normals)
orig_fit_vals = jit_barlat_compute_all_yields_and_normals(barlat_fit_params,
    R_matrices, sigma_c_values)
pred_vals = jit_barlat_compute_all_yields_and_normals(opt_params, R_matrices,
    sigma_c_values)

fig2, ax2 = plt.subplots(figsize=(22, 8), ncols=2)

ax2[0].plot(spec_num, Y * np.ones_like(angles), color="black", label="$Y$",
    zorder=0)
ax2[0].scatter(spec_num, orig_fit_vals[0, :], color="white", edgecolor="red",
    zorder=1, label="Paper fit")
ax2[0].scatter(spec_num, pred_vals[0, :], color="white", edgecolor="blue",
    zorder=2, label="BFGS fit")
ax2[0].set_xlabel("Specimen Number", fontsize=22)
ax2[0].set_ylabel("Predicted Effective Stress", fontsize=22)
ax2[0].set_title("Barlat Yield Predictions", fontsize=22)
ax2[0].set_ylim(0, 700.)
ax2[0].legend(loc="lower right", fontsize=20)

ax2[1].scatter(spec_num, ratio_c_values, color="black", marker="x",
    label="Experiment", zorder=0, s=50)
ax2[1].scatter(spec_num, orig_fit_vals[1, :], color="white", edgecolor="red",
    zorder=1, label="Paper fit")
ax2[1].scatter(spec_num, pred_vals[1, :], color="white", edgecolor="blue",
    zorder=2, label="BFGS fit")
ax2[1].set_xlabel("Specimen Number", fontsize=22)
ax2[1].set_ylabel("Predicted Lankford Ratio", fontsize=22)
ax2[1].set_title("Barlat Lankford Ratio Predictions", fontsize=22)
ax2[1].legend(loc="upper left", fontsize=20)
