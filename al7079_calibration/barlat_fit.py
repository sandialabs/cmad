import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
import time

from cmad.verification.functions import jax_barlat_yield
from functools import partial
from jax import jit, grad
from scipy.optimize import fmin_l_bfgs_b

# local import
from al7079 import (slab_data, calibration_weights,
    calibrated_barlat_coefficients)


def jax_compute_yield_and_normal(R, sigma_c, params):
    unit_sigma = jnp.array([[0., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.]])
    sigma = sigma_c * unit_sigma
    sigma_mat = R.T @ sigma @ R
    phi = jax_barlat_yield(sigma_mat, params)
    normal_fun = grad(jax_barlat_yield)
    normal_mat = normal_fun(sigma_mat, params)
    normal = R @ normal_mat @ R.T
    ratio = normal[2, 2] / normal[0, 0]
    return jnp.atleast_2d([phi, ratio]).T


def compute_all_yields_and_normals(params, rotations, sigma_values):
    vals = jnp.hstack([jax_compute_yield_and_normal(R, sigma_c, params)
        for R, sigma_c in zip(rotations, sigma_values)])
    return vals


def objective(params, rotations, sigma_values, ratio_values, Y, weights):
    w_sigma, w_ratio = weights
    vals = compute_all_yields_and_normals(params, rotations, sigma_values)
    obj_sigma = w_sigma * jnp.sum(jnp.array([(sigma / Y - 1.)**2 \
        for sigma in vals[0, :]]))
    obj_ratio = w_ratio * jnp.sum(jnp.array([(pred_ratio / meas_ratio - 1.)**2 \
        for pred_ratio, meas_ratio in zip(vals[1, :], ratio_values)]))
    return obj_sigma + obj_ratio

def obj_and_grad(params, rotations, sigma_values, ratio_values, Y, weights):
    obj_grad = grad(objective)
    J = objective(params, rotations, sigma_values, ratio_values, Y, weights)
    dJ_dp = obj_grad(params, rotations, sigma_values, ratio_values, Y, weights)

    return J, dJ_dp


plt.close("all")

alpha_angles, alpha_sigma_c_values, alpha_ratio_c_values, R_alphas = \
    slab_data("alpha")
beta_angles, beta_sigma_c_values, beta_ratio_c_values, R_betas = \
    slab_data("beta")
gamma_angles, gamma_sigma_c_values, gamma_ratio_c_values, R_gammas = \
    slab_data("gamma")

fit_params = calibrated_barlat_coefficients()
Y = alpha_sigma_c_values[0]

angles = np.r_[alpha_angles, beta_angles, gamma_angles]
sigma_c_values = np.r_[alpha_sigma_c_values, beta_sigma_c_values, \
    gamma_sigma_c_values]
ratio_c_values = np.r_[alpha_ratio_c_values, beta_ratio_c_values, \
    gamma_ratio_c_values]
R_matrices = R_alphas + R_betas + R_gammas

#partial_objective = partial(objective, rotations=R_matrices,
#    sigma_values=sigma_c_values, ratio_values=ratio_c_values, Y=Y,
#    weights=calibration_weights())
#obj = jit(partial_objective)
#obj_grad = jit(grad(partial_objective))
#J = obj(fit_params)
#dJ_dp = obj_grad(fit_params)

#start_time = time.time()
opt_obj_and_grad = jit(partial(obj_and_grad, rotations=R_matrices,
    sigma_values=sigma_c_values, ratio_values=ratio_c_values, Y=Y,
    weights=calibration_weights()))
#opt_obj_and_grad(fit_params)
#end_time = time.time()

#jit_time = end_time - start_time
#print(f"jit compilation time = {jit_time:.3e} seconds")

#start_time = time.time()
opt_params, fun_val, cvg_dict = fmin_l_bfgs_b(opt_obj_and_grad, fit_params)
#end_time = time.time()

#opt_time = end_time - start_time
#print(f"optimization time = {opt_time:.3e} seconds")

jit_compute_all_yields_and_normals = jit(compute_all_yields_and_normals)
orig_fit_vals = jit_compute_all_yields_and_normals(fit_params, R_matrices,
    sigma_c_values)

pred_vals = compute_all_yields_and_normals(opt_params, R_matrices,
    sigma_c_values)

spec_num = np.arange(len(angles))
fig, ax = plt.subplots(figsize=(22, 8), ncols=2)

ax[0].plot(spec_num, Y * np.ones_like(angles), color="black", label="$Y$",
    zorder=0)
ax[0].scatter(spec_num, orig_fit_vals[0, :], color="white", edgecolor="red",
    zorder=1, label="Paper fit")
ax[0].scatter(spec_num, pred_vals[0, :], color="white", edgecolor="blue",
    zorder=2, label="BFGS fit")
ax[0].set_xlabel("Specimen Number", fontsize=22)
ax[0].set_ylabel("Predicted Effective Stress", fontsize=22)
ax[0].set_title("Barlat Yield Predictions", fontsize=22)
ax[0].set_ylim(0, 700.)
ax[0].legend(loc="lower right", fontsize=20)

ax[1].scatter(spec_num, ratio_c_values, color="black", marker="x",
    label="Experiment", zorder=0, s=50)
ax[1].scatter(spec_num, orig_fit_vals[1, :], color="white", edgecolor="red",
    zorder=1, label="Paper fit")
ax[1].scatter(spec_num, pred_vals[1, :], color="white", edgecolor="blue",
    zorder=2, label="BFGS fit")
ax[1].set_xlabel("Specimen Number", fontsize=22)
ax[1].set_ylabel("Predicted Lankford Ratio", fontsize=22)
ax[1].set_title("Barlat Lankford Ratio Predictions", fontsize=22)
ax[1].legend(loc="upper left", fontsize=20)
