import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

from functools import partial
from jax import jit, grad, tree_map, jacrev
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import fmin_l_bfgs_b

from cmad.neural_networks.input_convex_neural_network \
    import InputConvexNeuralNetwork
from cmad.parameters.parameters import Parameters
from cmad.verification.functions import (hill_yield, hill_yield_normal,
    jax_hill_yield, jax_hill_yield_normal)

# branch import
from cmad.models.effective_stress import (hill_effective_stress,
    hybrid_hill_effective_stress)

# local import
from al7079 import (slab_data, calibration_weights,
    calibrated_hill_coefficients, params_hill_voce,
    params_icnn_hybrid_hill_voce)


def extract_material_dev_cauchy_vector(sigma_c, rotation):
    unit_sigma = np.array([[0., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.]])
    sigma = sigma_c * unit_sigma
    sigma_mat = rotation.T @ sigma @ rotation
    dev_sigma_mat = sigma_mat - np.trace(sigma_mat) / 3. * np.eye(3)
    vec_dev_sigma_mat = np.array([dev_sigma_mat[0, 0], dev_sigma_mat[1, 1],
        dev_sigma_mat[2, 2], dev_sigma_mat[0, 1], dev_sigma_mat[0, 2],
        dev_sigma_mat[1, 2]])

    return vec_dev_sigma_mat


def make_input_scaler(sigma_c_values, rotations, Scaler=MinMaxScaler):
    input_scaler = Scaler()
    #input_scaler = Scaler(feature_range=(0.5, 1))

    unscaled_features = np.vstack([
         extract_material_dev_cauchy_vector(sigma_c, rotation)
         for sigma_c, rotation in zip(sigma_c_values, rotations)
    ])

    input_scaler.fit(unscaled_features)
    #scaled_features = input_scaler.fit_transform(unscaled_features)
    #scaled_features2 = unscaled_features * input_scaler.scale_ \
    #    + input_scaler.min_

    return input_scaler

def make_output_scaler(sigma_c_values, Scaler=MinMaxScaler):
    output_scaler = Scaler()
    output_scaler.fit(sigma_c_values.reshape(-1, 1))

    #scaled_targets = output_scaler.fit_transform(sigma_c_values.reshape(-1, 1))
    #unscaled_targets = (scaled_targets - output_scaler.min_) \
    #    / output_scaler.scale_

    return output_scaler


class EffectiveStressObjective():
    def __init__(self, parameters, sigma_c_values, ratio_c_values, rotations,
            weights, yield_and_normal_fun):

        self.parameters = parameters
        self._sigma_c_values = sigma_c_values
        self._ratio_c_values = ratio_c_values
        self._rotations = rotations
        assert len(weights) == 2
        self._weights = weights # w_sigma, w_r
        self._qois = jit(yield_and_normal_fun)
        self._qoi_derivs = jit(jacrev(yield_and_normal_fun))

    def evaluate(self, flat_active_values):
        self.parameters.set_active_values_from_flat(flat_active_values)

        return self._evaluate()

    def _evaluate(self):
        w_sigma, w_ratio = self._weights
        param_values = self.parameters.values
        get_active_jac = self.parameters.scalar_active_params_jacobian
        Y = float(param_values["plastic"]["flow stress"]["initial yield"]["Y"])

        zip_data = zip(self._sigma_c_values, self._ratio_c_values,
            self._rotations)

        J = 0.
        dJ_dp = np.zeros((1, self.parameters.num_active_params))

        for sigma_c, ratio_c, R in zip_data:

            phi, ratio = self._qois(param_values, sigma_c, R)
            res_sigma = (float(phi) / Y - 1.)
            res_ratio = (float(ratio) / ratio_c - 1.)
            J += 0.5 * (w_sigma * res_sigma**2 + w_ratio * res_ratio**2)

            dphi, dratio = (np.asarray(get_active_jac(u)) for u in \
                self._qoi_derivs(param_values, sigma_c, R))
            dJ_dp += w_sigma * res_sigma / Y * dphi \
                + w_ratio * res_ratio / ratio_c * dratio


        dJ_dp = self.parameters.transform_grad(dJ_dp.squeeze())

        return J, dJ_dp


def calibrate_hill(params, train_data, val_data, weights,
        yield_and_normal_fun):
    train_sigma_c_values, train_ratio_c_values, train_R_matrices = train_data
    val_sigma_c_values, val_R_matrices = val_data

    obj = EffectiveStressObjective(params, train_sigma_c_values,
        train_ratio_c_values, train_R_matrices, weights,
        yield_and_normal_fun)

    scaled_values = params.flat_active_values(True).copy()
    opt_params, fun_val, cvg_dict = fmin_l_bfgs_b(obj.evaluate, scaled_values,
        bounds=params.opt_bounds)

    print(fun_val)
    print(cvg_dict)

    num_val_idx = len(val_sigma_c_values)
    val_phi = np.zeros(num_val_idx)
    val_ratio = np.zeros(num_val_idx)

    params.set_active_values_from_flat(opt_params)
    unscaled_opt_params = params.flat_active_values()

    for idx, (sigma_c, R) in enumerate(zip(val_sigma_c_values, val_R_matrices)):
        phi, ratio = yield_and_normal_fun(params.values, sigma_c, R)
        val_phi[idx] = phi
        val_ratio[idx] = ratio

    return unscaled_opt_params, val_phi, val_ratio


def compute_yield_and_normal(params, sigma_c, R, effective_stress_fun):
    # update parameters from opt_params
    unit_sigma = jnp.array([[0., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.]])
    sigma = sigma_c * unit_sigma
    sigma_mat = R.T @ sigma @ R
    phi = effective_stress_fun(sigma_mat, params["plastic"])
    normal_mat = grad(effective_stress_fun)(sigma_mat, params["plastic"])
    normal = R @ normal_mat @ R.T
    ratio = normal[2, 2] / normal[0, 0]
    return phi, ratio


plt.close("all")

alpha_angles, alpha_sigma_c_values, alpha_ratio_c_values, R_alphas = \
    slab_data("alpha")
beta_angles, beta_sigma_c_values, beta_ratio_c_values, R_betas = \
    slab_data("beta")
gamma_angles, gamma_sigma_c_values, gamma_ratio_c_values, R_gammas = \
    slab_data("gamma")

fit_params = calibrated_hill_coefficients()
Y = alpha_sigma_c_values[0]
weights = calibration_weights()

p_elastic = np.array([70.22857142857143e3, 0.33396551724137924]) # not being optimized
p_hill = np.r_[np.array(Y), fit_params]
p_voce = np.array([200., 20.]) # not being optimized now
params = params_hill_voce(p_elastic, p_hill, p_voce)

angles = np.r_[alpha_angles, beta_angles, gamma_angles]
sigma_c_values = np.r_[alpha_sigma_c_values, beta_sigma_c_values, \
    gamma_sigma_c_values]
ratio_c_values = np.r_[alpha_ratio_c_values, beta_ratio_c_values, \
    gamma_ratio_c_values]
R_matrices = R_alphas + R_betas + R_gammas

num_angles = len(angles)
hold_out_idx = np.where(angles == 60. * np.pi / 180.)[0]
train_idx = np.setdiff1d(np.arange(num_angles), hold_out_idx)

val_idx = np.arange(num_angles)
train_idx = np.arange(num_angles)

val_sigma_c_values = np.array([sigma_c_values[ii] for ii in val_idx])
val_ratio_c_values = np.array([ratio_c_values[ii] for ii in val_idx])
val_R_matrices = np.array([R_matrices[ii] for ii in val_idx])
#val_data = (val_sigma_c_values, val_R_matrices)


# training with all the data
#train_sigma_c_values = np.array([sigma_c_values[ii] for ii in train_idx])
#train_ratio_c_values = np.array([ratio_c_values[ii] for ii in train_idx])
#train_R_matrices = np.array([R_matrices[ii] for ii in train_idx])
#train_data = (train_sigma_c_values, train_ratio_c_values, train_R_matrices)

hold_out_idx = np.where(angles == 60. * np.pi / 180.)[0]
train_idx = np.setdiff1d(np.arange(num_angles), hold_out_idx)
train_sigma_c_values = np.array([sigma_c_values[ii] for ii in train_idx])
train_ratio_c_values = np.array([ratio_c_values[ii] for ii in train_idx])
train_R_matrices = np.array([R_matrices[ii] for ii in train_idx])
#train_data = (train_sigma_c_values, train_ratio_c_values, train_R_matrices)

data_type = "tension"
#data_type = "compression"
#data_type = "both"

if data_type == "tension" or data_type == "compression":
    if data_type == "tension":
        sign = 1.
    else:
        sign = -1.
    total_val_sigma_c_values = sign * val_sigma_c_values
    total_train_sigma_c_values = sign * train_sigma_c_values
    train_data = (total_train_sigma_c_values, train_ratio_c_values,
        train_R_matrices)
    val_data = (total_val_sigma_c_values, val_R_matrices)
elif data_type == "both":
    total_val_sigma_c_values = np.r_[-val_sigma_c_values, val_sigma_c_values]
    total_train_sigma_c_values = np.r_[-train_sigma_c_values, train_sigma_c_values]
    train_data = (total_train_sigma_c_values, np.r_[train_ratio_c_values,
        train_ratio_c_values],
        np.r_[train_R_matrices, train_R_matrices])
    val_data = (total_val_sigma_c_values,
        np.r_[val_R_matrices, val_R_matrices])

input_scaler = make_input_scaler(total_train_sigma_c_values,
    train_R_matrices)
output_scaler = make_output_scaler(total_train_sigma_c_values)

paper_compute_yield_and_normal = partial(compute_yield_and_normal,
    effective_stress_fun=hill_effective_stress)
paper_params, paper_phi, paper_ratio = calibrate_hill(params, train_data,
    val_data, weights, paper_compute_yield_and_normal)

layer_widths = [6, 6, 1]
phi_disc_nn = InputConvexNeuralNetwork(layer_widths,
    input_scaler=input_scaler, output_scaler=output_scaler, seed=40)
hybrid_icnn_params = {"x params": phi_disc_nn.x_params,
    "z params": phi_disc_nn.z_params}
hybrid_params = params_icnn_hybrid_hill_voce(p_elastic, p_hill, p_voce,
    hybrid_icnn_params)
hybrid_yield_and_normal_fun = partial(compute_yield_and_normal,
    effective_stress_fun=partial(hybrid_hill_effective_stress,
    nn_fun=phi_disc_nn.evaluate))

held_out_params, held_out_phi, held_out_ratio = \
    calibrate_hill(hybrid_params, train_data, val_data, weights,
    hybrid_yield_and_normal_fun)

trad_hold_out_obj = 0.5 * (weights[0] * np.sum((paper_phi[hold_out_idx] / Y \
    - 1.)**2) + weights[1] \
    * np.sum((paper_ratio[hold_out_idx] / ratio_c_values[hold_out_idx] \
    - 1.)**2))

nn_hold_out_obj = 0.5 * (weights[0] * np.sum((held_out_phi[hold_out_idx] / Y \
    - 1.)**2) + weights[1] \
    * np.sum((held_out_ratio[hold_out_idx] / ratio_c_values[hold_out_idx] \
    - 1.)**2))

spec_num = np.arange(num_angles)
fig, ax = plt.subplots(figsize=(22, 8), ncols=2)

ax[0].plot(spec_num, Y * np.ones_like(spec_num), color="black", label="$Y$",
    zorder=0)
ax[0].scatter(hold_out_idx, Y * np.ones_like(hold_out_idx), color="green",
    s=100, label="Validation Data", zorder=1)
ax[0].scatter(spec_num, paper_phi, color="white", edgecolor="red",
    zorder=1, label="Hill")
ax[0].scatter(spec_num, held_out_phi, color="white", edgecolor="blue",
    zorder=2, label="Neural Network")
ax[0].set_xlabel("Specimen Number", fontsize=22)
ax[0].set_ylabel("Predicted Effective Stress (MPa)", fontsize=22)
ax[0].set_title("Yield Predictions", fontsize=22)
ax[0].legend(loc="best", fontsize=20)

ax[1].scatter(spec_num, val_ratio_c_values, color="black", marker="x",
    label="Experiment", zorder=1, s=50)
ax[1].scatter(spec_num[hold_out_idx], val_ratio_c_values[hold_out_idx],
    color="green", marker="x", label="Validation Data", zorder=2, s=50)
ax[1].scatter(spec_num, paper_ratio, color="white", edgecolor="red",
    zorder=0, label="Hill")
ax[1].scatter(spec_num, held_out_ratio, color="white", edgecolor="blue",
    zorder=2, label="Neural Network")
ax[1].set_xlabel("Specimen Number", fontsize=22)
ax[1].set_ylabel("Predicted Lankford Ratio", fontsize=22)
ax[1].set_title("Lankford Ratio Predictions", fontsize=22)
ax[1].legend(loc="best", fontsize=20)
