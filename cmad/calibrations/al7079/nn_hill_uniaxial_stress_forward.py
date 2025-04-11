import numpy as np
import matplotlib.pyplot as plt
import pickle
from functools import partial
from jax import jit

from cmad.calibrations.al7079.support import (
    slab_data,
    calibration_weights,
    calibrated_barlat_coefficients,
    calibrated_hill_coefficients,
    params_hill_voce,
    params_hybrid_hill_voce
)
from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.models.var_types import get_sym_tensor_from_vector
from cmad.qois.calibration import Calibration
from cmad.solver.nonlinear_solver import newton_solve

from cmad.neural_networks.input_convex_neural_network \
    import InputConvexNeuralNetwork

from cmad.models.effective_stress import(
    beta_initial_guess,
    beta_make_newton_solve,
    J2_effective_stress,
    hybrid_hill_effective_stress,
    hill_effective_stress,
    make_safe_update_fun,
    scaled_effective_stress,
)

from cmad.verification.functions import (
    jax_barlat_yield,
    jax_hill_yield
)


def pack_barlat_params(barlat_coeffs):
    sp_coeffs = barlat_coeffs[:9]  # single prime c coeffs
    dp_coeffs = barlat_coeffs[9:-1]  # double prime c coeffs
    sp_12, sp_13, sp_21, sp_23, sp_31, sp_32, sp_44, sp_55, sp_66 \
        = sp_coeffs
    dp_12, dp_13, dp_21, dp_23, dp_31, dp_32, dp_44, dp_55, dp_66 \
        = dp_coeffs
    a = barlat_coeffs[-1]
    effective_stress_params = {"barlat": {
        "sp_12": sp_12, "sp_13": sp_13,
        "sp_21": sp_21, "sp_23": sp_23,
        "sp_31": sp_31, "sp_32": sp_32,
        "sp_44": sp_44, "sp_55": sp_55, "sp_66": sp_66,
        "dp_12": dp_12, "dp_13": dp_13,
        "dp_21": dp_21, "dp_23": dp_23,
        "dp_31": dp_31, "dp_32": dp_32,
        "dp_44": dp_44, "dp_55": dp_55, "dp_66": dp_66,
        "a": a}
    }

    return effective_stress_params


barlat_coeffs = calibrated_barlat_coefficients()
jit_jax_barlat_yield = jit(partial(jax_barlat_yield,
    barlat_params=barlat_coeffs
))

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

alpha_angles, alpha_sigma_c_values, alpha_ratio_c_values, R_alphas = \
    slab_data("alpha")
beta_angles, beta_sigma_c_values, beta_ratio_c_values, R_betas = \
    slab_data("beta")
gamma_angles, gamma_sigma_c_values, gamma_ratio_c_values, R_gammas = \
    slab_data("gamma")

Y = alpha_sigma_c_values[0]
scaled_hybrid_effective_stress = jit(partial(scaled_effective_stress,
    effective_stress_fun=hybrid_effective_stress,
    update_fun=beta_make_newton_solve(hybrid_effective_stress, Y)
))

plt.close("all")


hill_fit_params = calibrated_hill_coefficients()
p_elastic = np.array([70.22857142857143e3, 0.33396551724137924]) # not being optimized
E = p_elastic[0]
nu = p_elastic[1]
p_hill = np.r_[np.array(Y), hill_fit_params]
p_voce = np.array([200., 20.]) # need to fiddle with these

def_type = DefType.UNIAXIAL_STRESS
params = params_hill_voce(p_elastic, p_hill, p_voce)

num_steps = 500
ndims = def_type_ndims(def_type)
I = np.eye(ndims)
F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
epsilon_11 = np.linspace(0, 5. * 0.02, num_steps + 1)
# models assume this is for the uniaxial stress axis
F[0, 0, :] += epsilon_11

params.values["plastic"]["effective stress"] = hybrid_model_params["plastic"]["effective stress"]
models = [SmallElasticPlastic(params, def_type, uniaxial_stress_idx=1,
    effective_stress_fun=scaled_hybrid_effective_stress)]

#models = [SmallElasticPlastic(params, def_type, uniaxial_stress_idx=1),
#    SmallRateElasticPlastic(params, def_type, uniaxial_stress_idx=1)]

fig, ax = plt.subplots(figsize=(22, 8), ncols=2)
Rmats = R_alphas + R_betas + R_gammas
for rr, Rmat in enumerate(Rmats):
    for model in models:

        params.set_rotation_matrix(Rmat)

        num_residuals = model.num_residuals
        xi_at_step = [[None] * num_residuals for ii in range(num_steps + 1)]
        model.set_xi_to_init_vals()
        model.store_xi(xi_at_step, model.xi_prev(), 0)

        # both in global (experimental) coordinates
        cauchy = np.zeros((3, 3, num_steps + 1))
        plastic_strain_zz = np.zeros(num_steps + 1)
        plastic_strain_xx = np.zeros(num_steps + 1)
        strain_ratio = np.zeros(num_steps + 1)

        for step in range(1, num_steps + 1):
            u = [F[:, :, step]]
            u_prev = [F[:, :, step - 1]]
            model.gather_global(u, u_prev)

            # need a looser tolerance than conventional models
            newton_solve(model, abs_tol=1e-13, rel_tol=1e-13,
                max_ls_evals=5)
            model.store_xi(xi_at_step, model.xi(), step)

            model.seed_none()
            model.evaluate_cauchy()
            cauchy[:, :, step] = model.Sigma().copy()
            off_axis_elastic_strain = -cauchy[1, 1, step] * nu / E
            strain_ratio[step] = \
                (xi_at_step[step][2][1] - 1. - off_axis_elastic_strain) \
                / (xi_at_step[step][2][0] - 1. - off_axis_elastic_strain)
            plastic_strain_zz[step] = xi_at_step[step][2][1] - 1. - off_axis_elastic_strain
            plastic_strain_xx[step] = xi_at_step[step][2][0] - 1. - off_axis_elastic_strain

            model.advance_xi()

        off_axis_stress_norm = \
            np.abs(np.linalg.norm(cauchy) - np.linalg.norm(cauchy[1, 1, :]))
        assert off_axis_stress_norm < 1e-11

        ax[0].plot(epsilon_11[:num_steps + 1], cauchy[1, 1, :], zorder=0)
        ax[1].plot(plastic_strain_xx[1:], plastic_strain_zz[1:], zorder=0)
        print(f"{rr} ratio = {strain_ratio[-1]}")

    ax[0].set_xlabel(r"$\varepsilon_{yy}$", fontsize=22)
    ax[0].set_ylabel("Uniaxial Stress", fontsize=22)
    ax[0].set_title("Uniaxial Stress vs Strain", fontsize=22)
    # ax[0].legend(loc="best", fontsize=20)
    ax[1].set_xlabel(r"$\varepsilon^p_{xx}$", fontsize=22)
    ax[1].set_ylabel(r"$\varepsilon^p_{zz}$", fontsize=22)
    ax[1].set_title("Off-axis Plastic Strain", fontsize=22)
    # ax[1].legend(loc="best", fontsize=20)
