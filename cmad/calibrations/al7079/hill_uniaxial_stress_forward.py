import numpy as np
import matplotlib.pyplot as plt

from cmad.calibrations.al7079.support import (
    slab_data,
    calibration_weights,
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


# need to work out how compute global strain for each model

#def compute_strain_small_elastic_plastic(F, Rmat, xi, step):
#    strain = np.zeros((3, 3))
#    axial_strain = F[0, 0, step]
#    strain[0, 0] = xi[step][2][0] - 1.
#    strain[1, 1] = axial_strain - 1.
#    strain[2, 2] = xi[step][2][1] - 1.
#
#    material_plastic_strain = get_sym_tensor_from_vector(xi[step][0], 3)
#    global_plastic_strain = Rmat @ material_plastic_strain @ Rmat.T
#    strain[0, 1] = global_plastic_strain[0, 1]
#    strain[0, 2] = global_plastic_strain[0, 2]
#    strain[1, 2] = global_plastic_strain[1, 2]
#    strain[1, 0] = strain[0, 1]
#    strain[2, 0] = strain[0, 2]
#    strain[2, 1] = strain[1, 2]
#
#    return strain


#def compute_strain_small_rate_elastic_plastic(F, Rmat, xi, step):
#    strain = np.zeros((3, 3))
#    axial_strain = F[0, 0, step]
#    strain[0, 0] = xi[step][2][0] - 1.
#    strain[1, 1] = axial_strain - 1.
#    strain[2, 2] = xi[step][2][1] - 1.
#
#    material_plastic_strain = np.zeros((3, 3))
#    material_off_axis_plastic_strain_vec = np.zeros(3)
#    # not finished
#    for ii in range(step):
#        material_plastic_strain += \
#            get_sym_tensor_from_vector(xi[step][3], 3)
#    global_plastic_strain = Rmat @ material_plastic_strain @ Rmat.T
#    strain[0, 1] = global_plastic_strain[0, 1]
#    strain[0, 2] = global_plastic_strain[0, 2]
#    strain[1, 2] = global_plastic_strain[1, 2]
#    strain[1, 0] = strain[0, 1]
#    strain[2, 0] = strain[0, 2]
#    strain[2, 1] = strain[1, 2]

#    return strain


plt.close("all")

alpha_angles, alpha_sigma_c_values, alpha_ratio_c_values, R_alphas = \
    slab_data("alpha")
beta_angles, beta_sigma_c_values, beta_ratio_c_values, R_betas = \
    slab_data("beta")
gamma_angles, gamma_sigma_c_values, gamma_ratio_c_values, R_gammas = \
    slab_data("gamma")

fit_params = calibrated_hill_coefficients()
Y = alpha_sigma_c_values[0]

p_elastic = np.array([70.22857142857143e3, 0.33396551724137924]) # not being optimized
E = p_elastic[0]
p_hill = np.r_[np.array(Y), fit_params]
p_voce = np.array([1., 20.]) # need to fiddle with these

def_type = DefType.UNIAXIAL_STRESS
params = params_hill_voce(p_elastic, p_hill, p_voce)

num_steps = 50
ndims = def_type_ndims(def_type)
I = np.eye(ndims)
F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
epsilon_11 = np.linspace(0, 5. * 0.02, num_steps + 1)
# models assume this is for the uniaxial stress axis
F[0, 0, :] += epsilon_11

models = [SmallElasticPlastic(params, def_type, uniaxial_stress_idx=1),
    SmallRateElasticPlastic(params, def_type, uniaxial_stress_idx=1)]
#compute_strain_funs = [compute_strain_small_elastic_plastic,
#    compute_strain_small_rate_elastic_plastic]

Rmats = R_alphas + R_betas + R_gammas

for rr, Rmat in enumerate(Rmats):
    for model in models:
    #for model, compute_strain_fun in zip(models, compute_strain_funs):

        params.set_rotation_matrix(Rmat)

        num_residuals = model.num_residuals
        nones_list = [None] * num_residuals
        xi_at_step = [nones_list for ii in range(num_steps + 1)]
        model.set_xi_to_init_vals()
        model.store_xi(xi_at_step, model.xi_prev(), 0)

        # both in global (experimental) coordinates
        cauchy = np.zeros((3, 3, num_steps + 1))
        #strain = np.zeros((3, 3, num_steps + 1))

        for step in range(1, num_steps + 1):
            u = [F[:, :, step]]
            u_prev = [F[:, :, step - 1]]
            model.gather_global(u, u_prev)

            newton_solve(model)
            model.store_xi(xi_at_step, model.xi(), step)

            model.seed_none()
            model.evaluate_cauchy()
            cauchy[:, :, step] = model.Sigma().copy()
            #strain[:, :, step] = compute_strain_fun(F, Rmat, xi_at_step, step)

            model.advance_xi()
            #evals, evecs = np.linalg.eig(strain)
            #print(f"strain = {strain}")
            #print(f"evecs = {evecs}")

        #print(f"iter {rr}: full cauchy norm = {np.linalg.norm(cauchy)}")
        #print(f"iter {rr}: cauchy_yy norm = {np.linalg.norm(cauchy[1, 1, :])}")
        #print(f"iter {rr}: off axis stress norm = {off_axis_stress_norm}\n")
        off_axis_stress_norm = \
            np.abs(np.linalg.norm(cauchy) - np.linalg.norm(cauchy[1, 1, :]))
        assert off_axis_stress_norm < 1e-12

    fig, ax = plt.subplots(figsize=(22, 8), ncols=2)

    ax[0].plot(epsilon_11, cauchy[1, 1, :], color="black", label=r"$\sigma_{yy}$",
        zorder=0)
    ax[0].set_xlabel(r"$\varepsilon_{yy}$", fontsize=22)
    ax[0].set_ylabel("Uniaxial Stress", fontsize=22)
    ax[0].set_title("Uniaxial Stress vs Strain", fontsize=22)
    ax[0].legend(loc="best", fontsize=20)
    # update to plot off-axis strains in ax[1]
