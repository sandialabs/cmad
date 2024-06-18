import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from scipy.optimize import fmin_l_bfgs_b

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
from cmad.qois.calibration import UniaxialCalibration
from cmad.solver.nonlinear_solver import newton_solve
from cmad.models.var_types import get_sym_tensor_from_vector
from cmad.objectives.objective import Objective


def compute_cauchy(model, F, Rmat):

    model.parameters.values["rotation matrix"] = Rmat

    xi_at_step = [[None, None, None] for ii in range(num_steps + 1)]
    model.set_xi_to_init_vals()
    model.store_xi(xi_at_step, model.xi_prev(), 0)
    cauchy = np.zeros((3, 3, num_steps + 1))

    for step in range(1, num_steps + 1):
        u = [F[:, :, step]]
        u_prev = [F[:, :, step - 1]]
        model.gather_global(u, u_prev)

        newton_solve(model)
        model.store_xi(xi_at_step, model.xi(), step)

        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()

        model.advance_xi()

    return jnp.stack([
        cauchy[1, 1, :],
        np.array([xi_at_step[ii][2][0] - 1. for ii in range(num_steps + 1)]),
        np.array([xi_at_step[ii][2][1] - 1. for ii in range(num_steps + 1)])],
        axis=0)


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
p_voce = np.array([1., 200.]) # need to fiddle with these

def_type = DefType.UNIAXIAL_STRESS
params_true = params_hill_voce(p_elastic, p_hill, p_voce)
uniaxial_stress_idx = 1
stretch_var_idx = 2
model_true = SmallElasticPlastic(params_true, def_type,
    uniaxial_stress_idx=uniaxial_stress_idx)

num_steps = 50
ndims = def_type_ndims(def_type)
I = np.eye(ndims)
F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
epsilon_11 = np.linspace(0, 5. * 0.02, num_steps + 1)
F[0, 0, :] += epsilon_11

Rmats = R_alphas + R_betas + R_gammas
data = np.stack([compute_cauchy(model_true, F, Rmat) for Rmat in Rmats], axis=0)

p_hill_J2_equivalent = np.r_[np.array(Y), np.ones(6)]
params = params_hill_voce(p_elastic, p_hill_J2_equivalent, p_voce)
model = SmallElasticPlastic(params, def_type, uniaxial_stress_idx=1)
initial_guess = model.parameters.flat_active_values(True).copy()
num_active_params = model.parameters.num_active_params
opt_bounds = params.opt_bounds

weights_qoi = np.array([2e-3, 2e1, 2e1])
weights_time = np.ones(num_steps + 1)
weights_time[0:10] = 0.
qoi = UniaxialCalibration(model, F, data[0], weights_qoi,
    uniaxial_stress_idx, stretch_var_idx)
objective = Objective(qoi, sensitivity_type="adjoint gradient", weights=weights_time)

def multiobjective(Rmats, data):
    def mobjective(varlist):
        J_total = 0.
        grad_total = np.zeros(varlist.shape)
        for j,Rmat in enumerate(Rmats):
            objective._qoi.model().parameters.set_rotation_matrix(Rmat)
            objective._qoi._data = data[j]

            J, grad = objective.evaluate(varlist)

            J_total += J
            grad_total += grad 
        return J_total, grad_total
    return mobjective

opt_params, fun_vals, cvg_dict = fmin_l_bfgs_b(
    multiobjective(Rmats,data), params_true.flat_active_values(True)+.1, 
    bounds=opt_bounds, iprint=1, maxiter=400)

model.parameters.set_active_values_from_flat(opt_params)
unscaled_opt_params = model.parameters.flat_active_values()
param_diff = np.linalg.norm(unscaled_opt_params - fit_params)
