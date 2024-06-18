import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from cmad.calibrations.al7079.support import (
    slab_data,
    calibration_weights,
    calibrated_hill_coefficients,
    params_hill_voce,
    params_hybrid_hill_voce
)
from cmad.models.deformation_types import DefType
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.solver.nonlinear_solver import newton_solve
from cmad.verification.functions import hill_yield, hill_yield_normal
from cmad.verification.solutions import (
    compute_elastic_fields,
    compute_plastic_fields
)


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

stress_mask = np.array([[0., 0., 0.],
    [0., 1., 0.],
    [0., 0., 0.]])
max_alpha = 0.1
num_elastic_steps = 100
num_plastic_steps = 100

num_steps = num_elastic_steps + num_plastic_steps

E, nu = p_elastic
S, D = p_voce
isotropic_param_values = np.array([E, nu, Y, S, D])

pt_hill_yield = partial(hill_yield, hill_params=fit_params)
pt_hill_yield_normal = partial(hill_yield_normal, hill_params=fit_params)
mat_stress_mask = R_alphas[1].T @ stress_mask @ R_alphas[1]


mat_plastic_stress, mat_plastic_strain, alpha = compute_plastic_fields(mat_stress_mask, 
    pt_hill_yield, pt_hill_yield_normal, isotropic_param_values, max_alpha,
    num_plastic_steps)

mat_stress_tensor_at_yield = mat_plastic_stress[:, :, 0]
min_elastic_stress_scale = 0.1
max_elastic_stress_scale = 0.99

mat_elastic_stress, mat_elastic_strain = \
    compute_elastic_fields(mat_stress_tensor_at_yield,
    min_elastic_stress_scale, max_elastic_stress_scale, p_elastic,
    num_elastic_steps)

mat_stress = np.dstack([mat_elastic_stress, mat_plastic_stress])
mat_strain = np.dstack([mat_elastic_strain, mat_plastic_strain])

stress = np.dstack([R_alphas[1] @ mat_stress[:, :, ii] @ R_alphas[1].T \
    for ii in range(num_steps)])
strain = np.dstack([R_alphas[1] @ mat_strain[:, :, ii] @ R_alphas[1].T \
    for ii in range(num_steps)])

ndims = 3
I = np.eye(ndims)
F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
F[:, :, 1:] += strain[:ndims, :ndims, :]

model = SmallElasticPlastic(params, DefType.FULL_3D)
#model = SmallRateElasticPlastic(params, DefType.FULL_3D)

xi_at_step = [[None, None] for ii in range(num_steps + 1)]
model.set_xi_to_init_vals()
model.store_xi(xi_at_step, model.xi_prev(), 0)
cauchy = np.zeros((3, 3, num_steps + 1))

for step in range(1, num_steps + 1):

    u = [R_alphas[1].T @ F[:, :, step] @ R_alphas[1]]
    u_prev = [R_alphas[1].T @ F[:, :, step - 1] @ R_alphas[1]]
    model.gather_global(u, u_prev)

    newton_solve(model)
    model.store_xi(xi_at_step, model.xi(), step)

    model.seed_none()
    model.evaluate_cauchy()
    cauchy[:, :, step] = R_alphas[1] @ model.Sigma().copy() @ R_alphas[1].T

    model.advance_xi()

analytical_sigma_yy = stress[1, 1, :]
numerical_sigma_yy = cauchy[1, 1, 1:]
diff_tol = 1e-9
assert np.linalg.norm(analytical_sigma_yy - numerical_sigma_yy) < diff_tol

fig, ax = plt.subplots(figsize=(11, 8))
ax.plot(strain[1, 1, :], cauchy[1, 1, 1:], marker="o")
ax.set_xlabel("$\\varepsilon_{yy}$", fontsize=22)
ax.set_ylabel("$\\sigma{yy}$ (MPa)", fontsize=22)
ax.set_title("Uniaxial Stress in Experiment Coordinates ($x, y, z$)", fontsize=22)
