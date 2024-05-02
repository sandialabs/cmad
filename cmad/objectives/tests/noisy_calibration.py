import numpy as np
import matplotlib.pyplot as plt

from jax import tree_map

from scipy.optimize import fmin_l_bfgs_b

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.neural_networks.simple_neural_network import SimpleNeuralNetwork
from cmad.objectives.objective import Objective
from cmad.parameters.parameters import Parameters
from cmad.qois.calibration import Calibration
from cmad.solver.nonlinear_solver import newton_solve


def newton_optimization(initial_guess, objective, grad_tol=1e-4):

    x = initial_guess.copy()
    grad_norm = 1e+10

    ii = 0
    while True:
        J, grad, hessian = objective.evaluate(x)

        grad_norm = np.linalg.norm(grad)
        print(f"iter {ii}: J = {J:.2e}, ||g|| = {grad_norm:.2e},"
              f" det(H) = {np.linalg.det(hessian):.2e},"
              f" cond(H) = {np.linalg.cond(hessian):.2e}")

        if (grad_norm < grad_tol):
            break

        x -= np.linalg.solve(hessian, grad)
        J, grad, hessian = objective.evaluate(x)
        ii += 1

    return x


def create_J2_parameters():
    E = 70e3
    nu = 0.3
    Y = 200.
    K = 0e3
    S = 200.
    D = 20.

    elastic_params = {"E": E, "nu": nu}
    J2_effective_stress_params = {"J2": 0.}
    initial_yield_params = {"Y": Y}
    voce_params = {"S": S, "D": D}
    hardening_params = {"voce": voce_params}

    Y_log_scale = np.array([200.])
    S_log_scale = np.array([200.])
    D_log_scale = np.array([20.])

    J2_values = {
        "rotation matrix": np.eye(3),
        "elastic": elastic_params,
        "plastic": {
            "effective stress": J2_effective_stress_params,
            "flow stress": {
                "initial yield": initial_yield_params,
                "hardening": hardening_params}}}

    J2_active_flags = J2_values.copy()
    J2_active_flags = tree_map(lambda a: False, J2_active_flags)
    J2_active_flags["plastic"]["flow stress"] = tree_map(
        lambda x: True, J2_active_flags["plastic"]["flow stress"])

    J2_transforms = J2_values.copy()
    J2_transforms = tree_map(lambda a: None, J2_transforms)
    J2_flow_stress_transforms = J2_transforms["plastic"]["flow stress"]
    J2_flow_stress_transforms["initial yield"]["Y"] = Y_log_scale
    J2_flow_stress_transforms["hardening"]["voce"]["S"] = S_log_scale
    J2_flow_stress_transforms["hardening"]["voce"]["D"] = D_log_scale

    J2_parameters = \
        Parameters(J2_values, J2_active_flags, J2_transforms)

    return J2_parameters


def create_J2_parameters_nn(nn_params):
    E = 70e3
    nu = 0.3
    Y = 250.

    elastic_params = {"E": E, "nu": nu}
    J2_effective_stress_params = {"J2": 0.}
    initial_yield_params = {"Y": Y}
    hardening_params = {"neural network": nn_params}

    J2_values = {
        "rotation matrix": np.eye(3),
        "elastic": elastic_params,
        "plastic": {
            "effective stress": J2_effective_stress_params,
            "flow stress": {
                "initial yield": initial_yield_params,
                "hardening": hardening_params}}}

    J2_active_flags = J2_values.copy()
    J2_active_flags = tree_map(lambda x: False, J2_active_flags)
    J2_active_flags["plastic"]["flow stress"] = tree_map(
        lambda x: True, J2_active_flags["plastic"]["flow stress"])

    J2_transforms = J2_values.copy()
    J2_transforms = tree_map(lambda x: None, J2_transforms)

    Y_log_scale = np.array([200.])
    J2_active_flags["plastic"]["flow stress"]["initial yield"]["Y"] = True
    J2_transforms["plastic"]["flow stress"]["initial yield"]["Y"] = Y_log_scale
    J2_transforms["plastic"]["flow stress"]["hardening"]["neural network"] \
        = tree_map(lambda x: np.array([1.]),
        J2_transforms["plastic"]["flow stress"]["hardening"]["neural network"],
        is_leaf=lambda x: x is None)

    J2_parameters = \
        Parameters(J2_values, J2_active_flags, J2_transforms)

    return J2_parameters


def compute_cauchy(model, F):

    num_steps = F.shape[2] - 1
    model.set_xi_to_init_vals()
    cauchy = np.zeros((3, 3, num_steps + 1))

    for step in range(1, num_steps + 1):

        u = [F[:, :, step]]
        u_prev = [F[:, :, step - 1]]
        model.gather_global(u, u_prev)

        model.seed_xi()
        newton_solve(model)
        model.advance_xi()

        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()

    return cauchy


def plot_cauchy(true_cauchy, pred_cauchy, data):

    total_num_steps = data.shape[2]
    steps = np.arange(0, total_num_steps)
    true_sigma_xx = true_cauchy[0, 0, :]
    true_sigma_yy = true_cauchy[1, 1, :]
    pred_sigma_xx = pred_cauchy[0, 0, :]
    pred_sigma_yy = pred_cauchy[1, 1, :]

    test_total_num_steps = pred_cauchy.shape[2]
    test_steps = np.arange(0, test_total_num_steps)

    data_xx = data[0, 0, :]
    data_yy = data[1, 1, :]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(test_steps, true_sigma_xx, color="blue", alpha=0.5,
               label="True $\\sigma_{xx}$", zorder=0)
    ax.scatter(test_steps, true_sigma_yy, color="red", alpha=0.5,
               label="True $\\sigma_{yy}$", zorder=0)
    ax.scatter(test_steps, pred_sigma_xx, color="purple", alpha=0.5,
               label="Predicted $\\sigma_{xx}$", zorder=1)
    ax.scatter(test_steps, pred_sigma_yy, color="orange", alpha=0.5,
               label="Predicted $\\sigma_{yy}$", zorder=1)
    ax.scatter(steps, data_xx, color="black", marker="X", zorder=2, alpha=0.5)
    ax.scatter(steps, data_yy, color="black", marker="X", zorder=2, alpha=0.5)
    ax.set_xlabel("Step", fontsize=22)
    ax.set_ylabel("Stress", fontsize=22)
    ax.set_title(
        "Experiment: $J_2$ Yield with Swift-Voce Hardening", fontsize=22)
    ax.legend(loc="best", fontsize=18)

    return fig


strain_increment = 0.02
num_pts_per_increment = 50
init_strain = strain_increment / num_pts_per_increment
eps_xx = np.r_[np.zeros(1),
               np.linspace(init_strain, strain_increment,
                           num_pts_per_increment),
               np.ones(num_pts_per_increment) * strain_increment]
eps_yy = np.r_[np.zeros(1),
    np.zeros(num_pts_per_increment) * strain_increment,
    np.linspace(
        init_strain,
        strain_increment,
        num_pts_per_increment)]

test_eps_xx = np.r_[eps_xx,
                    np.linspace(strain_increment, 2. * strain_increment,
                                num_pts_per_increment),
                    np.ones(num_pts_per_increment) * 2. * strain_increment,
                    np.linspace(2. * strain_increment, 3. * strain_increment,
                                num_pts_per_increment)]

test_eps_yy = np.r_[eps_yy,
                    np.ones(num_pts_per_increment) * strain_increment,
                    np.linspace(strain_increment, 2. * strain_increment,
                                num_pts_per_increment),
                    np.ones(num_pts_per_increment) * 2. * strain_increment]

def_type = DefType.PLANE_STRESS
ndims = def_type_ndims(def_type)
num_steps = 100
I = np.eye(ndims)
F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
F[0, 0, :] += eps_xx[:num_steps + 1]
F[1, 1, :] += eps_yy[:num_steps + 1]

test_num_steps = len(test_eps_xx)
test_F = np.repeat(I[:, :, np.newaxis], test_num_steps, axis=2)
test_F[0, 0, :] += test_eps_xx
test_F[1, 1, :] += test_eps_yy

weight = np.zeros((3, 3))
weight[0, 0] = 1.
weight[1, 1] = 1.

J2_parameters = create_J2_parameters()
model_true = SmallElasticPlastic(J2_parameters, def_type)

voce_hardening = True
nn_hardening = False
assert voce_hardening + nn_hardening == 1

if voce_hardening:
    J2_parameters = create_J2_parameters()
    model = SmallElasticPlastic(J2_parameters, def_type)
    # convoluted chain of commands to get an initial guess not equal
    # to the true parameters
    true_active_param_values = model_true.parameters.flat_active_values(False)
    initial_guess = 1.1 * true_active_param_values
    model.parameters.set_active_values_from_flat(initial_guess, False)
    initial_guess = model.parameters.flat_active_values(True).copy()

if nn_hardening:
    layer_widths = [1, 5, 1]
    hardening_nn = SimpleNeuralNetwork(layer_widths, input_scale=2.,
                                       output_scale=1e2)
    J2_parameters = create_J2_parameters_nn(hardening_nn.params)
    nn_hardening_fun = {"neural network": hardening_nn.evaluate}
    model = SmallRateElasticPlastic(J2_parameters, def_type,
                                    hardening_funs=nn_hardening_fun)
    initial_guess = model.parameters.flat_active_values(True).copy()

num_active_params = model.parameters.num_active_params
opt_bounds = J2_parameters.opt_bounds

rng = np.random.default_rng(seed=22)
noise_std = 5.

cauchy = compute_cauchy(model_true, F)
data = cauchy + rng.normal(0., noise_std, cauchy.shape)

qoi = Calibration(model, F, data, weight)
objective = Objective(qoi, sensitivity_type="adjoint gradient")

minimize_lbfgs = True
minimize_newton = True

if minimize_lbfgs:

    max_iters = 50
    opt_params, fun_vals, cvg_dict = fmin_l_bfgs_b(
        objective.evaluate, initial_guess, bounds=opt_bounds, iprint=1,
        maxiter=max_iters)

if minimize_newton:
    print("\n")
    objective = Objective(qoi, sensitivity_type="direct-adjoint hessian")
    opt_params = newton_optimization(initial_guess, objective)

model.parameters.set_active_values_from_flat(opt_params)
unscaled_opt_params = model.parameters.flat_active_values()

cauchy_actual = compute_cauchy(model_true, F)
cauchy_fit = compute_cauchy(model, F)

plt.close("all")
cauchy_fig = plot_cauchy(cauchy_actual, cauchy_fit, data)
