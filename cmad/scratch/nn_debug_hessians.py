import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

from jax import tree_map

from scipy.optimize import fmin_l_bfgs_b

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.deriv_types import DerivType
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.neural_networks.simple_neural_network import SimpleNeuralNetwork
from cmad.objectives.objective import Objective
from cmad.parameters.parameters import Parameters
from cmad.qois.calibration import Calibration
from cmad.qois.tests.test_J2_fd_checks import (fd_grad_check_components,
                                               fd_grad_check)
from cmad.solver.nonlinear_solver import newton_solve


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

        model.evaluate_hessians()

        d2C_dxi2 = model.d2C_dxi2
        d2C_dxi_dxi_prev = model.d2C_dxi_dxi_prev
        d2C_dxi_prev2 = model.d2C_dxi_prev2

        d2C_dparams2 = model.d2C_dparams2
        d2C_dxi_dparams = model.d2C_dxi_dparams
        d2C_dxi_prev_dparams = model.d2C_dxi_prev_dparams

        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()

    return cauchy


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

layer_widths = [1, 2, 1]
hardening_nn = SimpleNeuralNetwork(layer_widths, input_scale=2.,
                                   output_scale=1e2)

J2_parameters_nn = create_J2_parameters_nn(hardening_nn.params)
nn_hardening_fun = {"neural network": hardening_nn.evaluate}
#model = SmallRateElasticPlastic(J2_parameters_nn, def_type,
model = SmallElasticPlastic(J2_parameters_nn, def_type,
                                hardening_funs=nn_hardening_fun)

initial_guess = model.parameters.flat_active_values(True).copy()
num_active_params = model.parameters.num_active_params
opt_bounds = J2_parameters_nn.opt_bounds

rng = np.random.default_rng(seed=22)
noise_std = 5.

cauchy = compute_cauchy(model, F)
