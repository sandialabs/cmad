import numpy as np
import matplotlib.pyplot as plt
import unittest

from scipy.optimize import fmin_l_bfgs_b

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.objectives.objective import Objective
from cmad.qois.calibration import Calibration
from cmad.solver.nonlinear_solver import newton_solve
from cmad.test_support.test_problems import J2AnalyticalProblem


def compute_cauchy(model, F):

    num_steps = F.shape[2] - 1
    model.set_xi_to_init_vals()
    cauchy = np.zeros((3, 3, num_steps + 1))

    for step in range(1, num_steps + 1):

        u = [F[:, :, step]]
        u_prev = [F[:, :, step - 1]]
        model.gather_global(u, u_prev)

        newton_solve(model)
        model.advance_xi()

        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()

    return cauchy


def plot_cauchy(cauchy):

    total_num_steps = cauchy.shape[2]
    steps = np.arange(0, total_num_steps)
    sigma_xx = cauchy[0, 0, :]
    sigma_yy = cauchy[1, 1, :]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(steps, sigma_xx, color="blue", label="$\\sigma_{xx}$")
    ax.scatter(steps, sigma_yy, color="red", label="$\\sigma_{yy}$")
    ax.set_xlabel("Step", fontsize=22)
    ax.set_ylabel("Stress", fontsize=22)
    ax.set_title("Experiment: $J_2$ Yield with Voce Hardening", fontsize=22)
    ax.legend(loc="best", fontsize=18)

    return fig


class TestJ2Calibrations(unittest.TestCase):
    def test_J2_calibrations(self):
        J2_analytical_problem = J2AnalyticalProblem()

        strain_increment = 0.01
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

        def_type = DefType.PLANE_STRESS
        ndims = def_type_ndims(def_type)
        num_steps = len(eps_xx) - 1
        I = np.eye(ndims)
        F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
        F[0, 0, :] += eps_xx
        F[1, 1, :] += eps_yy

        weight = np.zeros((3, 3))
        weight[0, 0] = 1.
        weight[1, 1] = 1.

        model = SmallRateElasticPlastic(J2_analytical_problem.J2_parameters,
                                        def_type)
        true_params = model.parameters.flat_active_values()
        num_active_params = model.parameters.num_active_params
        opt_bounds = model.parameters.opt_bounds

        initial_guess = 0.1 * np.ones(num_active_params)
        diff_tol = 1e-7

        cauchy = compute_cauchy(model, F)
        # cauchy_fig = plot_cauchy(cauchy)

        qoi = Calibration(model, F, cauchy, weight)
        objectives = [Objective(qoi, sensitivity_type="adjoint gradient"),
                      Objective(qoi, sensitivity_type="direct gradient")]

        for objective in objectives:

            opt_params, fun_vals, cvg_dict = fmin_l_bfgs_b(
                objective.evaluate, initial_guess, bounds=opt_bounds, factr=10)
            model.parameters.set_active_values_from_flat(opt_params)
            unscaled_opt_params = model.parameters.flat_active_values()

            assert np.linalg.norm(unscaled_opt_params - true_params) < diff_tol


if __name__ == "__main__":
    J2_calibrations_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestJ2Calibrations)
    unittest.TextTestRunner(verbosity=2).run(J2_calibrations_test_suite)
