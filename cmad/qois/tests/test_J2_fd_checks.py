import numpy as np
import matplotlib.pyplot as plt
import unittest

from jax import tree_map

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.parameters.parameters import Parameters
from cmad.objectives.objective import Objective
from cmad.qois.calibration import Calibration
from cmad.solver.nonlinear_solver import newton_solve
from cmad.test_support.test_problems import J2AnalyticalProblem


def compute_cauchy(model, F):
    num_steps = F.shape[2] - 1
    xi_at_step = [[None] * model.num_residuals \
        for ii in range(num_steps + 1)]

    model.set_xi_to_init_vals()
    model.store_xi(xi_at_step, model.xi_prev(), 0)

    cauchy = np.zeros((3, 3, num_steps + 1))

    for step in range(1, num_steps + 1):

        u = [F[:, :, step]]
        u_prev = [F[:, :, step - 1]]
        model.gather_global(u, u_prev)

        newton_solve(model)
        model.store_xi(xi_at_step, model.xi(), step)

        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()

        model.advance_xi()

    return cauchy


def fd_hessian_check_components(qoi, hs=np.logspace(-2, -10, 9), seed=22):
    model = qoi.model()
    flat_active_values = model.parameters.flat_active_values(True)
    num_active_params = model.parameters.num_active_params
    hessian_obj = Objective(qoi, "direct-adjoint hessian")

    J_ref, grad_ref, hessian_ref = hessian_obj.evaluate(flat_active_values)

    unique_idx = np.triu_indices(num_active_params)
    num_unique_entries = len(unique_idx[0])
    fd_error = np.zeros((len(hs), num_unique_entries))

    for hh, h in enumerate(hs):
        fd_hessian = np.zeros((num_active_params, num_active_params))

        for ii in range(num_active_params):
            for jj in range(num_active_params):

                if ii == jj:
                    params_plus = flat_active_values.copy()
                    params_plus[ii] += h
                    model.parameters.set_active_values_from_flat(params_plus)
                    J_plus = compute_fun(qoi)

                    params_minus = flat_active_values.copy()
                    params_minus[ii] -= h
                    model.parameters.set_active_values_from_flat(params_minus)
                    J_minus = compute_fun(qoi)

                    fd_hessian[ii, ii] = (J_plus + J_minus - 2. * J_ref) \
                        / h**2

                elif ii < jj:

                    params_pp = flat_active_values.copy()
                    params_pp[ii] += h
                    params_pp[jj] += h
                    model.parameters.set_active_values_from_flat(params_pp)
                    J_pp = compute_fun(qoi)

                    params_mm = flat_active_values.copy()
                    params_mm[ii] -= h
                    params_mm[jj] -= h
                    model.parameters.set_active_values_from_flat(params_mm)
                    J_mm = compute_fun(qoi)

                    params_pm = flat_active_values.copy()
                    params_pm[ii] += h
                    params_pm[jj] -= h
                    model.parameters.set_active_values_from_flat(params_pm)
                    J_pm = compute_fun(qoi)

                    params_mp = flat_active_values.copy()
                    params_mp[ii] -= h
                    params_mp[jj] += h
                    model.parameters.set_active_values_from_flat(params_mp)
                    J_mp = compute_fun(qoi)

                    fd_hessian[ii, jj] = (J_pp + J_mm - J_pm - J_mp) \
                        / (4. * h**2)

                else:
                    fd_hessian[ii, jj] = fd_hessian[jj, ii]

        fd_error[hh, :] = np.abs(hessian_ref[unique_idx] \
                        - fd_hessian[unique_idx])

    return fd_error


def fd_hessian_check(qoi, hs=np.logspace(-2, -10, 9), seed=22):
    model = qoi.model()
    flat_active_values = model.parameters.flat_active_values(True)
    num_active_params = model.parameters.num_active_params
    hessian_obj = Objective(qoi, "direct-adjoint hessian")

    J_ref, grad_ref, hessian_ref = hessian_obj.evaluate(flat_active_values)

    #print(f"J ref = {J_ref}")
    #print(f"grad ref = {grad_ref}")
    #print(f"hessian ref = {hessian_ref}")

    np.random.seed(seed)
    d = np.random.uniform(low=-1.0, size=num_active_params)

    dir_deriv_ref = (d @ hessian_ref @ d.T).squeeze()
    #print(f"hessian directional deriv = {dir_deriv_ref:.10e}")

    fd_error = np.zeros(len(hs))

    for ii, h in enumerate(hs):
        params_plus = flat_active_values.copy()
        params_plus += h * d
        model.parameters.set_active_values_from_flat(params_plus)
        J_plus = compute_fun(qoi)

        params_minus = flat_active_values.copy()
        params_minus -= h * d
        model.parameters.set_active_values_from_flat(params_minus)
        J_minus = compute_fun(qoi)

        dir_deriv = (J_plus + J_minus - 2. * J_ref) / h**2
        fd_error[ii] = np.abs(dir_deriv - dir_deriv_ref)
        #print(f"hessian approx directional deriv = {dir_deriv:.10e}")

    return fd_error


def fd_grad_check_components(qoi, hs=np.logspace(-2, -10, 9)):

    model = qoi.model()
    flat_active_values = model.parameters.flat_active_values(True)
    num_active_params = model.parameters.num_active_params

    direct_obj = Objective(qoi, sensitivity_type="direct gradient")
    adjoint_obj = Objective(qoi, sensitivity_type="adjoint gradient")

    J_ref, direct_grad_ref = \
        direct_obj.evaluate(flat_active_values)
    J_ref, adjoint_grad_ref = \
        adjoint_obj.evaluate(flat_active_values)

    fs_fd_error = np.zeros((num_active_params, len(hs)))
    adjoint_fd_error = np.zeros((num_active_params, len(hs)))

    for kk in range(num_active_params):
        for ii, h in enumerate(hs):
            params_plus = flat_active_values.copy()
            params_plus[kk] += h
            model.parameters.set_active_values_from_flat(params_plus)
            J_plus = compute_fun(qoi)

            params_minus = flat_active_values.copy()
            params_minus[kk] -= h
            model.parameters.set_active_values_from_flat(params_minus)
            J_minus = compute_fun(qoi)

            fs_fd_error[kk, ii] = np.abs((J_plus - J_minus) / (2. * h)
                                         - direct_grad_ref[kk])

            adjoint_fd_error[kk, ii] = np.abs((J_plus - J_minus) / (2. * h)
                                              - adjoint_grad_ref[kk])

    return fs_fd_error, adjoint_fd_error


def fd_grad_check(qoi, hs=np.logspace(-2, -10, 9), seed=22):

    model = qoi.model()
    flat_active_values = model.parameters.flat_active_values(True)
    num_active_params = model.parameters.num_active_params

    direct_obj = Objective(qoi, sensitivity_type="direct gradient")
    adjoint_obj = Objective(qoi, sensitivity_type="adjoint gradient")

    J_ref, direct_grad_ref = \
        direct_obj.evaluate(flat_active_values)
    J_ref, adjoint_grad_ref = \
        adjoint_obj.evaluate(flat_active_values)

    np.random.seed(seed)
    d = np.random.uniform(low=-1.0, size=num_active_params)

    fs_dir_deriv_ref = d.dot(direct_grad_ref)
    adjoint_dir_deriv_ref = d.dot(adjoint_grad_ref)

    fs_fd_error = np.zeros(len(hs))
    adjoint_fd_error = np.zeros(len(hs))

    for ii, h in enumerate(hs):
        params_plus = flat_active_values.copy()
        params_plus += h * d
        model.parameters.set_active_values_from_flat(params_plus)
        J_plus = compute_fun(qoi)

        params_minus = flat_active_values.copy()
        params_minus -= h * d
        model.parameters.set_active_values_from_flat(params_minus)
        J_minus = compute_fun(qoi)

        fd_dir_deriv = (J_plus - J_minus) / (2. * h)
        fs_fd_error[ii] = np.abs(fd_dir_deriv - fs_dir_deriv_ref)
        adjoint_fd_error[ii] = np.abs(fd_dir_deriv - adjoint_dir_deriv_ref)

    return fs_fd_error, adjoint_fd_error


def complex_step_grad_check(qoi, qoi_complex, hs=np.logspace(-2, -10, 9),
        seed=22):

    model = qoi.model()
    flat_active_values = model.parameters.flat_active_values(True)
    num_active_params = model.parameters.num_active_params

    direct_obj = Objective(qoi, sensitivity_type="direct gradient")
    adjoint_obj = Objective(qoi, sensitivity_type="adjoint gradient")

    J_ref, direct_grad_ref = \
        direct_obj.evaluate(flat_active_values)
    J_ref, adjoint_grad_ref = \
        adjoint_obj.evaluate(flat_active_values)

    model_complex = qoi_complex.model()
    flat_active_values_complex = model_complex.parameters.flat_active_values(True)

    np.random.seed(seed)
    d = np.random.uniform(low=-1.0, size=num_active_params)

    fs_dir_deriv_ref = d.dot(direct_grad_ref)
    adjoint_dir_deriv_ref = d.dot(adjoint_grad_ref)

    fs_fd_error = np.zeros(len(hs))
    adjoint_fd_error = np.zeros(len(hs))

    for ii, h in enumerate(hs):
        params_plus_complex = flat_active_values_complex.copy()
        params_plus_complex += complex(0, 1) * h * d
        model_complex.parameters.set_active_values_from_flat(params_plus_complex,
            is_complex=True)
        complex_dir_deriv = compute_fun(qoi_complex).imag / h
        fs_fd_error[ii] = np.abs(complex_dir_deriv - fs_dir_deriv_ref)
        adjoint_fd_error[ii] = np.abs(complex_dir_deriv - adjoint_dir_deriv_ref)

    return fs_fd_error, adjoint_fd_error


def complex_step_hessian_check(qoi, qoi_complex, hs=np.logspace(-2, -10, 9), seed=22):
    model = qoi.model()
    flat_active_values = model.parameters.flat_active_values(True)
    num_active_params = model.parameters.num_active_params
    hessian_obj = Objective(qoi, "direct-adjoint hessian")

    J_ref, grad_ref, hessian_ref = hessian_obj.evaluate(flat_active_values)

    model_complex = qoi_complex.model()
    flat_active_values_complex = model_complex.parameters.flat_active_values(True)

    np.random.seed(seed)
    d = np.random.uniform(low=-1.0, size=num_active_params)

    dir_deriv_ref = (d @ hessian_ref @ d.T).squeeze()

    fd_error = np.zeros(len(hs))

    for ii, h in enumerate(hs):
        # Evaluate objective function with perturbation
        params_plus_complex = flat_active_values_complex.copy()
        params_plus_complex += complex(0.5, np.sqrt(3.) / 2.) * h * d
        model_complex.parameters.set_active_values_from_flat(params_plus_complex,
            is_complex=True)
        J_1 = compute_fun(qoi_complex)

        params_plus_complex = flat_active_values_complex.copy()
        params_plus_complex += complex(-0.5, -np.sqrt(3.) / 2.) * h * d
        model_complex.parameters.set_active_values_from_flat(params_plus_complex,
            is_complex=True)
        J_2 = compute_fun(qoi_complex)

        dir_deriv = (J_1 + J_2).imag / (np.sqrt(3.) / 2. * h**2)
        fd_error[ii] = np.abs(dir_deriv - dir_deriv_ref)

    return fd_error


def compute_fun(qoi):

    model = qoi.model()
    F = qoi.global_state()

    num_steps = F.shape[2] - 1
    model.set_xi_to_init_vals()

    J = 0.

    for step in range(1, num_steps + 1):

        u = [F[:, :, step]]
        u_prev = [F[:, :, step - 1]]
        model.gather_global(u, u_prev)

        model.seed_xi()
        newton_solve(model)

        model.seed_none()
        qoi.evaluate(step)
        J += qoi.J()

        model.advance_xi()

    return J


class TestJ2FDChecks(unittest.TestCase):


    def test_J2_finite_difference_checks(self):
        Models = [SmallElasticPlastic, SmallRateElasticPlastic]
        scale_params = [False, True]

        for Model, scale_params in zip(Models, scale_params):
            self.plane_stress_fd_checks(Model, scale_params)


    def get_plane_stress_deformation_gradient():
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

        def_type = DefType.PLANE_STRESS
        ndims = def_type_ndims(def_type)
        num_steps = 100
        I = np.eye(ndims)
        F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
        F[0, 0, :] += eps_xx[:num_steps + 1]
        F[1, 1, :] += eps_yy[:num_steps + 1]

        return F


    def get_plane_stress_data_and_weight(cauchy, noise_std=0., seed=22):
        rng = np.random.default_rng(seed=seed)
        data = cauchy + rng.normal(0., noise_std, cauchy.shape)

        weight = np.zeros((3, 3))
        weight[0, 0] = 1.
        weight[1, 1] = 1.

        return data, weight


    @staticmethod
    def plane_stress_fd_checks(Model, scale_params):

        J2_analytical_problem = J2AnalyticalProblem(scale_params)
        J2_parameters = J2_analytical_problem.J2_parameters

        J2_analytical_problem_complex = J2AnalyticalProblem(scale_params)
        J2_parameters_complex = J2_analytical_problem_complex.J2_parameters

        F = TestJ2FDChecks.get_plane_stress_deformation_gradient()

        # instantiate real model
        model = Model(J2_parameters, DefType.PLANE_STRESS, is_complex=False)

        cauchy = compute_cauchy(model, F)

        # create the real qoi
        data, weight = TestJ2FDChecks.get_plane_stress_data_and_weight(cauchy)
        qoi = Calibration(model, F, data, weight)

        # evaluate FD tests at values that are offset
        # from those used to generate the calibration data
        true_active_param_values = model.parameters.flat_active_values(False)
        offset_param_values = 1.1 * true_active_param_values
        complex_offset_param_values = offset_param_values.astype(complex)

        # instantiate complex model and qoi
        model_complex = Model(J2_parameters_complex,
            def_type=DefType.PLANE_STRESS, is_complex=True)
        qoi_complex = Calibration(model_complex, F, data.astype(complex),
            weight)

        # FD perturbations
        h = np.logspace(0, -9, 20)

        model.parameters.set_active_values_from_flat(offset_param_values, False)
        fs_fd_dir_deriv_error, adjoint_fd_dir_deriv_error = \
            fd_grad_check(qoi, h)
        assert np.allclose(fs_fd_dir_deriv_error,
                           adjoint_fd_dir_deriv_error)

        error_drop_tol = 6.
        min_fd_error = np.min(fs_fd_dir_deriv_error)
        max_fd_error = np.max(fs_fd_dir_deriv_error)
        grad_log10_error_drop = np.log10(max_fd_error / min_fd_error)
        assert grad_log10_error_drop > error_drop_tol

        model.parameters.set_active_values_from_flat(offset_param_values, False)
        fs_fd_component_error, adjoint_fd_component_error = \
            fd_grad_check_components(qoi)
        fd_component_diff_tol = 1e-8
        fd_components_diff = fs_fd_component_error \
            - adjoint_fd_component_error
        assert np.linalg.norm(fd_components_diff) < fd_component_diff_tol

        model_complex.parameters.set_active_values_from_flat(
            complex_offset_param_values, False, is_complex=True
        )
        fs_complex_step_dir_deriv_error, adjoint_complex_step_dir_deriv_error = \
            complex_step_grad_check(qoi, qoi_complex, h)
        assert np.allclose(fs_complex_step_dir_deriv_error,
                           adjoint_complex_step_dir_deriv_error)

        complex_min_fd_error = np.min(fs_complex_step_dir_deriv_error)
        complex_max_fd_error = np.max(fs_complex_step_dir_deriv_error)
        complex_grad_log10_error_drop = \
            np.log10(complex_max_fd_error / complex_min_fd_error)
        assert complex_grad_log10_error_drop > error_drop_tol

        model.parameters.set_active_values_from_flat(offset_param_values, False)
        hessian_fd_dir_deriv_error = fd_hessian_check(qoi, h)
        min_fd_error = np.min(hessian_fd_dir_deriv_error)
        max_fd_error = np.max(hessian_fd_dir_deriv_error)
        hessian_log10_error_drop = np.log10(max_fd_error / min_fd_error)
        assert hessian_log10_error_drop > error_drop_tol

        model_complex.parameters.set_active_values_from_flat(
            complex_offset_param_values, False, is_complex=True)
        hessian_complex_step_dir_deriv_error = complex_step_hessian_check(
            qoi, qoi_complex, h)
        min_complex_step_error = np.min(hessian_complex_step_dir_deriv_error)
        max_complex_step_error = np.max(hessian_complex_step_dir_deriv_error)
        complex_hessian_log10_error_drop = \
            np.log10(max_complex_step_error / min_complex_step_error)
        assert complex_hessian_log10_error_drop > error_drop_tol

        model.parameters.set_active_values_from_flat(offset_param_values, False)
        hessian_fd_component_error = fd_hessian_check_components(qoi, h)

if __name__ == "__main__":
    J2_FD_checks_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestJ2FDChecks)
    unittest.TextTestRunner(verbosity=2).run(J2_FD_checks_test_suite)
