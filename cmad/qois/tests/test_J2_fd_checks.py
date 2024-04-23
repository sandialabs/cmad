import numpy as np
import matplotlib.pyplot as plt
import unittest

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.parameters.parameters import Parameters
from cmad.objectives.objective import Objective
from cmad.qois.calibration import Calibration
from cmad.solver.nonlinear_solver import newton_solve
from cmad.test_support.test_problems import J2AnalyticalProblem


def evaluate_hessian(qoi):
    model = qoi.model()
    flat_active_values = model.parameters.flat_active_values(True)
    num_active_params = model.parameters.num_active_params
    hessian_obj = Objective(qoi, "direct-adjoint hessian")
    return hessian_obj.evaluate(flat_active_values)


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

    np.random.seed(seed)
    d = np.random.uniform(low=-1.0, size=num_active_params)

    dir_deriv_ref = (d @ hessian_ref @ d.T).squeeze()

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


#class TestJ2FDChecks(unittest.TestCase):
#def test_J2_finite_difference_grads(self):
def test_J2_finite_difference_grads():

    # with parameter scaling
    #J2_analytical_problem = J2AnalyticalProblem()

    # without parameter scaling
    scale_params = True
    J2_analytical_problem = J2AnalyticalProblem(scale_params)

    # numbers of steps to run solver for (does not include the IC)
    num_steps = 100

    # uniaxial stress analytical solution
    stress_mask = np.zeros((3, 3))
    stress_mask[0, 0] = 1.
    max_alpha = 0.5
    stress, strain, alpha = \
        J2_analytical_problem.analytical_solution(stress_mask, max_alpha,
                                                  num_steps)

    # construct the global state variable F
    def_type = DefType.FULL_3D
    ndims = def_type_ndims(def_type)
    I = np.eye(ndims)
    F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
    F[0, 0, 1:] += strain[0, 0, :]
    F[1, 1, 1:] += strain[1, 1, :]
    F[2, 2, 1:] += strain[2, 2, :]

    # weight for calibration qoi
    weight = np.zeros((3, 3))
    weight[0, 0] = 1.

    model = SmallRateElasticPlastic(J2_analytical_problem.J2_parameters,
                                    def_type)

    # tweak the parameters to make the solution not uniaxial
    new_params = np.array([210., 21., 109.])

    zero_data = np.zeros((3, 3, num_steps + 1))
    qoi = Calibration(model, F, zero_data, weight)

    model.parameters.set_active_values_from_flat(new_params, False)
    fs_fd_dir_deriv_error, adjoint_fd_dir_deriv_error = \
        fd_grad_check(qoi, seed=10)

    model.parameters.set_active_values_from_flat(new_params, False)
    fs_fd_component_error, adjoint_fd_component_error = \
        fd_grad_check_components(qoi)

    assert np.allclose(fs_fd_dir_deriv_error,
                       adjoint_fd_dir_deriv_error)

    error_drop_tol = 4.
    min_fd_error = np.min(fs_fd_dir_deriv_error)
    max_fd_error = np.max(fs_fd_dir_deriv_error)
    log10_error_drop = np.log10(max_fd_error / min_fd_error)
    assert log10_error_drop > error_drop_tol

    diff_tol = 1e-6
    fd_components_diff = fs_fd_component_error \
        - adjoint_fd_component_error
    assert np.linalg.norm(fd_components_diff) < diff_tol

    hessian = evaluate_hessian(qoi)
    #hessian_fd_error = fd_hessian_check(qoi)
    #hessian_fd_error = fd_hessian_check(qoi, np.logspace(-2, -10, 9))
    hessian_fd_error = fd_hessian_check(qoi, np.logspace(0, -4, 20))
    #hessian_fd_component_error = fd_hessian_check_components(qoi)
    hessian_fd_component_error = fd_hessian_check_components(qoi, np.logspace(0, -4, 20))

    assert False


#if __name__ == "__main__":
#    J2_FD_checks_test_suite = unittest.TestLoader().loadTestsFromTestCase(
#        TestJ2FDChecks)
#    unittest.TextTestRunner(verbosity=2).run(J2_FD_checks_test_suite)

test_J2_finite_difference_grads()
