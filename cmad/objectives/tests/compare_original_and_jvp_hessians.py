import numpy as np
import matplotlib.pyplot as plt

from jax import jit, value_and_grad, hessian, tree_map
from jax.lax import fori_loop, while_loop

from functools import partial

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.objectives.jvp_objective import JVPObjective
from cmad.objectives.objective import Objective
from cmad.parameters.parameters import Parameters
from cmad.qois.calibration import Calibration
from cmad.solver.nonlinear_solver import newton_solve, make_newton_solve


def get_xis(update_fun, model, F, parameters):
    num_steps = F.shape[-1] - 1
    xis = [model._init_xi]
    for step in range(1, num_steps + 1):
        u = [F[:, :, step]]
        u_prev = [F[:, :, step - 1]]
        xis.append(update_fun(xis[-1], parameters, u, u_prev))
    return xis


def get_xis_cauchy(update_fun, model, F, parameters):
    num_steps = F.shape[-1] - 1
    xis = get_xis(update_fun, model, F, parameters)
    cauchy = [jnp.zeros((3, 3))]

    for step in range(1,num_steps + 1):
        u = [F[:, :, step]]
        u_prev = [F[:, :, step - 1]]
        cauchy.append(model.cauchy(xis[step], xis[step-1],
            parameters, u, u_prev))
    return xis, cauchy


# original non-JVP approach
def compute_xi_and_cauchy(model, F):
    num_steps = F.shape[2] - 1
    xi_at_step = [[None, None, None] for ii in range(num_steps + 1)]

    model.set_xi_to_init_vals()
    model.store_xi(xi_at_step, model.xi_prev(), 0)

    cauchy = np.zeros((3, 3, num_steps + 1))

    for step in range(1, num_steps + 1):

        u = [F[:, :, step]]
        u_prev = [F[:, :, step - 1]]
        model.gather_global(u, u_prev)

        model.seed_xi()
        newton_solve(model)
        model.store_xi(xi_at_step, model.xi(), step)

        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()

        model.advance_xi()

    return xi_at_step, cauchy


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


def jvp_fd_grad_check_components(qoi, update_fun, hs=np.logspace(-2, -10, 9)):

    model = qoi.model()
    flat_active_values = model.parameters.flat_active_values(True)
    num_active_params = model.parameters.num_active_params

    obj = JVPObjective(qoi, update_fun)

    J_ref, grad_ref = \
        obj.evaluate_objective_and_grad(flat_active_values)

    fd_error = np.zeros((num_active_params, len(hs)))

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

            fd_error[kk, ii] = np.abs((J_plus - J_minus) / (2. * h)
                - grad_ref[kk])


    return fd_error


def jvp_fd_grad_check(qoi, update_fun, hs=np.logspace(-2, -10, 9), seed=22):

    model = qoi.model()
    flat_active_values = model.parameters.flat_active_values(True)
    num_active_params = model.parameters.num_active_params

    obj = JVPObjective(qoi, update_fun)

    J_ref, grad_ref = \
        obj.evaluate_objective_and_grad(flat_active_values)

    np.random.seed(seed)
    d = np.random.uniform(low=-1.0, size=num_active_params)

    dir_deriv_ref = d.dot(grad_ref)

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

        fd_dir_deriv = (J_plus - J_minus) / (2. * h)
        fd_error[ii] = np.abs(fd_dir_deriv - dir_deriv_ref)

    return fd_error


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
    print(f"original J ref = {J_ref}")
    print(f"original grad ref = {grad_ref}")
    print(f"original hessian ref = {hessian_ref}")

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


def jvp_fd_hessian_check_components(qoi, update_fun,
        hs=np.logspace(-2, -10, 9), seed=22):

    model = qoi.model()
    flat_active_values = model.parameters.flat_active_values(True)
    num_active_params = model.parameters.num_active_params

    hessian_obj = JVPObjective(qoi, update_fun)

    J_ref = hessian_obj.evaluate_objective(flat_active_values)
    hessian_ref = hessian_obj.evaluate_hessian(flat_active_values)

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


def jvp_fd_hessian_check(qoi, update_fun, hs=np.logspace(-2, -10, 9), seed=22):
    model = qoi.model()
    flat_active_values = model.parameters.flat_active_values(True)
    num_active_params = model.parameters.num_active_params
    hessian_obj = JVPObjective(qoi, update_fun)

    J_ref, grad_ref = hessian_obj.evaluate_objective_and_grad(flat_active_values)
    hessian_ref = hessian_obj.evaluate_hessian(flat_active_values)
    print(f"jvp J ref = {J_ref}")
    print(f"jvp grad ref = {grad_ref}")
    print(f"jvp hessian ref = {hessian_ref}")

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


if __name__ == "__main__":

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

    J2_parameters = create_J2_parameters()
    model = SmallElasticPlastic(J2_parameters, def_type)
    xi, cauchy = compute_xi_and_cauchy(model, F)

    rng = np.random.default_rng(seed=22)
    noise_std = 0.
    data = cauchy + rng.normal(0., noise_std, cauchy.shape)
    weight = np.zeros((3, 3))
    weight[0, 0] = 1.
    weight[1, 1] = 1.

    h = np.logspace(1, -8, 20)

    # test original hessian
    true_active_param_values = model.parameters.flat_active_values(False)
    initial_guess = 1.1 * true_active_param_values
    qoi = Calibration(model, F, data, weight)

    model.parameters.set_active_values_from_flat(initial_guess, False)
    grad_fd_error = fd_grad_check(qoi, h)[0]

    model.parameters.set_active_values_from_flat(initial_guess, False)
    hessian_fd_error = fd_hessian_check(qoi, h)

    model.parameters.set_active_values_from_flat(initial_guess, False)
    grad_fd_component_error = fd_grad_check_components(qoi, h)

    model.parameters.set_active_values_from_flat(initial_guess, False)
    hessian_fd_component_error = fd_hessian_check_components(qoi, h)

    # test jvp hessian
    jvp_J2_parameters = create_J2_parameters()
    jvp_model = SmallElasticPlastic(jvp_J2_parameters, def_type)
    jvp_qoi = Calibration(jvp_model, F, data, weight)
    update = make_newton_solve(jvp_model._residual, jvp_model._init_xi)

    jvp_true_active_param_values = jvp_model.parameters.flat_active_values(False)
    jvp_initial_guess = 1.1 * jvp_true_active_param_values

    jvp_model.parameters.set_active_values_from_flat(jvp_initial_guess, False)
    jvp_grad_fd_error = jvp_fd_grad_check(qoi, update, h)

    jvp_model.parameters.set_active_values_from_flat(jvp_initial_guess, False)
    jvp_hessian_fd_error = jvp_fd_hessian_check(jvp_qoi, update, h)

    jvp_model.parameters.set_active_values_from_flat(jvp_initial_guess, False)
    jvp_grad_fd_component_error = jvp_fd_grad_check_components(qoi, update, h)

    jvp_model.parameters.set_active_values_from_flat(jvp_initial_guess, False)
    jvp_hessian_fd_component_error = jvp_fd_hessian_check_components(jvp_qoi, update, h)
