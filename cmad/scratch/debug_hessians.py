import numpy as np
import unittest

import jax.numpy as jnp

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.deriv_types import DerivType
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.qois.calibration import Calibration
from cmad.solver.nonlinear_solver import newton_solve
from cmad.test_support.plotting import plot_uniaxial_cauchy
from cmad.test_support.test_problems import J2AnalyticalProblem

from jax import tree_map, jit
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten, tree_flatten_with_path, tree_reduce



def run_test(model_type, def_type, num_steps=100, max_alpha=0.5):

    J2_analytical_problem = J2AnalyticalProblem()
    models = get_models(J2_analytical_problem, model_type, def_type)

    ndims = def_type_ndims(def_type)
    I = np.eye(ndims)

    stress_masks = get_stress_masks(def_type)

    for stress_mask in stress_masks:
        stress, strain, alpha = \
            J2_analytical_problem.analytical_solution(stress_mask,
                                                      max_alpha, num_steps)
        F = get_F(I, strain, num_steps)

        weight = np.abs(stress_mask)

        #for model in models:
        #    run_model_and_compare(model, F, weight, alpha, stress)
        model = models[1]
        run_model_and_compare(model, F, weight, alpha, stress)


def get_F(I, strain, num_steps):
    ndims = I.shape[0]
    F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
    F[:, :, 1:] += strain[:ndims, :ndims, :]

    return F


def get_stress_masks(def_type):
    if def_type == DefType.FULL_3D or def_type == DefType.PLANE_STRESS:
        stress_masks = [None] * 2
        # uniaxial stress
        stress_masks[0] = np.zeros((3, 3))
        stress_masks[0][0, 0] = 1.
        # equal and opposite biaxial stress
        stress_masks[1] = np.eye(3)
        stress_masks[1][1, 1] = -1.
        stress_masks[1][2, 2] = 0.
    elif def_type == DefType.UNIAXIAL_STRESS:
        stress_masks = [np.zeros((3, 3))] * 1
        stress_masks[0][0, 0] = 1.
    else:
        raise NotImplementedError

    return stress_masks


def get_models(problem, model_type, def_type):
    if model_type == "small":
        J2_model = \
            SmallElasticPlastic(problem.J2_parameters, def_type)
        hill_model = \
            SmallElasticPlastic(problem.hill_parameters, def_type)
        hosford_model = \
            SmallElasticPlastic(problem.hosford_parameters, def_type)
    elif model_type == "small rate":
        J2_model = \
            SmallRateElasticPlastic(problem.J2_parameters, def_type)
        hill_model = \
            SmallRateElasticPlastic(problem.hill_parameters, def_type)
        hosford_model = \
            SmallRateElasticPlastic(problem.hosford_parameters, def_type)
    else:
        raise NotImplementedError

    return J2_model, hill_model, hosford_model


def get_model_state_hessian(model, first_deriv_type, second_deriv_type):
    hessians = model._d2C

    num_dofs = model.num_dofs
    num_residuals = model.num_residuals

    d2C_dxi2 = np.concatenate([np.concatenate(
        [hessians[first_deriv_type][row_res_idx][second_deriv_type][col_res_idx]
        for col_res_idx in range(num_residuals)], axis=2)
        for row_res_idx in range(num_residuals)], axis=1)

    return d2C_dxi2


def get_model_params_hessian(model, second_deriv_type):
    first_deriv_type = DerivType.DPARAMS
    hessians = model._d2C

    num_active_params = model.parameters.num_active_params
    num_dofs = model.num_dofs
    num_residuals = model.num_residuals
    d2C_dxi2 = np.zeros((num_dofs, num_dofs, num_dofs))

    #m = tree_map(lambda y : y[2], tree_map(lambda x: x, hessians[2]))

    #z = jnp.concatenate(tree_flatten(tree_map(lambda x: x.reshape(num_dofs, -1),
    #    hessians[2]["elastic"]["E"][2]))[0], axis=1)

    #z = tree_flatten(tree_map(jnp.concatenate(tree_flatten(tree_map(lambda x: x.reshape(num_dofs, -1),
    #    y))[0], axis=1), y))[0]

    z = jnp.concatenate(tree_flatten(tree_map(lambda x: x.reshape(num_dofs, -1),
        hessians[2]["elastic"]["E"][2]))[0], axis=1)

    #z = jnp.stack(tree_map(jnp.concatenate(
    #    tree_map: lambda x: x.reshape(num_dofs, -1), axis=1)

    #flats, _ = tree_flatten_with_path(hessians[2])

    #z = jnp.stack(jnp.concatenate(tree_map(lambda y:
    #    tree_flatten(tree_map(lambda x: x.reshape(num_dofs, -1),
    #    y[2]))[0], axis=1), hessians[2]), axis=2)

    #z = jnp.stack(tree_flatten(tree_map(
    #    jnp.concatenate(lambda y: tree_flatten(tree_map(
    #    lambda x: x.reshape(num_dofs, -1), y[2]))[0], axis=1),
    #    hessians[2]))[0], axis=2)

    #z = jnp.stack(jnp.concatenate(tree_map(lambda y:
    #    tree_flatten(tree_map(lambda x: x.reshape(num_dofs, -1),
    #    y[2]))[0]), hessians[2]), axis=2)


    #z = np.concatenate(tree_map(lambda x:
    #    ravel_pytree(hessians[2]["elastic"]["E"][2])))

    #flat_hess = np.concatenate(tree_map(lambda x: ravel_pytree(x)[0],
    #    hessians[first_deriv_type]), axis=

    assert False

    return d2C_dxi2


def run_model_and_compare(model, F, weight, alpha, stress):
    num_steps = F.shape[2] - 1
    xi_at_step = [[None, None] for ii in range(num_steps + 1)]

    model.set_xi_to_init_vals()
    model.store_xi(xi_at_step, model.xi_prev(), 0)

    cauchy = np.zeros((3, 3, num_steps + 1))
    qoi = Calibration(model, F, cauchy.copy(), weight)
    J = 0.

    for step in range(1, num_steps + 1):

        u = [F[:, :, step]]
        u_prev = [F[:, :, step - 1]]
        model.gather_global(u, u_prev)

        newton_solve(model)
        model.store_xi(xi_at_step, model.xi(), step)

        model.seed_none()
        qoi.evaluate(step)
        J += qoi.J()

        model.evaluate_hessians()

        d2C_dxi2 = get_model_state_hessian(model, DerivType.DXI, DerivType.DXI)
        d2C_dxi_dxi_prev = get_model_state_hessian(model,
                           DerivType.DXI, DerivType.DXI_PREV)
        d2C_dxi_prev2 = get_model_state_hessian(model,
                        DerivType.DXI_PREV, DerivType.DXI_PREV)

        z = get_model_params_hessian(model, DerivType.DPARAMS)

        qoi.evaluate_hessians(step)
        y = qoi._d2J

        #z = model._hessian[0](*model.variables())
        #print(z)
        assert False

        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()

        model.advance_xi()

    diff_tol = 1e-6
    model_alpha = \
        np.array([xi_at_step[step][1][0]
                  for step in range(1, num_steps + 1)])
    #assert np.linalg.norm(model_alpha - alpha) < diff_tol

    cauchy_diff = cauchy[:, :, 1:] - stress
    #assert np.linalg.norm(cauchy_diff) < diff_tol

    obj_diff = J - 0.5 * np.linalg.norm(weight[:, :, np.newaxis] * cauchy)**2
    #assert np.linalg.norm(obj_diff) < diff_tol



run_test("small", DefType.FULL_3D)
