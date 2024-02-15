import numpy as np
import unittest

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.qois.calibration import Calibration
from cmad.solver.nonlinear_solver import newton_solve
from cmad.test_support.plotting import plot_uniaxial_cauchy
from cmad.test_support.test_problems import J2AnalyticalProblem


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

        for model in models:
            run_model_and_compare(model, F, weight, alpha, stress)


def get_F(I, strain, num_steps):
    ndims = I.shape[0]
    F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
    F[:, :, 1:] += strain[:ndims, :ndims, :]

    return F


def get_stress_masks(def_type):
    if def_type == DefType.FULL_3D or def_type == DefType.PLANE_STRESS:
        stress_masks = [np.zeros((3, 3))] * 2
        # uniaxial stress
        stress_masks[0][0, 0] = 1.
        # equal and opposite biaxial stress
        stress_masks[1][0, 0] = 1.
        stress_masks[1][1, 1] = -1.
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

        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()

        model.advance_xi()

    diff_tol = 1e-6
    model_alpha = \
        np.array([xi_at_step[step][1][0]
                  for step in range(1, num_steps + 1)])
    assert np.linalg.norm(model_alpha - alpha) < diff_tol

    cauchy_diff = cauchy[:, :, 1:] - stress
    assert np.linalg.norm(cauchy_diff) < diff_tol

    obj_diff = J - 0.5 * np.linalg.norm(weight[:, :, np.newaxis] * cauchy)**2
    assert np.linalg.norm(obj_diff) < diff_tol


class TestJ2Models(unittest.TestCase):
    def test_small_3D(self):
        run_test("small", DefType.FULL_3D, num_steps=100)

    def test_small_plane_stress(self):
        run_test("small", DefType.PLANE_STRESS)

    def test_small_uniaxial_stress(self):
        run_test("small", DefType.UNIAXIAL_STRESS)

    def test_small_rate_3D(self):
        run_test("small rate", DefType.FULL_3D)

    def test_small_rate_plane_stress(self):
        run_test("small rate", DefType.PLANE_STRESS)

    def test_small_rate_uniaxial_stress(self):
        run_test("small rate", DefType.UNIAXIAL_STRESS)


if __name__ == "__main__":
    small_rate_ep_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestJ2Models)
    unittest.TextTestRunner(verbosity=2).run(small_rate_ep_test_suite)
