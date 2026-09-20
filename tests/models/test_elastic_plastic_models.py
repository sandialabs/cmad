import unittest

import numpy as np

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.global_fields import mp_U_from_F
from cmad.models.nonlinear_solver import newton_solve
from cmad.models.rate_elastic_plastic import RateElasticPlastic
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.qois.calibration import Calibration
from tests.support.plotting import plot_uniaxial_cauchy
from tests.support.test_problems import J2AnalyticalProblem


def run_test(model_type, def_type, num_steps=100, max_alpha=0.5):

    J2_analytical_problem = J2AnalyticalProblem()
    models = get_models(J2_analytical_problem, model_type, def_type)

    ndims = def_type_ndims(def_type)
    I = np.eye(ndims)

    stress_masks = get_stress_masks(def_type)

    diff_tol = 1e-6
    for stress_mask in stress_masks:
        stress, strain, alpha = \
            J2_analytical_problem.analytical_solution(stress_mask,
                                                      max_alpha, num_steps)
        F = get_F(I, strain, num_steps)

        weight = np.abs(stress_mask)

        for model in models:
            alpha_diff, cauchy_diff, obj_diff = \
                run_model_and_compare(model, F, weight, alpha, stress)
            assert np.linalg.norm(alpha_diff) < diff_tol
            assert np.linalg.norm(cauchy_diff) < diff_tol
            assert np.linalg.norm(obj_diff) < diff_tol


def finite_errors(def_type, step_counts, max_alpha=0.5):
    """Max errors over the path in alpha and in the Cauchy stress of the
    finite deformation models at each step count, one row per stress mask
    and one column per model."""
    J2_analytical_problem = J2AnalyticalProblem()
    models = get_models(J2_analytical_problem, "rate finite", def_type)

    ndims = def_type_ndims(def_type)
    I = np.eye(ndims)

    stress_masks = get_stress_masks(def_type)

    errors = {}
    for num_steps in step_counts:
        alpha_errors = np.zeros((len(stress_masks), len(models)))
        cauchy_errors = np.zeros_like(alpha_errors)
        for mask_idx, stress_mask in enumerate(stress_masks):
            stress, strain, alpha = \
                J2_analytical_problem.analytical_solution(stress_mask,
                                                          max_alpha, num_steps)
            F = get_F(I, strain, num_steps, finite=True)

            weight = np.abs(stress_mask)

            for model_idx, model in enumerate(models):
                alpha_diff, cauchy_diff, _ = \
                    run_model_and_compare(model, F, weight, alpha, stress)
                alpha_errors[mask_idx, model_idx] = np.abs(alpha_diff).max()
                cauchy_errors[mask_idx, model_idx] = np.abs(cauchy_diff).max()
        errors[num_steps] = (alpha_errors, cauchy_errors)

    return errors


def get_F(I, strain, num_steps, finite=False):
    ndims = I.shape[0]
    F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
    if finite:
        # For the finite deformation model the analytical strain is the
        # logarithmic strain, diagonal for every stress mask here.
        for ii in range(ndims):
            F[ii, ii, 1:] = np.exp(strain[ii, ii, :])
    else:
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
            RateElasticPlastic(problem.J2_parameters, def_type)
        hill_model = \
            RateElasticPlastic(problem.hill_parameters, def_type)
        hosford_model = \
            RateElasticPlastic(problem.hosford_parameters, def_type)
    elif model_type == "rate finite":
        J2_model = \
            RateElasticPlastic(problem.J2_parameters, def_type,
                               finite_deformation=True)
        hill_model = \
            RateElasticPlastic(problem.hill_parameters, def_type,
                               finite_deformation=True)
        hosford_model = \
            RateElasticPlastic(problem.hosford_parameters, def_type,
                               finite_deformation=True)
    else:
        raise NotImplementedError

    return J2_model, hill_model, hosford_model


def run_model_and_compare(model, F, weight, alpha, stress):
    num_steps = F.shape[2] - 1
    xi_at_step = [[None, None] for ii in range(num_steps + 1)]

    model.set_xi_to_init_vals()
    model.store_xi(xi_at_step, model.xi_prev(), 0)

    cauchy = np.zeros((3, 3, num_steps + 1))
    qoi = Calibration(model, cauchy.copy(), weight)
    J = 0.

    for step in range(1, num_steps + 1):

        model.gather_global(
            mp_U_from_F(F[:, :, step]),
            mp_U_from_F(F[:, :, step - 1]),
        )

        newton_solve(model)
        model.store_xi(xi_at_step, model.xi(), step)

        model.seed_none()
        qoi.evaluate(step)
        J += qoi.J()

        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()

        model.advance_xi()

    model_alpha = \
        np.array([xi_at_step[step][1][0]
                  for step in range(1, num_steps + 1)])
    alpha_diff = model_alpha - alpha
    cauchy_diff = cauchy[:, :, 1:] - stress
    obj_diff = J - 0.5 * np.linalg.norm(weight[:, :, np.newaxis] * cauchy)**2

    return alpha_diff, cauchy_diff, obj_diff


class TestJ2Models(unittest.TestCase):
    def test_small_3D(self):
        run_test("small", DefType.FULL_3D)

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


class TestJ2ModelsFiniteDeformation(unittest.TestCase):
    """The rate model with finite deformation converges to the analytical
    solution, read as logarithmic strain and Cauchy stress, at second order
    in the step.
    """

    def _check(self, def_type):
        max_alpha = 0.5
        coarse, fine = 51, 101
        errors = finite_errors(def_type, (coarse, fine), max_alpha)

        for coarse_errors, fine_errors in zip(
                errors[coarse], errors[fine], strict=True):
            ratio = coarse_errors / fine_errors
            self.assertGreater(ratio.min(), 3.9)
            self.assertLess(ratio.max(), 4.1)

        # The error in the midpoint rate of deformation is the cube of the
        # step over 12, which sums to this over the path.
        alpha_step = max_alpha / (fine - 1)
        alpha_errors, _ = errors[fine]
        self.assertLess(
            alpha_errors.max(), 2. * max_alpha * alpha_step**2 / 12.)

    def test_rate_finite_3D(self):
        self._check(DefType.FULL_3D)

    def test_rate_finite_plane_stress(self):
        self._check(DefType.PLANE_STRESS)

    def test_rate_finite_uniaxial_stress(self):
        self._check(DefType.UNIAXIAL_STRESS)


if __name__ == "__main__":
    small_rate_ep_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestJ2Models)
    unittest.TextTestRunner(verbosity=2).run(small_rate_ep_test_suite)
