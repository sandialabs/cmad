"""Verification for the hypoelastic (rate form) elastic-plastic model.

Three checks: small strain consistency with the J2 analytic solution,
objectivity under a superposed rigid rotation of the deformation history,
and independence of the converged state from the time step size.
"""
import unittest

import numpy as np

from cmad.models.deformation_types import DefType
from cmad.models.global_fields import StepTime, mp_U_from_F
from cmad.models.hypo_elastic_plastic import HypoElasticPlastic
from cmad.models.nonlinear_solver import newton_solve
from tests.support.test_problems import J2AnalyticalProblem


def _drive(model, F, times):
    """Run the material point through ``F`` at the given step ``times``.

    Returns the Cauchy stress history (3, 3, n+1) and the alpha history.
    """
    num_steps = F.shape[2] - 1
    model.set_xi_to_init_vals()
    cauchy = np.zeros((3, 3, num_steps + 1))
    alpha = np.zeros(num_steps + 1)
    for step in range(1, num_steps + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step]),
            mp_U_from_F(F[:, :, step - 1]),
        )
        model.gather_time(StepTime(times[step], times[step - 1]))
        newton_solve(model)
        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
        alpha[step] = float(model.xi()[1][0])
        model.advance_xi()
    return cauchy, alpha


def _uniaxial_F(problem, max_alpha, num_steps):
    """Uniaxial stress ``F = I + strain`` from the J2 analytic solution."""
    stress_mask = np.zeros((3, 3))
    stress_mask[0, 0] = 1.0
    stress, strain, alpha = problem.analytical_solution(
        stress_mask, max_alpha, num_steps)
    eye = np.eye(3)
    F = np.repeat(eye[:, :, np.newaxis], num_steps + 1, axis=2)
    F[:, :, 1:] += strain
    return F, stress, alpha


def _rotation():
    """A fixed proper rotation with all three axes engaged."""
    a, b = 0.7, 0.4
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0],
                   [0.0, 0.0, 1.0]])
    Ry = np.array([[np.cos(b), 0.0, np.sin(b)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(b), 0.0, np.cos(b)]])
    return Rz @ Ry


class TestHypoElasticPlastic(unittest.TestCase):

    def test_small_strain_matches_j2(self):
        problem = J2AnalyticalProblem()
        max_alpha, num_steps = 1e-3, 100
        F, stress, alpha = _uniaxial_F(problem, max_alpha, num_steps)
        model = HypoElasticPlastic(problem.J2_parameters, DefType.FULL_3D)
        times = np.arange(num_steps + 1, dtype=float)
        cauchy, model_alpha = _drive(model, F, times)

        # The finite kinematics correction is O(strain) at the yield strain
        # Y/E ~ 1e-3, so the model agrees with small strain J2 to that order.
        stress_scale = np.abs(stress).max()
        stress_error = np.abs(cauchy[:, :, 1:] - stress).max() / stress_scale
        alpha_error = np.abs(model_alpha[1:] - alpha).max() / max_alpha
        self.assertLess(stress_error, 5e-3)
        self.assertLess(alpha_error, 5e-3)

    def test_objectivity_under_superposed_rotation(self):
        problem = J2AnalyticalProblem()
        max_alpha, num_steps = 0.2, 50
        F, _, _ = _uniaxial_F(problem, max_alpha, num_steps)
        times = np.arange(num_steps + 1, dtype=float)

        model = HypoElasticPlastic(problem.J2_parameters, DefType.FULL_3D)
        cauchy, alpha = _drive(model, F, times)

        Q = _rotation()
        QF = np.einsum("ij,jkt->ikt", Q, F)
        model_rot = HypoElasticPlastic(problem.J2_parameters, DefType.FULL_3D)
        cauchy_rot, alpha_rot = _drive(model_rot, QF, times)

        for step in range(1, num_steps + 1):
            expected = Q @ cauchy[:, :, step] @ Q.T
            np.testing.assert_allclose(
                cauchy_rot[:, :, step], expected, atol=1e-8)
        np.testing.assert_allclose(alpha_rot, alpha, atol=1e-10)

    def test_dt_independence(self):
        problem = J2AnalyticalProblem()
        max_alpha, num_steps = 0.2, 50
        F, _, _ = _uniaxial_F(problem, max_alpha, num_steps)
        steps = np.arange(num_steps + 1, dtype=float)

        model = HypoElasticPlastic(problem.J2_parameters, DefType.FULL_3D)
        cauchy_unit, alpha_unit = _drive(model, F, steps)

        model_half = HypoElasticPlastic(problem.J2_parameters, DefType.FULL_3D)
        cauchy_half, alpha_half = _drive(model_half, F, 0.5 * steps)

        np.testing.assert_allclose(cauchy_half, cauchy_unit, atol=1e-8)
        np.testing.assert_allclose(alpha_half, alpha_unit, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
