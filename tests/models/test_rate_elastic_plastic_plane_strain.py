"""Plane strain in RateElasticPlastic against the FULL_3D model.

Plane strain stores the in-plane stress and the 33 stress. Driven with a
2x2 F past yield it must give the stress and alpha of the FULL_3D model
driven with the same F embedded with F_33 = 1.
"""
import unittest

import numpy as np

from cmad.models.deformation_types import DefType
from cmad.models.global_fields import mp_U_from_F
from cmad.models.nonlinear_solver import newton_solve
from cmad.models.rate_elastic_plastic import RateElasticPlastic
from tests.support.test_problems import J2AnalyticalProblem

LOCAL_TOL = 1e-12
NUM_STEPS = 40
STRETCH_X, STRETCH_Y, SHEAR = 0.02, -0.005, 0.03


def _history():
    """In-plane stretch plus shear, ``(2, 2, NUM_STEPS + 1)``."""
    t = np.linspace(0.0, 1.0, NUM_STEPS + 1)
    F = np.zeros((2, 2, NUM_STEPS + 1))
    F[0, 0] = 1.0 + STRETCH_X * t
    F[0, 1] = SHEAR * t
    F[1, 1] = 1.0 + STRETCH_Y * t
    return F


def _embed(F_2D):
    F = np.repeat(np.eye(3)[:, :, np.newaxis], F_2D.shape[2], axis=2)
    F[:2, :2, :] = F_2D
    return F


def _drive(model, F):
    num_steps = F.shape[2] - 1
    model.set_xi_to_init_vals()
    cauchy = np.zeros((3, 3, num_steps + 1))
    alpha = np.zeros(num_steps + 1)
    for step in range(1, num_steps + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step]), mp_U_from_F(F[:, :, step - 1]))
        newton_solve(model, abs_tol=LOCAL_TOL, rel_tol=LOCAL_TOL)
        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
        alpha[step] = float(model.xi()[1][0])
        model.advance_xi()
    return cauchy, alpha


class TestRatePlaneStrain(unittest.TestCase):

    def _assert_matches_full_3d(self, finite_deformation):
        params = J2AnalyticalProblem().J2_parameters
        F_2D = _history()
        model_2D = RateElasticPlastic(
            params, DefType.PLANE_STRAIN,
            finite_deformation=finite_deformation)
        model_3D = RateElasticPlastic(
            params, DefType.FULL_3D, finite_deformation=finite_deformation)
        cauchy_2D, alpha_2D = _drive(model_2D, F_2D)
        cauchy_3D, alpha_3D = _drive(model_3D, _embed(F_2D))

        self.assertGreater(alpha_3D[-1], 0.0)

        # Both solves stop at LOCAL_TOL, which bounds their difference.
        stress_bound = 2.0 * LOCAL_TOL * model_3D.shear_scale_factor
        E = float(params.values["elastic"]["E"])
        self.assertLess(np.abs(cauchy_2D - cauchy_3D).max(), stress_bound)
        self.assertLess(np.abs(alpha_2D - alpha_3D).max(), stress_bound / E)

    def test_small_strain(self):
        self._assert_matches_full_3d(finite_deformation=False)

    def test_finite_deformation(self):
        self._assert_matches_full_3d(finite_deformation=True)


if __name__ == "__main__":
    unittest.main()
