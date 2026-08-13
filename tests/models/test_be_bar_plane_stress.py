"""Finite plane stress in BeBarElasticPlastic.

Plane strain embeds ``F_33 = 1``; plane stress instead carries the out of
plane stretch as a local unknown, fixed by ``sigma_33 = 0``. Uniaxial
stress is a plane stress state, so driving the in-plane stretches along
the uniaxial stress path must recover that unknown as the lateral
stretch, and the axial stress as the reference value.

The path is sampled from
:func:`tests.support.test_problems.finite_uniaxial_j2_voce` in the
hardening variable rather than interpolated between its endpoints.
Interpolating log-linearly is not a uniaxial stress path in between, the
transverse stretch being off by up to 2e-4, which leaves the in-plane
transverse stress nonzero while the out of plane one is driven to zero.
That is a difference in the path rather than in its discretization, so it
does not shrink as steps are added.
"""
import unittest

import numpy as np

from cmad.models.be_bar_elastic_plastic import BeBarElasticPlastic
from cmad.models.deformation_types import DefType
from cmad.models.global_fields import mp_U_from_F
from cmad.models.nonlinear_solver import newton_solve
from tests.support.test_problems import (
    J2AnalyticalProblem,
    finite_uniaxial_j2_voce,
)

# E, nu, Y, S, D (matches J2AnalyticalProblem)
_E, _NU, _Y, _S, _D = 200e3, 0.3, 200.0, 200.0, 20.0
_ALPHA = 0.15


def _drive_uniaxial(num_steps: int) -> tuple[np.ndarray, float]:
    """Step along the uniaxial stress path from rest in ``num_steps``.

    Returns the final Cauchy stress and the solved out of plane stretch.
    """
    params = J2AnalyticalProblem(scale_params=False).J2_parameters
    model = BeBarElasticPlastic(params, def_type=DefType.PLANE_STRESS)
    model.set_xi_to_init_vals()

    F_prev = np.eye(2)
    for step in range(1, num_steps + 1):
        lambda_axial, lambda_lateral, _ = finite_uniaxial_j2_voce(
            _E, _NU, _Y, _S, _D, _ALPHA * step / num_steps,
        )
        F = np.diag([lambda_axial, lambda_lateral])
        model.gather_global(mp_U_from_F(F), mp_U_from_F(F_prev))
        newton_solve(model, max_iters=50)
        model.advance_xi()
        F_prev = F

    out_of_plane_stretch = float(np.asarray(model.xi()[3])[0])
    model.seed_none()
    model.evaluate_cauchy()
    return np.asarray(model.Sigma()), out_of_plane_stretch


class TestBeBarPlaneStress(unittest.TestCase):

    def test_solves_the_out_of_plane_stretch(self) -> None:
        _l_ax, l_lat, cauchy_ref = finite_uniaxial_j2_voce(
            _E, _NU, _Y, _S, _D, _ALPHA,
        )

        coarse, coarse_stretch = _drive_uniaxial(20)
        fine, fine_stretch = _drive_uniaxial(80)

        # The unknown stretch converges to the uniaxial lateral one, and
        # the axial stress to the reference, under step refinement.
        err_coarse = abs(coarse_stretch - l_lat) / l_lat
        err_fine = abs(fine_stretch - l_lat) / l_lat
        self.assertLess(err_fine, err_coarse)
        self.assertLess(err_fine, 1e-6)

        self.assertLess(
            abs(fine[0, 0] - cauchy_ref) / cauchy_ref,
            abs(coarse[0, 0] - cauchy_ref) / cauchy_ref,
        )
        self.assertLess(abs(fine[0, 0] - cauchy_ref) / cauchy_ref, 5e-4)

        # In-plane transverse stress vanishes in the same limit, while the
        # out of plane one is imposed and so holds to solver tolerance.
        self.assertLess(abs(fine[1, 1]) / cauchy_ref, 1e-5)
        self.assertLess(abs(fine[2, 2]) / cauchy_ref, 1e-10)


if __name__ == "__main__":
    unittest.main()
