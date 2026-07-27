"""Finite uniaxial J2 convergence check for BeBarElasticPlastic.

The discrete be_bar update is a consistent discretization of finite J2,
so under uniaxial proportional loading it converges to the continuous
reference :func:`tests.support.test_problems.finite_uniaxial_j2_voce` as
the step size shrinks.
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


def _drive_uniaxial(
        lambda_axial: float, lambda_lateral: float, num_steps: int,
) -> np.ndarray:
    """Drive the material point to ``diag(l_ax, l_lat, l_lat)`` in
    ``num_steps`` log-linear steps and return the final Cauchy stress."""
    params = J2AnalyticalProblem(scale_params=False).J2_parameters
    model = BeBarElasticPlastic(params, def_type=DefType.FULL_3D)
    model.set_xi_to_init_vals()
    for step in range(1, num_steps + 1):
        t = step / num_steps
        t_prev = (step - 1) / num_steps
        F = np.diag([lambda_axial**t, lambda_lateral**t, lambda_lateral**t])
        F_prev = np.diag([
            lambda_axial**t_prev, lambda_lateral**t_prev,
            lambda_lateral**t_prev,
        ])
        model.gather_global(mp_U_from_F(F), mp_U_from_F(F_prev))
        newton_solve(model, max_iters=50)
        model.advance_xi()
    model.seed_none()
    model.evaluate_cauchy()
    return np.asarray(model.Sigma())


class TestBeBarElasticPlastic(unittest.TestCase):

    def test_uniaxial_converges_to_continuous_reference(self) -> None:
        l_ax, l_lat, cauchy_ref = finite_uniaxial_j2_voce(
            _E, _NU, _Y, _S, _D, _ALPHA,
        )

        sigma_coarse = _drive_uniaxial(l_ax, l_lat, num_steps=20)
        sigma_fine = _drive_uniaxial(l_ax, l_lat, num_steps=80)

        err_coarse = abs(sigma_coarse[0, 0] - cauchy_ref) / cauchy_ref
        err_fine = abs(sigma_fine[0, 0] - cauchy_ref) / cauchy_ref

        # converges toward the reference under step refinement
        self.assertLess(err_fine, err_coarse)
        self.assertLess(err_fine, 5e-4)
        # lateral stress vanishes (uniaxial stress) in the same limit
        self.assertLess(abs(sigma_fine[1, 1]) / cauchy_ref, 1e-3)
        self.assertLess(abs(sigma_fine[2, 2]) / cauchy_ref, 1e-3)


if __name__ == "__main__":
    unittest.main()
