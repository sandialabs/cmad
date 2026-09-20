"""Finite uniaxial stress in BeBarElasticPlastic.

Plane stress prescribes both in-plane stretches and solves the third;
uniaxial stress prescribes only the one along the loading axis and solves
the other two, from the two off-axis normal stresses vanishing. So the
whole lateral response is the model's answer here rather than an input,
and driving the axial stretch along the uniaxial stress path has to
recover both lateral stretches and the axial stress.

The path is sampled from
:func:`tests.support.test_problems.finite_uniaxial_j2_voce` in the
hardening variable, as in the plane stress test: interpolating between
its endpoints is not a uniaxial stress path in between, and that is an
error in the path rather than in its discretization, so it would not
shrink as steps are added.

The loading axis is isotropic here -- the elastic response is neohookean
and the yield is J2 -- so ``uniaxial_stress_idx`` only moves the same
answer onto another global axis, which is what pins the plumbing.
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
        num_steps: int, uniaxial_stress_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Step along the uniaxial stress path from rest in ``num_steps``.

    Only the stretch along the loading axis is prescribed, through the
    1x1 deformation gradient a uniaxial stress model is driven by.

    Returns the final Cauchy stress and the two solved off-axis stretches.
    """
    params = J2AnalyticalProblem(scale_params=False).J2_parameters
    model = BeBarElasticPlastic(
        params, def_type=DefType.UNIAXIAL_STRESS,
        uniaxial_stress_idx=uniaxial_stress_idx)
    model.set_xi_to_init_vals()

    F_prev = np.eye(1)
    for step in range(1, num_steps + 1):
        lambda_axial, _l_lat, _cauchy = finite_uniaxial_j2_voce(
            _E, _NU, _Y, _S, _D, _ALPHA * step / num_steps,
        )
        F = np.diag([lambda_axial])
        model.gather_global(mp_U_from_F(F), mp_U_from_F(F_prev))
        newton_solve(model, max_iters=50)
        model.advance_xi()
        F_prev = F

    off_axis_stretches = np.asarray(model.xi()[3])
    model.seed_none()
    model.evaluate_cauchy()
    return np.asarray(model.Sigma()), off_axis_stretches


class TestBeBarUniaxialStress(unittest.TestCase):

    def test_solves_the_off_axis_stretches(self) -> None:
        _l_ax, l_lat, cauchy_ref = finite_uniaxial_j2_voce(
            _E, _NU, _Y, _S, _D, _ALPHA,
        )

        coarse, coarse_stretches = _drive_uniaxial(20)
        fine, fine_stretches = _drive_uniaxial(80)

        # Both unknown stretches converge to the uniaxial lateral one, and
        # the axial stress to the reference, under step refinement.
        err_coarse = np.max(np.abs(coarse_stretches - l_lat)) / l_lat
        err_fine = np.max(np.abs(fine_stretches - l_lat)) / l_lat
        self.assertLess(err_fine, err_coarse)
        self.assertLess(err_fine, 1e-6)

        self.assertLess(
            abs(fine[0, 0] - cauchy_ref) / cauchy_ref,
            abs(coarse[0, 0] - cauchy_ref) / cauchy_ref,
        )
        self.assertLess(abs(fine[0, 0] - cauchy_ref) / cauchy_ref, 5e-4)

        # The two off-axis stresses are imposed rather than approached, so
        # they hold to solver tolerance at either resolution.
        self.assertLess(abs(fine[1, 1]) / cauchy_ref, 1e-10)
        self.assertLess(abs(fine[2, 2]) / cauchy_ref, 1e-10)

        # The state stays diagonal: the uniaxial deformation gradient is,
        # and nothing in the return map rotates it off.
        off_diagonal = fine - np.diag(np.diag(fine))
        self.assertLess(np.max(np.abs(off_diagonal)) / cauchy_ref, 1e-12)

    def test_loading_axis_is_a_relabeling(self) -> None:
        """The model is isotropic, so moving the loading axis moves the
        answer with it and changes nothing else."""
        axial, _stretches = _drive_uniaxial(20, uniaxial_stress_idx=0)

        for idx in (1, 2):
            cauchy, stretches = _drive_uniaxial(20, uniaxial_stress_idx=idx)
            off_axis = [ii for ii in range(3) if ii != idx]

            self.assertAlmostEqual(
                cauchy[idx, idx] / axial[0, 0], 1.0, places=10)
            for ii in off_axis:
                self.assertLess(
                    abs(cauchy[ii, ii]) / abs(axial[0, 0]), 1e-10)
            np.testing.assert_allclose(stretches, _stretches, rtol=1e-10)

    def test_rejects_unsupported_def_types(self) -> None:
        params = J2AnalyticalProblem(scale_params=False).J2_parameters
        with self.assertRaises(NotImplementedError):
            BeBarElasticPlastic(params, def_type=DefType.PURE_SHEAR)


if __name__ == "__main__":
    unittest.main()
