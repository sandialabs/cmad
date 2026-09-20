"""The material point Newton starts a step from the model's initial guess.

A model that carries an advected elastic or stress state has a second
return map root at the yield front, on the yield surface but with a
negative plastic increment. Starting the local solve from the elastic
predictor is what keeps it away from that root, so the solver has to apply
the guess rather than leave xi where advance_xi put it.

Driven with ``max_iters=0`` so the iteration never runs and xi is exactly
the state the solver chose to start from.
"""
import unittest

import numpy as np

from cmad.models.be_bar_elastic_plastic import (
    BeBarElasticPlastic,
    elastic_predictor,
)
from cmad.models.deformation_types import DefType
from cmad.models.global_fields import mp_U_from_F
from cmad.models.nonlinear_solver import newton_solve
from cmad.models.rate_elastic_plastic import RateElasticPlastic
from tests.support.test_problems import J2AnalyticalProblem

# past yield for the J2AnalyticalProblem material, with a nonzero deviator
_F = np.diag([1.01, 0.995, 0.995])


class TestMaterialPointInitialGuess(unittest.TestCase):

    def test_starts_from_the_elastic_predictor(self) -> None:
        params = J2AnalyticalProblem(scale_params=False).J2_parameters
        model = BeBarElasticPlastic(
            params, def_type=DefType.FULL_3D,
            initial_guess="elastic predictor")
        model.set_xi_to_init_vals()
        model.gather_global(mp_U_from_F(_F), mp_U_from_F(np.eye(3)))

        newton_solve(model, max_iters=0)

        expected = elastic_predictor(
            model.xi_prev(), model.xi_prev(), model.parameters.values,
            model._U, model._U_prev, DefType.FULL_3D, model._oop_stretch_idx,
        )
        for block, expected_block in zip(model.xi(), expected, strict=True):
            np.testing.assert_allclose(
                np.asarray(block), np.asarray(expected_block), rtol=1e-14)

        # the guess is a move, not a restatement of the previous state
        self.assertGreater(np.abs(np.asarray(model.xi()[0])).max(), 1e-3)

    def test_radial_return_start_needs_no_newton_pass_for_J2(self) -> None:
        params = J2AnalyticalProblem(scale_params=False).J2_parameters
        model = RateElasticPlastic(
            params, def_type=DefType.FULL_3D, initial_guess="radial return")
        model.set_xi_to_init_vals()
        model.gather_global(mp_U_from_F(_F), mp_U_from_F(np.eye(3)))

        newton_solve(model, max_iters=0)

        model.seed_none()
        model.evaluate()
        self.assertLess(np.linalg.norm(model.C()), 1e-12)
        self.assertGreater(float(model.xi()[1][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
