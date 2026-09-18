"""The four sensitivity objectives with non-uniform step times.

The data is the return map's stresses on the two rate schedule, so each
objective gives J of zero at the truth, and at offset parameters the
gradients of the three numpy objectives agree with the gradient of the
traced objective.
"""
import unittest

import numpy as np
from jax.tree_util import tree_map

from cmad.models.deformation_types import DefType
from cmad.models.nonlinear_solver import make_newton_solve
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.objectives.mp_jvp_objective import MPJVPObjective
from cmad.objectives.mp_objective import (
    MPAdjointObjective,
    MPDirectAdjointObjective,
    MPDirectObjective,
)
from cmad.parameters.parameters import Parameters
from cmad.qois.calibration import Calibration
from tests.cli.test_primal_roundtrip import two_rate_schedule
from tests.models.test_rate_dependent_uniaxial import (
    JOHNSON_COOK,
    NUM_STEPS,
    johnson_cook_flow_stress,
    return_map,
    stress_bound,
)


def _parameters() -> Parameters:
    values = {
        "rotation matrix": np.eye(3),
        "elastic": {"E": 200e3, "nu": 0.3},
        "plastic": {
            "effective stress": {"J2": 0.0},
            "flow stress": JOHNSON_COOK,
        },
    }
    active = tree_map(lambda a: False, values)
    for name in ("A", "B", "n", "C"):
        active["plastic"]["flow stress"]["johnson_cook"][name] = True
    return Parameters(values, active, tree_map(lambda a: None, values))


class TestObjectivesWithTimes(unittest.TestCase):

    def test_zero_at_the_truth_and_gradients_agree(self) -> None:
        strains, times = two_rate_schedule()
        sigma_ref, _, _ = return_map(johnson_cook_flow_stress, strains, times)
        data = np.zeros((3, 3, NUM_STEPS + 1))
        data[0, 0, :] = sigma_ref
        F = (1.0 + strains)[None, None, :]

        model = SmallElasticPlastic(_parameters(), DefType.UNIAXIAL_STRESS)
        qoi = Calibration(model, data, np.eye(3))
        truth = model.parameters.flat_active_values(True)
        offset = 1.1 * truth
        J_bound = NUM_STEPS * stress_bound() ** 2

        traced = MPJVPObjective(
            qoi, F, make_newton_solve(model._residual), times)
        self.assertLess(float(traced.evaluate_objective(truth)), J_bound)
        _, traced_grad = traced.evaluate_objective_and_grad(offset)

        for Objective in (
                MPAdjointObjective, MPDirectObjective,
                MPDirectAdjointObjective):
            objective = Objective(qoi, F, times)
            self.assertLess(objective.evaluate(truth).J, J_bound)
            np.testing.assert_allclose(
                objective.evaluate(offset).grad, np.asarray(traced_grad),
                rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
