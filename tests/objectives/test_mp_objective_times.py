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
    PERZYNA,
    johnson_cook_flow_stress,
    perzyna_flow_stress,
    return_map,
    stress_bound,
)


def _parameters(flow_stress_params: dict, active_paths) -> Parameters:
    """The uniaxial stress J2 material on ``flow_stress_params``, with the
    leaves ``active_paths`` names active inside the ``flow stress`` subtree.
    """
    values = {
        "rotation matrix": np.eye(3),
        "elastic": {"E": 200e3, "nu": 0.3},
        "plastic": {
            "effective stress": {"J2": 0.0},
            "flow stress": flow_stress_params,
        },
    }
    active = tree_map(lambda a: False, values)
    for path in active_paths:
        node = active["plastic"]["flow stress"]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = True
    return Parameters(values, active, tree_map(lambda a: None, values))


_JOHNSON_COOK_ACTIVE = [("johnson_cook", name) for name in ("A", "B", "n", "C")]
_PERZYNA_ACTIVE = [
    ("perzyna", "initial yield", "Y"),
    ("perzyna", "hardening", "voce", "S"),
    ("perzyna", "hardening", "voce", "D"),
    ("perzyna", "eta"),
]


class TestObjectivesWithTimes(unittest.TestCase):

    def _check(self, flow_stress_params, active_paths, flow_stress) -> None:
        strains, times = two_rate_schedule()
        sigma_ref, _, _ = return_map(flow_stress, strains, times)
        data = np.zeros((3, 3, NUM_STEPS + 1))
        data[0, 0, :] = sigma_ref
        F = (1.0 + strains)[None, None, :]

        model = SmallElasticPlastic(
            _parameters(flow_stress_params, active_paths),
            DefType.UNIAXIAL_STRESS)
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

    def test_zero_at_the_truth_and_gradients_agree(self) -> None:
        self._check(
            JOHNSON_COOK, _JOHNSON_COOK_ACTIVE, johnson_cook_flow_stress)

    def test_perzyna_zero_at_the_truth_and_gradients_agree(self) -> None:
        """Perzyna with its viscosity active: eta is only identifiable
        against real step sizes, so this is the sensitivity path's check
        that the times reach the residual on every pass.
        """
        self._check(PERZYNA, _PERZYNA_ACTIVE, perzyna_flow_stress)


if __name__ == "__main__":
    unittest.main()
