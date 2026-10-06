"""The plastic dissipation at a material point.

The small strain models under uniaxial stress are checked step by step
against the scalar return map of ``test_rate_dependent_uniaxial.py``: the
dissipation is ``beta phi delta_gamma / dt`` with ``phi = |sigma_11|``,
zero while the point is elastic. The composite's heat generation is the
mechanics dissipation, and both are ``None`` for a material without
``taylor-quinney`` and for Elastic.
"""
import unittest

import numpy as np

from cmad.io.params_builder import build_parameters
from cmad.models.conduction import Conduction
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.global_fields import StepTime, mp_U_from_F
from cmad.models.nonlinear_solver import newton_solve
from cmad.models.rate_elastic_plastic import RateElasticPlastic
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.thermomechanics_model import ThermomechanicsModel
from tests.models.test_rate_dependent_uniaxial import (
    _E,
    _NU,
    JOHNSON_COOK,
    MAX_ITERS,
    PEAK_STRAIN,
    STRAIN_RATES,
    johnson_cook_flow_stress,
    return_map,
    schedule,
    stress_bound,
)

_BETA = 0.9
_T_REF = JOHNSON_COOK["johnson_cook"]["reference temperature"]


def _parameters(beta=_BETA, thermal=False):
    tree = {
        "elastic": {"E": _E, "nu": _NU},
        "plastic": {
            "effective stress": {"J2": {}},
            "flow stress": JOHNSON_COOK,
        },
    }
    if beta is not None:
        tree["plastic"]["taylor-quinney"] = beta
    if thermal:
        tree["thermal"] = {"conductivity": 16.0}
    return build_parameters(tree)


def _reference(strain_rate):
    """The return map's dissipation ``beta |sigma| delta_gamma / dt`` per
    step, and its bound from the local Newton tolerance."""
    strains, times = schedule(PEAK_STRAIN, strain_rate)
    sigma, alpha, _ = return_map(johnson_cook_flow_stress, strains, times)
    dt = np.diff(times)
    dissipation = np.zeros(len(strains))
    dissipation[1:] = _BETA * np.abs(sigma[1:]) * np.diff(alpha) / dt
    bound = _BETA * (stress_bound() * np.diff(alpha).max()
                     + np.abs(sigma).max() * stress_bound() / _E) / dt.min()
    return strains, times, dissipation, bound


def _history(model, heat_source, strains, times, T=None):
    """``heat_source``, a callback of the model, per step through the
    uniaxial schedule with the state solved by the local Newton."""
    model.set_xi_to_init_vals()
    params = model.parameters.values
    history = np.zeros(len(strains))
    for step in range(1, len(strains)):
        U = mp_U_from_F(np.array([[1.0 + strains[step]]]), T)
        U_prev = mp_U_from_F(np.array([[1.0 + strains[step - 1]]]), T)
        step_time = StepTime(times[step], times[step - 1])
        model.gather_global(U, U_prev)
        model.gather_time(step_time)
        iterations, _ = newton_solve(model, max_iters=MAX_ITERS)
        assert iterations < MAX_ITERS
        history[step] = float(heat_source(
            model.xi(), model.xi_prev(), params, U, U_prev, step_time))
        model.advance_xi()
    return history


class TestDissipationMatchesTheReturnMap(unittest.TestCase):

    def _check(self, Model):
        for strain_rate in STRAIN_RATES:
            strains, times, expected, bound = _reference(strain_rate)
            model = Model(_parameters(), DefType.UNIAXIAL_STRESS)
            self.assertIsNotNone(model.dissipation)
            dissipation = _history(model, model.dissipation, strains, times)
            # the schedule starts elastic, where the dissipation must vanish
            self.assertGreater((expected == 0.0).sum(), 1)
            self.assertLess(np.abs(dissipation - expected).max(), bound)

    def test_small_elastic_plastic(self):
        self._check(SmallElasticPlastic)

    def test_rate_elastic_plastic(self):
        self._check(RateElasticPlastic)


class TestCompositeHeatGeneration(unittest.TestCase):

    def test_is_the_mechanics_dissipation(self):
        strains, times, expected, bound = _reference(STRAIN_RATES[0])
        parameters = _parameters(thermal=True)
        mechanics = SmallElasticPlastic(
            parameters, DefType.UNIAXIAL_STRESS, reference_temperature=_T_REF)
        composite = ThermomechanicsModel(
            mechanics, Conduction(parameters, reference_temperature=_T_REF))
        self.assertIsNotNone(composite.dissipation)
        self.assertIsNotNone(composite.heat_generation)
        generation = _history(
            composite, composite.heat_generation, strains, times, T=_T_REF)
        self.assertLess(np.abs(generation - expected).max(), bound)

    def test_none_without_taylor_quinney(self):
        parameters = _parameters(beta=None, thermal=True)
        mechanics = SmallElasticPlastic(parameters, DefType.UNIAXIAL_STRESS)
        self.assertIsNone(mechanics.dissipation)
        composite = ThermomechanicsModel(mechanics, Conduction(parameters))
        self.assertIsNone(composite.dissipation)
        self.assertIsNone(composite.heat_generation)

    def test_none_for_elastic(self):
        parameters = _parameters(thermal=True)
        composite = ThermomechanicsModel(
            Elastic(parameters), Conduction(parameters))
        self.assertIsNone(composite.dissipation)
        self.assertIsNone(composite.heat_generation)


if __name__ == "__main__":
    unittest.main()
