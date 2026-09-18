"""The yield function of the plastic models.

The rate-independent relation reproduces the expression the models
computed inline, ``phi - (Y + hardening(alpha))``, to the bit, and
``make_yield_function`` accepts every existing parameter tree and nothing
else. Johnson-Cook is checked by hand on every branch of its clips; its
slope at zero strain is also checked. Peric's rate law holds at the zero
of its yield function.
"""
import unittest

import numpy as np
from jax import grad
from scipy.optimize import brentq

from cmad.models.elastic_stress import two_mu_scale_factor
from cmad.models.flow_stress import POWER_LAW_OFFSET, make_yield_function
from cmad.models.hardening import voce_hardening
from cmad.models.paths import yield_threshold
from tests.support.test_problems import params_J2_voce

_E, _NU, _Y, _S, _D = 200e3, 0.3, 200.0, 200.0, 20.0


def _flow_params():
    return {
        "initial yield": {"Y": _Y},
        "hardening": {"voce": {"S": _S, "D": _D}},
    }


class TestRateIndependentYieldFunction(unittest.TestCase):

    def test_matches_the_inline_expression_bitwise(self) -> None:
        yield_function = make_yield_function(_flow_params())
        for phi in (0.0, 150.0, 400.0):
            for alpha in np.linspace(0.0, 0.5, 6):
                expected = phi - (_Y + voce_hardening(alpha, {"S": _S, "D": _D}))
                value = yield_function(phi, alpha, 0.0, None, _flow_params())
                self.assertEqual(float(value), float(expected))

    def test_accepts_every_existing_tree(self) -> None:
        flat_values = np.array([_E, _NU, _Y, _S, _D])
        for parameters in params_J2_voce(flat_values, False):
            make_yield_function(parameters.values["plastic"]["flow stress"])

    def test_rejects_an_unknown_key(self) -> None:
        flow_params = _flow_params()
        flow_params["rate factor"] = {}
        with self.assertRaises(ValueError):
            make_yield_function(flow_params)

    def test_rejects_a_missing_hardening(self) -> None:
        with self.assertRaises(ValueError):
            make_yield_function({"initial yield": {"Y": _Y}})

    def test_threshold_is_relative_to_the_initial_yield(self) -> None:
        flat_values = np.array([_E, _NU, _Y, _S, _D])
        params = params_J2_voce(flat_values, False)[0].values
        flow_params = params["plastic"]["flow stress"]
        yield_function = make_yield_function(flow_params)
        yield_tol = 1e-12
        expected = yield_tol * flow_params["initial yield"]["Y"] \
            / two_mu_scale_factor(params)
        threshold = yield_threshold(yield_tol, params, yield_function)
        self.assertEqual(float(threshold), float(expected))


# The 4340 steel constants of Johnson and Cook 1983, here only as numbers
# for the checks by hand.
_JC = {
    "johnson_cook": {
        "A": 792.0, "B": 510.0, "n": 0.26, "C": 0.014,
        "reference rate": 1.0,
        "reference temperature": 294.0, "melt temperature": 1793.0,
        "m": 1.03,
    },
}
_PHI = 900.0


def _johnson_cook_terms(alpha, alpha_dot, T):
    """The three factors of the flow stress by hand."""
    p = _JC["johnson_cook"]
    a0 = POWER_LAW_OFFSET
    strain_hardening = p["A"] + p["B"] * (alpha + a0) ** p["n"]
    rate_dependence = 1.0 + p["C"] * np.log(
        max(alpha_dot / p["reference rate"], 1.0))
    T_star = (T - p["reference temperature"]) \
        / (p["melt temperature"] - p["reference temperature"])
    T_star = min(max(T_star, 0.0), 1.0)
    thermal_softening = max(1.0 - (T_star + a0) ** p["m"], 0.0)
    return strain_hardening, rate_dependence, thermal_softening


class TestJohnsonCookYieldFunction(unittest.TestCase):

    def test_value_by_hand_on_every_branch(self) -> None:
        yield_function = make_yield_function(_JC)
        for alpha_dot in (10.0, 0.1, -1.0):
            for T in (100.0, 294.0, 500.0, 1793.0, None):
                value = yield_function(_PHI, 0.1, alpha_dot, T, _JC)
                expected = _PHI - float(np.prod(_johnson_cook_terms(
                    0.1, alpha_dot, 294.0 if T is None else T)))
                np.testing.assert_allclose(float(value), expected, rtol=1e-14)

    def test_slope_at_zero_strain_is_the_offset_slope(self) -> None:
        yield_function = make_yield_function(_JC)
        p = _JC["johnson_cook"]
        a0 = POWER_LAW_OFFSET
        _, rate_dependence, thermal_softening = _johnson_cook_terms(
            0.0, 10.0, 500.0)
        expected = -p["B"] * p["n"] * a0 ** (p["n"] - 1.0) \
            * rate_dependence * thermal_softening
        slope = grad(lambda a: yield_function(_PHI, a, 10.0, 500.0, _JC))(0.0)
        np.testing.assert_allclose(float(slope), expected, rtol=1e-12)


_PERIC = {"peric": {**_flow_params(), "eta": 10.0, "epsilon": 0.2}}


class TestPericYieldFunction(unittest.TestCase):

    def test_rate_law_holds_at_the_zero(self) -> None:
        yield_function = make_yield_function(_PERIC)
        p = _PERIC["peric"]
        alpha, alpha_dot = 0.05, 2.0
        sigma_y = float(_Y + voce_hardening(alpha, {"S": _S, "D": _D}))
        phi = brentq(
            lambda phi: float(yield_function(phi, alpha, alpha_dot, None, _PERIC)),
            sigma_y, 10.0 * sigma_y)
        rate = ((phi / sigma_y) ** (1.0 / p["epsilon"]) - 1.0) / p["eta"]
        np.testing.assert_allclose(rate, alpha_dot, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
