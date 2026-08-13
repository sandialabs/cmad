"""The two Voce parameterizations describe the same curve.

``voce`` uses the saturation increment ``S``, ``voce_modulus`` the
initial hardening modulus ``H = S * D``. They have to agree wherever
``S = H / D``, and the modulus form has to reproduce the slope it is
named for.
"""
import unittest

import numpy as np
from jax import grad

from cmad.models.hardening import (
    get_hardening_funs,
    voce_hardening,
    voce_modulus_hardening,
)

_S, _D = 1000.0, 2.5


class TestVoceParameterizations(unittest.TestCase):

    def test_same_curve(self) -> None:
        alpha = np.linspace(0.0, 1.0, 21)
        increment = np.array(
            [voce_hardening(a, {"S": _S, "D": _D}) for a in alpha])
        modulus = np.array(
            [voce_modulus_hardening(a, {"H": _S * _D, "D": _D})
             for a in alpha])
        np.testing.assert_allclose(modulus, increment, rtol=1e-14)

    def test_H_is_the_slope_at_zero(self) -> None:
        slope = grad(voce_modulus_hardening)(0.0, {"H": _S * _D, "D": _D})
        self.assertAlmostEqual(float(slope), _S * _D, delta=1e-8 * _S * _D)

    def test_registered(self) -> None:
        self.assertIn("voce_modulus", get_hardening_funs())


if __name__ == "__main__":
    unittest.main()
