"""The rate model with elastic constants that depend on the temperature.

The small strain plastic model is a total form, exact in the elastic
constants at each step, so under a temperature ramp the rate model must
match it to the local Newton tolerance. A point loaded, heated at fixed
strain, unloaded, and cooled must end at zero stress.
"""
import unittest

import numpy as np

from cmad.io.params_builder import build_parameters
from cmad.models.deformation_types import DefType
from cmad.models.elastic_constants import compute_mu
from cmad.models.global_fields import StepTime, mp_U_from_F
from cmad.models.nonlinear_solver import newton_solve
from cmad.models.rate_elastic_plastic import RateElasticPlastic
from cmad.models.small_elastic_plastic import SmallElasticPlastic

_E_COEFFICIENTS = [250e3, -100.0]
_NU_COEFFICIENTS = [0.25, 1e-4]
_Y, _S, _D = 500.0, 200.0, 20.0
LOCAL_TOL = 1e-12
NUM_STEPS = 40


def _youngs_modulus(T):
    return _E_COEFFICIENTS[0] + _E_COEFFICIENTS[1] * T


def _parameters(nu_of_T):
    nu = {"polynomial": {"coefficients": _NU_COEFFICIENTS}} \
        if nu_of_T else _NU_COEFFICIENTS[0]
    return build_parameters({
        "elastic": {
            "E": {"polynomial": {"coefficients": _E_COEFFICIENTS}},
            "nu": nu,
        },
        "plastic": {
            "effective stress": {"J2": {}},
            "flow stress": {
                "initial yield": {"Y": _Y},
                "hardening": {"voce": {"S": _S, "D": _D}},
            },
        },
    })


def _drive(model, F, T):
    """The Cauchy stress and alpha histories through ``F`` at the
    temperatures ``T``, one per step."""
    num_steps = F.shape[2] - 1
    model.set_xi_to_init_vals()
    cauchy = np.zeros((3, 3, num_steps + 1))
    alpha = np.zeros(num_steps + 1)
    for step in range(1, num_steps + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step], T[step]),
            mp_U_from_F(F[:, :, step - 1], T[step - 1]))
        model.gather_time(StepTime(float(step), float(step - 1)))
        newton_solve(model, abs_tol=LOCAL_TOL, rel_tol=LOCAL_TOL)
        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
        alpha[step] = float(model.xi()[1][0])
        model.advance_xi()
    return cauchy, alpha


class TestRampAgainstTheSmallStrainModel(unittest.TestCase):

    def _check(self, def_type, F):
        T = np.linspace(300.0, 600.0, NUM_STEPS + 1)
        cauchy_rate, alpha_rate = _drive(
            RateElasticPlastic(_parameters(nu_of_T=True), def_type), F, T)
        cauchy_small, alpha_small = _drive(
            SmallElasticPlastic(_parameters(nu_of_T=True), def_type), F, T)
        self.assertGreater(alpha_small[-1], 0.0)

        # Both solves stop at LOCAL_TOL, which bounds their difference.
        E = _youngs_modulus(T.min())
        nu = _NU_COEFFICIENTS[0] + _NU_COEFFICIENTS[1] * T.min()
        stress_bound = 2.0 * LOCAL_TOL * 2.0 * compute_mu(E, nu)
        self.assertLess(
            np.abs(cauchy_rate - cauchy_small).max(), stress_bound)
        self.assertLess(
            np.abs(alpha_rate - alpha_small).max(), stress_bound / E)

    def test_full_3d(self):
        t = np.linspace(0.0, 1.0, NUM_STEPS + 1)
        F = np.repeat(np.eye(3)[:, :, None], NUM_STEPS + 1, axis=2)
        F[0, 0, :] += 8e-3 * t
        F[1, 1, :] += 2e-3 * t
        F[0, 1, :] += 6e-3 * t
        self._check(DefType.FULL_3D, F)

    def test_uniaxial_stress(self):
        strains = np.linspace(0.0, 3e-3, NUM_STEPS + 1)
        self._check(DefType.UNIAXIAL_STRESS, (1.0 + strains)[None, None, :])


class TestLoadHeatUnloadCool(unittest.TestCase):

    def test_returns_to_zero_stress(self):
        strain, T_cold, T_hot = 2e-3, 300.0, 600.0
        n = 10
        up, down = np.linspace(0.0, 1.0, n + 1), np.linspace(1.0, 0.0, n + 1)
        strains = np.concatenate([
            strain * up, np.full(n, strain), strain * down[1:], np.zeros(n)])
        T = np.concatenate([
            np.full(n + 1, T_cold), T_cold + (T_hot - T_cold) * up[1:],
            np.full(n, T_hot), T_hot + (T_cold - T_hot) * up[1:]])
        self.assertLess(strain * _youngs_modulus(T_cold), _Y)

        model = RateElasticPlastic(
            _parameters(nu_of_T=False), DefType.UNIAXIAL_STRESS)
        cauchy, alpha = _drive(model, (1.0 + strains)[None, None, :], T)

        self.assertEqual(alpha[-1], 0.0)
        heated = cauchy[0, 0, 2 * n]
        self.assertAlmostEqual(
            heated, _youngs_modulus(T_hot) * strain, delta=1e-6)
        nu = _NU_COEFFICIENTS[0]
        bound = 2.0 * LOCAL_TOL * 2.0 * compute_mu(_youngs_modulus(T_cold), nu)
        self.assertLess(abs(cauchy[0, 0, -1]), bound)


if __name__ == "__main__":
    unittest.main()
