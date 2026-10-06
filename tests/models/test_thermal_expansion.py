"""The thermal stretch in every mechanics model.

Every reference is by hand: the free expansion ``F = theta I`` is stress
free; the stress at a constrained point is ``-3 kappa alpha dT I`` in small
strain and ``kappa/2 (J_e - 1/J_e) I`` with ``J_e = theta^-3`` in finite
deformation; in a constrained bar it is ``-E alpha dT``. The plastic models
are checked against the uniaxial return map on the mechanical strain and
against each other, and Johnson-Cook at a point with no temperature field
sits at the simulation's reference temperature.
"""
import unittest

import numpy as np

from cmad.io.params_builder import build_parameters
from cmad.models.be_bar_elastic_plastic import BeBarElasticPlastic
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.elastic_constants import compute_kappa, compute_mu
from cmad.models.elastic_stress import compressible_neohookean_cauchy_stress
from cmad.models.flow_stress import POWER_LAW_OFFSET, make_yield_function
from cmad.models.global_fields import StepTime, mp_U_from_F
from cmad.models.nonlinear_solver import newton_solve
from cmad.models.paths import compute_yield_threshold
from cmad.models.rate_elastic_plastic import RateElasticPlastic
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from tests.models.test_rate_dependent_uniaxial import (
    _E,
    JOHNSON_COOK,
    return_map,
    schedule,
)

_NU = 0.3
_ALPHA = 1.2e-5
_T_REF = 300.0
_T_HOT = 500.0
_Y_KNOTS = [300.0, 600.0]
_Y_VALUES = [200.0, 120.0]
_S, _D = 200.0, 20.0
LOCAL_TOL = 1e-12
# Both the model and the reference are exact at convergence, so a stress
# agrees to the local Newton tolerance times the residual scale 2 mu.
_STRESS_BOUND = 2.0 * LOCAL_TOL * 2.0 * compute_mu(_E, _NU)
_KAPPA = compute_kappa(_E, _NU)
_I = np.eye(3)


def _expansion(alpha=_ALPHA, T0=_T_REF):
    return {"alpha": alpha, "reference temperature": T0}


def _voce(Y):
    return {"initial yield": {"Y": Y},
            "hardening": {"voce": {"S": _S, "D": _D}}}


_ELASTIC_YIELD = _voce(2000.0)  # never reached in the checks that use it


def _parameters(flow_stress=None, expansion=_expansion(),
                elastic=None):
    tree = {"elastic": elastic or {"E": _E, "nu": _NU}}
    if expansion is not None:
        tree["thermal expansion"] = expansion
    if flow_stress is not None:
        tree["plastic"] = {"effective stress": {"J2": {}},
                           "flow stress": flow_stress}
    return build_parameters(tree)


def _thermal_stretch(T, alpha=_ALPHA):
    return 1.0 + alpha * (T - _T_REF)


def _history(F_of_step, num_steps):
    """``(n, n, num_steps + 1)`` from a function of the step."""
    return np.stack([F_of_step(k) for k in range(num_steps + 1)], axis=2)


def _drive(model, F, T, times=None):
    """The Cauchy stress history through ``F`` at the temperatures ``T``
    (``None`` for no temperature field, one number, or one per step) and
    the final local state."""
    num_steps = F.shape[2] - 1
    if T is None:
        T_steps = [None] * (num_steps + 1)
    else:
        T_steps = np.broadcast_to(np.asarray(T, dtype=float), (num_steps + 1,))
    if times is None:
        times = np.arange(num_steps + 1, dtype=float)
    model.set_xi_to_init_vals()
    cauchy = np.zeros((3, 3, num_steps + 1))
    for step in range(1, num_steps + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step], T_steps[step]),
            mp_U_from_F(F[:, :, step - 1], T_steps[step - 1]))
        model.gather_time(StepTime(times[step], times[step - 1]))
        newton_solve(model, abs_tol=LOCAL_TOL, rel_tol=LOCAL_TOL)
        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
        model.advance_xi()
    return cauchy, [np.asarray(block) for block in model.xi()]


class TestElastic(unittest.TestCase):

    def test_free_expansion_is_stress_free(self):
        theta = _thermal_stretch(_T_HOT)
        F = _history(lambda k: _I if k == 0 else theta * _I, 1)
        cauchy, _ = _drive(Elastic(_parameters(), reference_temperature=_T_REF),
                           F, _T_HOT)
        self.assertLess(np.abs(cauchy[:, :, 1]).max(), _STRESS_BOUND)

    def test_constrained_expansion_is_hydrostatic(self):
        F = _history(lambda k: _I, 1)
        cauchy, _ = _drive(Elastic(_parameters(), reference_temperature=_T_REF),
                           F, _T_HOT)
        expected = -3.0 * _KAPPA * _ALPHA * (_T_HOT - _T_REF)
        self.assertLess(
            np.abs(cauchy[:, :, 1] - expected * _I).max(), _STRESS_BOUND)

    def test_constrained_bar_under_uniaxial_stress(self):
        F = np.ones((1, 1, 2))
        model = Elastic(_parameters(), def_type=DefType.UNIAXIAL_STRESS,
                        reference_temperature=_T_REF)
        cauchy, xi = _drive(model, F, _T_HOT)
        dT = _T_HOT - _T_REF
        self.assertLess(abs(cauchy[0, 0, 1] + _E * _ALPHA * dT), _STRESS_BOUND)
        self.assertLess(abs(cauchy[1, 1, 1]), _STRESS_BOUND)
        self.assertLess(abs(cauchy[2, 2, 1]), _STRESS_BOUND)
        # the off axis stretches are set by the zero lateral stresses
        np.testing.assert_allclose(
            xi[1], 1.0 + (1.0 + _NU) * _ALPHA * dT, rtol=0.0,
            atol=_STRESS_BOUND / _E)

    def test_plane_stress_free_expansion(self):
        theta = _thermal_stretch(_T_HOT)
        F = _history(lambda k: np.eye(2) if k == 0 else theta * np.eye(2), 1)
        model = Elastic(_parameters(), def_type=DefType.PLANE_STRESS,
                        reference_temperature=_T_REF)
        cauchy, xi = _drive(model, F, _T_HOT)
        self.assertLess(np.abs(cauchy[:, :, 1]).max(), _STRESS_BOUND)
        self.assertAlmostEqual(
            float(xi[1][0]), theta, delta=_STRESS_BOUND / _E)

    def test_plane_strain_in_plane_free_expansion(self):
        dT = _T_HOT - _T_REF
        in_plane = 1.0 + (1.0 + _NU) * _ALPHA * dT
        F = _history(lambda k: np.eye(2) if k == 0 else in_plane * np.eye(2), 1)
        model = Elastic(_parameters(), def_type=DefType.PLANE_STRAIN,
                        reference_temperature=_T_REF)
        cauchy, _ = _drive(model, F, _T_HOT)
        self.assertLess(np.abs(cauchy[:2, :2, 1]).max(), _STRESS_BOUND)
        self.assertLess(abs(cauchy[2, 2, 1] + _E * _ALPHA * dT), _STRESS_BOUND)


class TestCoefficientOfTemperature(unittest.TestCase):

    def test_polynomial_alpha_with_its_own_reference_temperature(self):
        a0, a1, T0, T_ref = 1.0e-5, 1.0e-8, 300.0, 400.0
        expansion = {"alpha": {"polynomial": {"coefficients": [a0, a1]}},
                     "reference temperature": T0}
        model = Elastic(_parameters(expansion=expansion),
                        def_type=DefType.UNIAXIAL_STRESS,
                        reference_temperature=T_ref)

        def length(T):
            return 1.0 + (a0 + a1 * T) * (T - T0)

        for T in (350.0, 500.0):
            cauchy, _ = _drive(model, np.ones((1, 1, 2)), T)
            stretch = length(T) / length(T_ref)
            self.assertLess(
                abs(cauchy[0, 0, 1] + _E * (stretch - 1.0)), _STRESS_BOUND)

    def test_missing_key_raises(self):
        for expansion in ({"alpha": _ALPHA},
                          {"reference temperature": _T_REF}):
            with self.assertRaisesRegex(ValueError, "thermal expansion"):
                Elastic(_parameters(expansion=expansion))


class TestSmallStrainPlastic(unittest.TestCase):

    def test_return_map_on_the_mechanical_strain(self):
        flow_stress = _voce({"table": {"T": _Y_KNOTS, "values": _Y_VALUES}})
        model = SmallElasticPlastic(
            _parameters(flow_stress), DefType.UNIAXIAL_STRESS,
            reference_temperature=_T_REF)
        mechanical, times = schedule(3e-3, 0.01)
        for T_hot in (450.0, 550.0):
            Y = np.interp(T_hot, _Y_KNOTS, _Y_VALUES)

            def flow(alpha, rate, Y=Y):
                return Y + _S * (1.0 - np.exp(-_D * alpha))

            sigma_ref, _, _ = return_map(flow, mechanical, times)
            self.assertGreater(abs(sigma_ref[-1]), Y)
            total = _ALPHA * (T_hot - _T_REF) + mechanical
            cauchy, _ = _drive(model, (1.0 + total)[None, None, :], T_hot, times)
            self.assertLess(
                np.abs(cauchy[0, 0, :] - sigma_ref).max(), _STRESS_BOUND)


def _parameters_of_temperature(Y):
    return _parameters(
        _voce(Y),
        expansion={"alpha": {"polynomial": {"coefficients": [1.0e-5, 1.0e-8]}},
                   "reference temperature": _T_REF},
        elastic={"E": {"polynomial": {"coefficients": [250e3, -100.0]}},
                 "nu": {"polynomial": {"coefficients": [0.25, 1e-4]}}})


class TestRateModel(unittest.TestCase):

    def _check_against_the_small_strain_model(self, def_type, F, T):
        cauchy_rate, xi_rate = _drive(
            RateElasticPlastic(_parameters_of_temperature(500.0), def_type,
                               reference_temperature=_T_REF), F, T)
        cauchy_small, xi_small = _drive(
            SmallElasticPlastic(_parameters_of_temperature(500.0), def_type,
                                reference_temperature=_T_REF), F, T)
        self.assertGreater(float(xi_small[1][0]), 0.0)
        E, nu = 250e3 - 100.0 * T.min(), 0.25 + 1e-4 * T.min()
        bound = 2.0 * LOCAL_TOL * 2.0 * compute_mu(E, nu)
        self.assertLess(np.abs(cauchy_rate - cauchy_small).max(), bound)
        self.assertLess(abs(float(xi_rate[1][0] - xi_small[1][0])), bound / E)

    def test_ramp_full_3d(self):
        n = 40
        t = np.linspace(0.0, 1.0, n + 1)
        F = np.repeat(_I[:, :, None], n + 1, axis=2)
        F[0, 0, :] += 8e-3 * t
        F[1, 1, :] += 2e-3 * t
        F[0, 1, :] += 6e-3 * t
        self._check_against_the_small_strain_model(
            DefType.FULL_3D, F, np.linspace(300.0, 600.0, n + 1))

    def test_ramp_constrained_bar(self):
        n = 40
        self._check_against_the_small_strain_model(
            DefType.UNIAXIAL_STRESS, np.ones((1, 1, n + 1)),
            np.linspace(300.0, 600.0, n + 1))

    def test_constrained_heat_and_cool_returns_to_zero_stress(self):
        n = 20
        T = np.concatenate([np.linspace(300.0, 600.0, n + 1),
                            np.linspace(600.0, 300.0, n + 1)[1:]])
        model = RateElasticPlastic(
            _parameters(_ELASTIC_YIELD, expansion={
                "alpha": {"polynomial": {"coefficients": [1.0e-5, 1.0e-8]}},
                "reference temperature": _T_REF}),
            DefType.UNIAXIAL_STRESS, reference_temperature=_T_REF)
        cauchy, xi = _drive(model, np.ones((1, 1, 2 * n + 1)), T)
        self.assertEqual(float(xi[1][0]), 0.0)
        self.assertGreater(abs(cauchy[0, 0, n]), 500.0)
        self.assertLess(abs(cauchy[0, 0, -1]), _STRESS_BOUND)


class TestFiniteDeformation(unittest.TestCase):

    def _check_free_and_constrained(self, model):
        theta = _thermal_stretch(_T_HOT)
        free, _ = _drive(
            model, _history(lambda k: _I if k == 0 else theta * _I, 1), _T_HOT)
        self.assertLess(np.abs(free[:, :, 1]).max(), _STRESS_BOUND)
        constrained, xi = _drive(model, _history(lambda k: _I, 1), _T_HOT)
        J_e = theta ** -3
        expected = 0.5 * _KAPPA * (J_e - 1.0 / J_e)
        self.assertLess(
            np.abs(constrained[:, :, 1] - expected * _I).max(), _STRESS_BOUND)
        return xi

    def test_neohookean_elastic(self):
        self._check_free_and_constrained(Elastic(
            _parameters(), elastic_stress_fun=compressible_neohookean_cauchy_stress,
            reference_temperature=_T_REF))

    def test_be_bar(self):
        xi = self._check_free_and_constrained(BeBarElasticPlastic(
            _parameters(_ELASTIC_YIELD), reference_temperature=_T_REF))
        self.assertEqual(float(xi[2][0]), 0.0)

    def test_finite_rate_model(self):
        n = 10
        T = np.linspace(_T_REF, _T_HOT, n + 1)
        model = RateElasticPlastic(
            _parameters(_ELASTIC_YIELD), finite_deformation=True,
            reference_temperature=_T_REF)
        free, _ = _drive(
            model, _history(lambda k: _thermal_stretch(T[k]) * _I, n), T)
        self.assertLess(np.abs(free).max(), _STRESS_BOUND)
        constrained, _ = _drive(model, _history(lambda k: _I, n), T)
        expected = -3.0 * _KAPPA * np.log(_thermal_stretch(_T_HOT))
        self.assertLess(
            np.abs(constrained[:, :, -1] - expected * _I).max(), _STRESS_BOUND)


class TestJohnsonCook(unittest.TestCase):

    def _drive_johnson_cook(self, reference_temperature, T):
        model = SmallElasticPlastic(
            _parameters(JOHNSON_COOK, expansion=None), DefType.UNIAXIAL_STRESS,
            reference_temperature=reference_temperature)
        strains, times = schedule(3e-3, 0.01)
        return _drive(model, (1.0 + strains)[None, None, :], T, times)[0][0, 0]

    def test_no_temperature_field_sits_at_the_reference_temperature(self):
        at_reference = self._drive_johnson_cook(500.0, None)
        with_field = self._drive_johnson_cook(500.0, 500.0)
        np.testing.assert_allclose(at_reference, with_field, rtol=1e-12)
        # at 294 K the softening factor is one, so the stress is higher;
        # this shows the two runs above see the 500 K
        cold = self._drive_johnson_cook(294.0, None)
        self.assertLess(at_reference[-1], cold[-1] - 10.0)

    def test_threshold_at_the_reference_temperature(self):
        p = JOHNSON_COOK["johnson_cook"]
        a0 = POWER_LAW_OFFSET
        T_star = (500.0 - p["reference temperature"]) \
            / (p["melt temperature"] - p["reference temperature"])
        flow_stress = (p["A"] + p["B"] * a0 ** p["n"]) \
            * (1.0 - (T_star + a0) ** p["m"])
        shear_scale_factor = 2.0 * compute_mu(_E, _NU)
        threshold = compute_yield_threshold(
            1e-12, {"plastic": {"flow stress": JOHNSON_COOK}},
            make_yield_function(JOHNSON_COOK), shear_scale_factor, 500.0)
        self.assertAlmostEqual(
            threshold, 1e-12 * flow_stress / shear_scale_factor, delta=1e-24)


if __name__ == "__main__":
    unittest.main()
