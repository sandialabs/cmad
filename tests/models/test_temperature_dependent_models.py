"""The models with a parameter written as a function of temperature.

Each check has an independent reference evaluated by hand at the point's
temperature: ``E(T) eps`` for the elastic model, the uniaxial return map
with ``Y(T)`` for the small strain plastic model, and ``-k(T) grad T`` for
conduction.
"""
import unittest

import numpy as np
from jax import numpy as jnp

from cmad.io.params_builder import build_parameters
from cmad.models.conduction import Conduction
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.elastic_constants import compute_mu
from cmad.models.global_fields import GlobalFieldsAtPoint, StepTime, mp_U_from_F
from cmad.models.nonlinear_solver import newton_solve
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from tests.models.test_rate_dependent_uniaxial import _E as _RETURN_MAP_E
from tests.models.test_rate_dependent_uniaxial import return_map, schedule

_NU = 0.3
_E_COEFFICIENTS = [250e3, -100.0]
_Y_KNOTS = [300.0, 600.0]
_Y_VALUES = [200.0, 120.0]
_S, _D = 200.0, 20.0
_K_COEFFICIENTS = [10.0, 0.02]
LOCAL_TOL = 1e-12


def _youngs_modulus(T):
    return _E_COEFFICIENTS[0] + _E_COEFFICIENTS[1] * T


def _Y(T):
    return np.interp(T, _Y_KNOTS, _Y_VALUES)


def _elastic_parameters():
    return build_parameters({
        "elastic": {
            "E": {"polynomial": {"coefficients": _E_COEFFICIENTS}},
            "nu": _NU,
        },
    })


def _plastic_parameters():
    return build_parameters({
        "elastic": {"E": _RETURN_MAP_E, "nu": _NU},
        "plastic": {
            "effective stress": {"J2": {}},
            "flow stress": {
                "initial yield": {
                    "Y": {"table": {"T": _Y_KNOTS, "values": _Y_VALUES}}},
                "hardening": {"voce": {"S": _S, "D": _D}},
            },
        },
    })


def _drive(model, F, T, times=None):
    """The Cauchy stress history of the point through ``F`` at ``T``."""
    num_steps = F.shape[2] - 1
    if times is None:
        times = np.arange(num_steps + 1, dtype=float)
    model.set_xi_to_init_vals()
    cauchy = np.zeros((3, 3, num_steps + 1))
    for step in range(1, num_steps + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step], T), mp_U_from_F(F[:, :, step - 1], T))
        model.gather_time(StepTime(times[step], times[step - 1]))
        newton_solve(model, abs_tol=LOCAL_TOL, rel_tol=LOCAL_TOL)
        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
        model.advance_xi()
    return cauchy


class TestElasticModulusOfTemperature(unittest.TestCase):

    def test_uniaxial_stress_is_E_of_T_times_strain(self):
        eps = 1e-3
        F = np.repeat(np.eye(3)[:, :, None], 2, axis=2)
        F[0, 0, 1] += eps
        model = Elastic(_elastic_parameters(), def_type=DefType.UNIAXIAL_STRESS)
        for T in (300.0, 500.0, 800.0):
            cauchy = _drive(model, F, T)[:, :, 1]
            # Both the model and the reference are exact at convergence, so
            # the difference is bounded by the local Newton tolerance.
            E = _youngs_modulus(T)
            bound = 2.0 * LOCAL_TOL * 2.0 * compute_mu(E, _NU)
            self.assertLess(abs(cauchy[0, 0] - E * eps), bound)
            self.assertLess(abs(cauchy[1, 1]), bound)
            self.assertLess(abs(cauchy[2, 2]), bound)


class TestYieldStressOfTemperature(unittest.TestCase):

    def test_uniaxial_return_map_with_Y_of_T(self):
        peak_strain, strain_rate = 3e-3, 0.01
        strains, times = schedule(peak_strain, strain_rate)
        F = (1.0 + strains)[None, None, :]
        model = SmallElasticPlastic(
            _plastic_parameters(), DefType.UNIAXIAL_STRESS)
        for T in (300.0, 450.0):

            def flow_stress(alpha, rate, T=T):
                return _Y(T) + _S * (1.0 - np.exp(-_D * alpha))

            sigma_ref, _alpha_ref, _ = return_map(flow_stress, strains, times)
            cauchy = _drive(model, F, T, times)
            self.assertGreater(abs(sigma_ref[-1]), _Y(T))
            bound = 2.0 * LOCAL_TOL * 2.0 * compute_mu(_RETURN_MAP_E, _NU)
            self.assertLess(np.abs(cauchy[0, 0, :] - sigma_ref).max(), bound)


class TestConductivityOfTemperature(unittest.TestCase):

    def _fields(self, T, grad_T):
        return GlobalFieldsAtPoint(
            fields={"T": jnp.array([T])},
            grad_fields={"T": jnp.asarray(grad_T)[None, :]})

    def test_flux_is_minus_k_of_T_grad_T(self):
        model = Conduction(build_parameters({"thermal": {
            "conductivity": {"polynomial": {"coefficients": _K_COEFFICIENTS}}}}))
        params = model.parameters.values
        grad_T = np.array([1.0, 2.0, 3.0])
        for T in (300.0, 700.0):
            k = _K_COEFFICIENTS[0] + _K_COEFFICIENTS[1] * T
            U = self._fields(T, grad_T)
            np.testing.assert_allclose(
                model.heat_flux_closed_form(params, U, U), -k * grad_T)
            np.testing.assert_allclose(
                model.heat_flux([], [], params, U, U), -k * grad_T)

    def test_no_temperature_field_uses_the_reference_temperature(self):
        reference_temperature = 450.0
        model = Conduction(
            build_parameters({"thermal": {"conductivity": {
                "polynomial": {"coefficients": _K_COEFFICIENTS}}}}),
            reference_temperature=reference_temperature)
        grad_T = np.array([1.0, 2.0, 3.0])
        U = GlobalFieldsAtPoint(
            fields={}, grad_fields={"T": jnp.asarray(grad_T)[None, :]})
        k = _K_COEFFICIENTS[0] + _K_COEFFICIENTS[1] * reference_temperature
        np.testing.assert_allclose(
            model.heat_flux_closed_form(model.parameters.values, U, U),
            -k * grad_T)


if __name__ == "__main__":
    unittest.main()
