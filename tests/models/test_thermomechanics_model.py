"""The composite of a mechanics model and a thermal model on one block."""
import unittest

import numpy as np
from jax import numpy as jnp

from cmad.io.params_builder import build_parameters
from cmad.models.conduction import Conduction
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.global_fields import GlobalFieldsAtPoint, StepTime, mp_U_from_F
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.thermomechanics_model import ThermomechanicsModel

_T_REF = 300.0
_STEP_TIME = StepTime(1.0, 0.0)


def _parameters():
    return build_parameters({
        "elastic": {"E": 200e3, "nu": 0.3},
        "thermal expansion": {"alpha": 1.2e-5, "reference temperature": _T_REF},
        "plastic": {
            "effective stress": {"J2": {}},
            "flow stress": {"initial yield": {"Y": 200.0},
                            "hardening": {"voce": {"S": 200.0, "D": 20.0}}},
        },
        "thermal": {"conductivity": 16.0},
    })


def _plastic_composite(initial_guess=None):
    parameters = _parameters()
    mechanics = SmallElasticPlastic(
        parameters, DefType.FULL_3D, initial_guess=initial_guess,
        reference_temperature=_T_REF)
    thermal = Conduction(parameters, reference_temperature=_T_REF)
    return mechanics, ThermomechanicsModel(mechanics, thermal)


class TestThermomechanicsModel(unittest.TestCase):

    def test_state_residual_and_stress_are_the_mechanics_models(self):
        mechanics, composite = _plastic_composite(initial_guess="radial return")
        self.assertEqual(composite.num_residuals, mechanics.num_residuals)
        self.assertEqual(composite.var_names, mechanics.var_names)
        self.assertEqual(composite.resid_names, mechanics.resid_names)
        for block, expected in zip(composite._init_xi, mechanics._init_xi, strict=True):
            np.testing.assert_array_equal(block, expected)

        F = np.eye(3)
        F[0, 0] += 2e-3
        U, U_prev = mp_U_from_F(F, 450.0), mp_U_from_F(np.eye(3), _T_REF)
        xi_prev = [jnp.asarray(block) for block in composite._init_xi]
        xi = [block + 1e-3 for block in xi_prev]
        params = composite.parameters.values
        np.testing.assert_array_equal(
            composite._residual(xi, xi_prev, params, U, U_prev, _STEP_TIME),
            mechanics._residual(xi, xi_prev, params, U, U_prev, _STEP_TIME))
        np.testing.assert_array_equal(
            composite.cauchy(xi, xi_prev, params, U, U_prev),
            mechanics.cauchy(xi, xi_prev, params, U, U_prev))
        assert composite.initial_guess_fn is not None
        assert mechanics.initial_guess_fn is not None
        for block, expected in zip(
                composite.initial_guess_fn(xi_prev, params, U, U_prev, _STEP_TIME),
                mechanics.initial_guess_fn(xi_prev, params, U, U_prev, _STEP_TIME),
                strict=True):
            np.testing.assert_array_equal(block, expected)

    def test_heat_flux_and_capacity_rate_are_the_thermal_models(self):
        _, composite = _plastic_composite()
        grad_T = np.array([1.0, -2.0, 0.5])
        U = GlobalFieldsAtPoint(
            fields={"u": jnp.zeros(3), "T": jnp.array([350.0])},
            grad_fields={"u": jnp.zeros((3, 3)), "T": jnp.asarray(grad_T)[None, :]})
        params = composite.parameters.values
        xi = [jnp.asarray(block) for block in composite._init_xi]
        np.testing.assert_allclose(
            composite.heat_flux(xi, xi, params, U, U), -16.0 * grad_T)
        np.testing.assert_allclose(
            composite.heat_flux_closed_form(params, U, U), -16.0 * grad_T)
        self.assertEqual(
            float(composite.heat_capacity_rate(params, U, U, _STEP_TIME)), 0.0)

    def test_closed_form_flag_is_the_conjunction(self):
        parameters = _parameters()
        elastic = ThermomechanicsModel(
            Elastic(parameters, reference_temperature=_T_REF),
            Conduction(parameters, reference_temperature=_T_REF))
        self.assertTrue(elastic.supports_closed_form)
        self.assertIsNotNone(elastic.cauchy_closed_form)
        _, plastic = _plastic_composite()
        self.assertFalse(plastic.supports_closed_form)
        self.assertTrue(plastic.supports_mixed)

    def test_derived_outputs_are_both(self):
        _, composite = _plastic_composite()
        self.assertEqual(
            composite.derived_output_field_names(), ["cauchy", "heat flux"])

    def test_two_parameters_objects_raise(self):
        mechanics = SmallElasticPlastic(
            _parameters(), DefType.FULL_3D, reference_temperature=_T_REF)
        with self.assertRaisesRegex(ValueError, "share one Parameters"):
            ThermomechanicsModel(
                mechanics, Conduction(_parameters(), reference_temperature=_T_REF))

    def test_not_an_input_file_type(self):
        with self.assertRaisesRegex(ValueError, "not an input file type"):
            ThermomechanicsModel.from_deck({}, _parameters(), None)


if __name__ == "__main__":
    unittest.main()
