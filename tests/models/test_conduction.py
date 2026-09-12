"""Tests for the conduction model at a point."""
import unittest
from typing import cast

import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_map

from cmad.models.conduction import Conduction
from cmad.models.global_fields import GlobalFieldsAtPoint, StepTime
from cmad.parameters.parameters import Parameters
from cmad.typing import PyTreeDict


def _parameters(
        k: float, rho: float | None = None, c: float | None = None,
) -> Parameters:
    thermal: dict[str, float] = {"conductivity": k}
    if rho is not None:
        thermal["density"] = rho
    if c is not None:
        thermal["specific heat"] = c
    values = cast(PyTreeDict, {"thermal": thermal})
    return Parameters(
        values, tree_map(lambda _: True, values), tree_map(lambda _: None, values),
    )


def _fields(T: float, grad_T: list[float]) -> GlobalFieldsAtPoint:
    return GlobalFieldsAtPoint(
        fields={"T": jnp.array([T])},
        grad_fields={"T": jnp.asarray(grad_T)[None, :]},
    )


class TestConduction(unittest.TestCase):
    def test_flux_is_minus_k_grad_T(self) -> None:
        model = Conduction(_parameters(16.0))
        params = model.parameters.values
        U = _fields(300.0, [1.0, -2.0, 0.5])
        q = model.heat_flux_closed_form(params, U, U)
        np.testing.assert_allclose(q, -16.0 * np.array([1.0, -2.0, 0.5]))
        np.testing.assert_allclose(model.heat_flux([], [], params, U, U), q)

    def test_capacity_rate_is_rho_c_dT_dt(self) -> None:
        model = Conduction(_parameters(16.0, rho=2.0, c=500.0))
        U = _fields(310.0, [0.0, 0.0, 0.0])
        U_prev = _fields(300.0, [0.0, 0.0, 0.0])
        rate = model.heat_capacity_rate(
            model.parameters.values, U, U_prev, StepTime(0.5, 0.0),
        )
        self.assertAlmostEqual(float(rate), 2.0 * 500.0 * 10.0 / 0.5)

    def test_no_capacity_is_a_zero_rate(self) -> None:
        model = Conduction(_parameters(16.0))
        U = _fields(310.0, [0.0, 0.0, 0.0])
        U_prev = _fields(300.0, [0.0, 0.0, 0.0])
        rate = model.heat_capacity_rate(
            model.parameters.values, U, U_prev, StepTime(0.5, 0.0),
        )
        self.assertEqual(float(rate), 0.0)

    def test_no_local_state(self) -> None:
        model = Conduction(_parameters(16.0))
        self.assertEqual(model.num_residuals, 0)
        self.assertEqual(model._init_xi, [])
        self.assertTrue(model.supports_closed_form)

    def test_density_without_specific_heat_raises(self) -> None:
        with self.assertRaises(ValueError):
            Conduction(_parameters(16.0, rho=2.0))


if __name__ == "__main__":
    unittest.main()
