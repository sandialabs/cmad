"""Material point sensitivities with exactly one active parameter.

The objectives accumulate the gradient as a ``(1, num_active_params)``
row and flatten it before handing it to ``transform_grad`` and the
optimizer. A lone active parameter makes that row ``(1, 1)``, the one
shape ``squeeze`` flattens too far — to a 0-d array that every consumer
then indexes. Calibrating a single parameter is an ordinary thing to
want (a viscosity on its own, say), so these pin the shape for all three
strategies and check the one-parameter gradient is right, not just
well shaped.
"""
import unittest

import numpy as np

from cmad.io.params_builder import build_parameters
from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.global_fields import mp_U_from_F
from cmad.models.nonlinear_solver import newton_solve
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.objectives.mp_objective import (
    MPAdjointObjective,
    MPDirectAdjointObjective,
    MPDirectObjective,
)
from cmad.parameters.parameters import Parameters
from cmad.qois.calibration import Calibration

_E, _NU, _Y, _S, _D = 200e3, 0.3, 200.0, 200.0, 20.0
_DEF_TYPE = DefType.PLANE_STRESS
_NUM_STEPS = 10


def _parameters(Y: float = _Y) -> Parameters:
    """Only ``Y`` active; everything else is a fixed material constant."""
    return build_parameters({
        "rotation matrix": np.eye(3).tolist(),
        "elastic": {"E": _E, "nu": _NU},
        "plastic": {
            "effective stress": {"J2": 0.0},
            "flow stress": {
                "initial yield": {"Y": {"value": Y, "active": True}},
                "hardening": {"voce": {"S": _S, "D": _D}},
            },
        },
    })


def _F_history() -> np.ndarray:
    ndims = def_type_ndims(_DEF_TYPE)
    F = np.repeat(np.eye(ndims)[:, :, np.newaxis], _NUM_STEPS + 1, axis=2)
    F[0, 0, :] += np.linspace(0.0, 0.03, _NUM_STEPS + 1)
    return F


def _weight() -> np.ndarray:
    weight = np.zeros((3, 3))
    weight[0, 0] = 1.
    return weight


def _compute_cauchy(model: SmallElasticPlastic, F: np.ndarray) -> np.ndarray:
    model.set_xi_to_init_vals()
    cauchy = np.zeros((3, 3, _NUM_STEPS + 1))
    for step in range(1, _NUM_STEPS + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step]),
            mp_U_from_F(F[:, :, step - 1]),
        )
        newton_solve(model)
        model.advance_xi()
        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
    return cauchy


class TestSingleActiveParameter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.F = _F_history()
        truth = SmallElasticPlastic(_parameters(), _DEF_TYPE)
        cls.data = _compute_cauchy(truth, cls.F)
        cls.x = np.array([0.8 * _Y])

    def _objective(self, cls_):  # type: ignore[no-untyped-def]
        model = SmallElasticPlastic(_parameters(*self.x), _DEF_TYPE)
        self.assertEqual(model.parameters.num_active_params, 1)
        return cls_(Calibration(model, self.data, _weight()), self.F)

    def _J_of(self, x: np.ndarray) -> float:
        return float(self._objective(MPAdjointObjective).evaluate(x).J)

    def test_all_strategies_return_a_length_one_gradient(self) -> None:
        for cls_ in (
                MPAdjointObjective, MPDirectObjective,
                MPDirectAdjointObjective,
        ):
            with self.subTest(strategy=cls_.__name__):
                result = self._objective(cls_).evaluate(self.x)
                self.assertEqual(np.asarray(result.grad).shape, (1,))
                self.assertTrue(np.all(np.isfinite(result.grad)))

    def test_gradient_matches_finite_differences(self) -> None:
        grad = self._objective(MPAdjointObjective).evaluate(self.x).grad

        h = 1e-6 * abs(self.x[0])
        fd = (self._J_of(self.x + h) - self._J_of(self.x - h)) / (2. * h)

        np.testing.assert_allclose(grad[0], fd, rtol=1e-6)

    def test_strategies_agree_on_the_single_gradient(self) -> None:
        reference = self._objective(MPAdjointObjective).evaluate(self.x).grad
        for cls_ in (MPDirectObjective, MPDirectAdjointObjective):
            with self.subTest(strategy=cls_.__name__):
                grad = self._objective(cls_).evaluate(self.x).grad
                np.testing.assert_allclose(
                    grad, reference, rtol=1e-8, atol=1e-8,
                )

    def test_direct_adjoint_returns_a_one_by_one_hessian(self) -> None:
        result = self._objective(MPDirectAdjointObjective).evaluate(self.x)
        self.assertEqual(np.asarray(result.hessian).shape, (1, 1))
        self.assertTrue(np.all(np.isfinite(result.hessian)))


if __name__ == "__main__":
    unittest.main()
