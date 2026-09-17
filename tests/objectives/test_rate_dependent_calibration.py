"""Material point calibration of a rate dependent model over real times.

A viscoplastic flow rule reads ``step_time.dt`` directly, so the step
times are part of the problem: the same deformation path walked at a
different speed is a different answer, and the viscosity ``eta`` is only
identifiable against times that are actually supplied. These check that
the time history reaches the residual on every material point sensitivity
path, and that the derivatives with respect to every model parameter
survive it.

``step_time`` is argnum 5 of the residual and the Model's cached
derivatives target argnums 0-4, so ``dt`` enters each step as a fixed
coefficient rather than a differentiation target. The risk is not in the
algebra but in the plumbing: an adjoint or Hessian pass that re-gathers a
step at a different ``dt`` than the forward pass solved it at would give
a gradient consistent with no problem at all. The finite difference check
below is what would catch that.
"""
import unittest
from functools import partial

import numpy as np

from cmad.io.params_builder import build_parameters
from cmad.models.be_bar_elastic_plastic import BeBarElasticPlastic
from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.global_fields import StepTime, mp_U_from_F
from cmad.models.nonlinear_solver import make_newton_solve, newton_solve
from cmad.objectives.mp_jvp_objective import MPJVPObjective
from cmad.objectives.mp_objective import (
    MPAdjointObjective,
    MPDirectAdjointObjective,
    MPDirectObjective,
)
from cmad.parameters.parameters import Parameters
from cmad.qois.calibration import Calibration

# E, nu, Y, S, D
_E, _NU, _Y, _S, _D = 200e3, 0.3, 200.0, 200.0, 20.0
_ETA = 5e3
_DEF_TYPE = DefType.FULL_3D
_NUM_STEPS = 16


def _parameters(Y: float = _Y, eta: float = _ETA) -> Parameters:
    """Perzyna be_bar materials tree with ``Y`` and ``eta`` active.

    No transforms, so the canonical flat coordinates the objectives take
    are the raw parameter values and a finite difference on one is a
    finite difference on the other.
    """
    return build_parameters({
        "elastic": {"E": _E, "nu": _NU},
        "plastic": {
            "effective stress": {"J2": {}},
            "flow stress": {
                "initial yield": {"Y": {"value": Y, "active": True}},
                "hardening": {"voce": {"S": _S, "D": _D}},
                "rate_dependence": {
                    "perzyna": {"eta": {"value": eta, "active": True}},
                },
            },
        },
    })


def _deformation_history() -> np.ndarray:
    """A monotone, nearly isochoric uniaxial stretch path well into flow."""
    ndims = def_type_ndims(_DEF_TYPE)
    s = np.linspace(0.0, 1.0, _NUM_STEPS + 1)
    lambda_axial = 1.0 + 0.06 * s
    lambda_lateral = 1.0 / np.sqrt(lambda_axial)

    F = np.zeros((ndims, ndims, _NUM_STEPS + 1))
    F[0, 0, :] = lambda_axial
    F[1, 1, :] = lambda_lateral
    F[2, 2, :] = lambda_lateral
    return F


def _nonuniform_times(total_time: float = 1.0) -> np.ndarray:
    """Strictly increasing times whose step sizes vary over the history.

    Uniform spacing would not distinguish a correctly threaded time
    history from one that quietly reused a single ``dt``.
    """
    rng = np.random.default_rng(0)
    dts = 1.0 + rng.uniform(0.0, 1.5, _NUM_STEPS)
    times = np.concatenate([[0.0], np.cumsum(dts)])
    return times * (total_time / times[-1])


def _compute_cauchy(
        model: BeBarElasticPlastic, F: np.ndarray, times: np.ndarray,
) -> np.ndarray:
    num_steps = F.shape[2] - 1
    model.set_xi_to_init_vals()
    cauchy = np.zeros((3, 3, num_steps + 1))
    for step in range(1, num_steps + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step]),
            mp_U_from_F(F[:, :, step - 1]),
        )
        model.gather_time(StepTime(times[step], times[step - 1]))
        newton_solve(model, max_iters=50)
        model.advance_xi()
        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
    return cauchy


def _weight() -> np.ndarray:
    weight = np.zeros((3, 3))
    weight[0, 0] = 1.
    return weight


def _make_qoi(data: np.ndarray, Y: float, eta: float) -> Calibration:
    """A QoI on a fresh model, so each objective owns its own state."""
    model = BeBarElasticPlastic(_parameters(Y, eta), def_type=_DEF_TYPE)
    return Calibration(model, data, _weight())


class TestRateDependentSensitivities(unittest.TestCase):
    """Gradients and Hessians of a Perzyna calibration over varying dt."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.F = _deformation_history()
        cls.times = _nonuniform_times()
        truth = BeBarElasticPlastic(_parameters(), def_type=_DEF_TYPE)
        cls.data = _compute_cauchy(truth, cls.F, cls.times)
        # Off the truth in both parameters, so neither derivative vanishes.
        cls.x = np.array([0.85 * _Y, 1.4 * _ETA])

    def _J_of(self, x: np.ndarray) -> float:
        """J alone, through an independent forward solve."""
        qoi = _make_qoi(self.data, *x)
        obj = MPAdjointObjective(qoi, self.F, self.times)
        return float(obj.evaluate(x).J)

    def test_gradient_matches_finite_differences(self) -> None:
        """The adjoint gradient against central differences on J.

        This is the check that the reverse pass re-gathers each step at
        the same ``dt`` the forward pass used: a mismatched schedule
        still produces a smooth, plausible gradient, just not one of
        this J.
        """
        qoi = _make_qoi(self.data, *self.x)
        grad = MPAdjointObjective(qoi, self.F, self.times).evaluate(self.x).grad

        fd = np.zeros_like(grad)
        for ii in range(self.x.size):
            h = 1e-6 * abs(self.x[ii])
            x_plus = self.x.copy()
            x_minus = self.x.copy()
            x_plus[ii] += h
            x_minus[ii] -= h
            fd[ii] = (self._J_of(x_plus) - self._J_of(x_minus)) / (2. * h)

        np.testing.assert_allclose(grad, fd, rtol=1e-5)

    def test_direct_and_adjoint_gradients_agree(self) -> None:
        """The forward tangent pass carries the same times as the reverse."""
        adjoint = MPAdjointObjective(
            _make_qoi(self.data, *self.x), self.F, self.times,
        ).evaluate(self.x)
        direct = MPDirectObjective(
            _make_qoi(self.data, *self.x), self.F, self.times,
        ).evaluate(self.x)

        np.testing.assert_allclose(adjoint.J, direct.J, rtol=1e-12)
        np.testing.assert_allclose(
            adjoint.grad, direct.grad, rtol=1e-9, atol=1e-9,
        )

    def test_jvp_and_direct_adjoint_agree(self) -> None:
        """The two structurally independent families over the same times.

        MPDirectAdjointObjective assembles J, gradient, and Hessian from
        numpy einsums over the model's derivative blocks; MPJVPObjective
        takes them end to end through a traced ``fori_loop``. They thread
        the time history through entirely separately, so agreement here
        pins both.
        """
        orig_qoi = _make_qoi(self.data, *self.x)
        orig = MPDirectAdjointObjective(orig_qoi, self.F, self.times)

        jvp_qoi = _make_qoi(self.data, *self.x)
        jvp_model = jvp_qoi.model()
        update_fun = make_newton_solve(
            jvp_model._residual, max_iters=50,
            initial_guess_fn=jvp_model.initial_guess_fn,
        )
        jvp = MPJVPObjective(jvp_qoi, self.F, update_fun, self.times)

        J_orig, grad_orig, hess_orig = orig.evaluate(self.x)
        J_jvp, grad_jvp = jvp.evaluate_objective_and_grad(self.x)
        hess_jvp = jvp.evaluate_hessian(self.x)

        np.testing.assert_allclose(J_orig, J_jvp, rtol=1e-10)
        np.testing.assert_allclose(grad_orig, grad_jvp, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(
            hess_orig, hess_jvp,
            rtol=1e-5, atol=1e-5 * np.abs(hess_orig).max(),
        )

    def test_times_reach_the_residual(self) -> None:
        """Walking the same path at a different speed is a different J.

        Perzyna's overstress scales with the plastic strain rate, so a
        history stretched over ten times the wall clock relaxes toward
        the rate independent answer. If the schedule never reached the
        residual, both would land on the default ``dt = 1`` and agree.
        """
        J_fast = float(MPAdjointObjective(
            _make_qoi(self.data, *self.x), self.F, self.times,
        ).evaluate(self.x).J)
        J_slow = float(MPAdjointObjective(
            _make_qoi(self.data, *self.x), self.F, _nonuniform_times(10.0),
        ).evaluate(self.x).J)

        self.assertGreater(abs(J_fast - J_slow) / abs(J_fast), 1e-3)

    def test_default_times_are_unit_steps(self) -> None:
        """Omitting times leaves the pre-existing dt = 1 behavior in place."""
        unit_times = np.arange(_NUM_STEPS + 1, dtype=np.float64)
        defaulted = MPAdjointObjective(
            _make_qoi(self.data, *self.x), self.F,
        ).evaluate(self.x)
        explicit = MPAdjointObjective(
            _make_qoi(self.data, *self.x), self.F, unit_times,
        ).evaluate(self.x)

        np.testing.assert_allclose(defaulted.J, explicit.J, rtol=1e-14)
        np.testing.assert_allclose(defaulted.grad, explicit.grad, rtol=1e-14)

    def test_wrong_number_of_times_raises(self) -> None:
        qoi = _make_qoi(self.data, *self.x)
        with self.assertRaises(ValueError) as cm:
            MPAdjointObjective(qoi, self.F, self.times[:-1])
        self.assertIn(f"{_NUM_STEPS + 1} times", str(cm.exception))

        with self.assertRaises(ValueError):
            MPJVPObjective(
                _make_qoi(self.data, *self.x), self.F,
                partial(make_newton_solve(qoi.model()._residual)),
                self.times[:-1],
            )


if __name__ == "__main__":
    unittest.main()
