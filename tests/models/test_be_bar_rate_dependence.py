"""Perzyna rate dependence in BeBarElasticPlastic.

The rate independent return map closes with consistency, ``f = 0``, so its
answer does not depend on how fast the step was taken. Perzyna replaces
that with ``delta_gamma = dt / eta * f``, which lets the stress sit
outside the yield surface by an overstress proportional to the plastic
strain rate.

There is a continuous reference for this,
:func:`tests.support.test_problems.finite_uniaxial_j2_voce_perzyna`: at a
constant equivalent plastic strain rate the overstress ``eta * alpha_dot``
is a constant shift of the initial yield, so the viscoplastic path is the
rate independent one at a raised ``Y``. The discrete return map converges
to it under step refinement, which is the main test here, driven along
that path with the times spaced so ``delta_alpha / dt`` is fixed.

Alongside it: the Perzyna relation recomputed pointwise from a converged
state, the invariance that only the product ``eta * alpha_dot`` matters,
and the ``eta -> 0`` limit taken against the model's own rate independent
branch.
"""
import unittest
from functools import partial

import numpy as np

from cmad.io.params_builder import build_parameters
from cmad.models.be_bar_elastic_plastic import (
    BeBarElasticPlastic,
    compute_yield_fun,
)
from cmad.models.deformation_types import DefType
from cmad.models.global_fields import StepTime, mp_U_from_F
from cmad.models.hardening import combined_hardening_fun, get_hardening_funs
from cmad.models.nonlinear_solver import newton_solve
from cmad.parameters.parameters import Parameters
from tests.support.test_problems import (
    finite_uniaxial_j2_voce,
    finite_uniaxial_j2_voce_perzyna,
)

# E, nu, Y, S, D
_E, _NU, _Y, _S, _D = 200e3, 0.3, 200.0, 200.0, 20.0
# hardening reached at the end of the constant rate path
_MAX_ALPHA = 0.15


def _parameters(eta: float | None) -> Parameters:
    """Deck-shaped materials tree through the deck's own builder.

    The Perzyna subtree is added only when ``eta`` is given; everything
    else is identical either way, so the rate independent limit compares
    like with like.
    """
    flow_stress: dict = {
        "initial yield": {"Y": _Y},
        "hardening": {"voce": {"S": _S, "D": _D}},
    }
    if eta is not None:
        flow_stress["rate_dependence"] = {"perzyna": {"eta": eta}}

    return build_parameters({
        "elastic": {"E": _E, "nu": _NU},
        "plastic": {
            "effective stress": {"J2": {}},
            "flow stress": flow_stress,
        },
    })


def _drive_along_path(
        eta: float | None, alpha_dot: float, num_steps: int,
        max_alpha: float = _MAX_ALPHA,
) -> BeBarElasticPlastic:
    """Step along the constant rate uniaxial stress path in ``num_steps``.

    The stretches are sampled from the continuous reference in ``alpha``
    rather than interpolated between its endpoints, and the times are
    spaced so ``delta_alpha / dt`` is exactly ``alpha_dot`` at every step,
    which is what makes the reference's constant overstress the right one
    to compare against.

    Returns the model at the end of the last step, before ``advance_xi``,
    so a caller can read both ``xi`` and ``xi_prev`` of that step.
    """
    Y_shift = eta * alpha_dot if eta is not None else 0.0
    model = BeBarElasticPlastic(_parameters(eta), def_type=DefType.FULL_3D)
    model.set_xi_to_init_vals()

    dt = (max_alpha / num_steps) / alpha_dot
    F_prev = np.eye(3)
    for step in range(1, num_steps + 1):
        alpha = max_alpha * step / num_steps
        lambda_axial, lambda_lateral, _ = finite_uniaxial_j2_voce(
            _E, _NU, _Y + Y_shift, _S, _D, alpha,
        )
        F = np.diag([lambda_axial, lambda_lateral, lambda_lateral])
        t = alpha / alpha_dot
        model.gather_global(mp_U_from_F(F), mp_U_from_F(F_prev))
        model.gather_time(StepTime(t, t - dt))
        newton_solve(model, max_iters=50)
        if step < num_steps:
            model.advance_xi()
        F_prev = F

    return model


def _cauchy(model: BeBarElasticPlastic) -> np.ndarray:
    model.seed_none()
    model.evaluate_cauchy()
    return np.asarray(model.Sigma())


class TestPerzynaContinuousReference(unittest.TestCase):
    """Against the continuous constant rate solution."""

    _ETA, _ALPHA_DOT = 1e4, 1e-3

    def test_converges_to_the_continuous_reference(self) -> None:
        _, _, cauchy_ref = finite_uniaxial_j2_voce_perzyna(
            _E, _NU, _Y, _S, _D, _MAX_ALPHA, self._ETA, self._ALPHA_DOT,
        )

        coarse = _cauchy(_drive_along_path(self._ETA, self._ALPHA_DOT, 20))
        fine = _cauchy(_drive_along_path(self._ETA, self._ALPHA_DOT, 80))

        err_coarse = abs(coarse[0, 0] - cauchy_ref) / cauchy_ref
        err_fine = abs(fine[0, 0] - cauchy_ref) / cauchy_ref

        self.assertLess(err_fine, err_coarse)
        self.assertLess(err_fine, 5e-4)
        # uniaxial stress: the lateral components vanish in the same limit
        self.assertLess(abs(fine[1, 1]) / cauchy_ref, 1e-3)
        self.assertLess(abs(fine[2, 2]) / cauchy_ref, 1e-3)

    def test_no_worse_than_the_rate_independent_control(self) -> None:
        """The viscoplastic branch converges as well as the rate
        independent one does through the same driver, so the residual
        error is the path discretization rather than the flow rule."""
        _, _, ref_rate = finite_uniaxial_j2_voce_perzyna(
            _E, _NU, _Y, _S, _D, _MAX_ALPHA, self._ETA, self._ALPHA_DOT,
        )
        _, _, ref_indep = finite_uniaxial_j2_voce(
            _E, _NU, _Y, _S, _D, _MAX_ALPHA,
        )

        rate = _cauchy(_drive_along_path(self._ETA, self._ALPHA_DOT, 80))
        indep = _cauchy(_drive_along_path(None, self._ALPHA_DOT, 80))

        err_rate = abs(rate[0, 0] - ref_rate) / ref_rate
        err_indep = abs(indep[0, 0] - ref_indep) / ref_indep

        self.assertLess(err_rate, 3.0 * err_indep)

    def test_only_the_product_of_eta_and_rate_matters(self) -> None:
        """The overstress is eta * alpha_dot, so trading one against the
        other leaves the stress unchanged."""
        slow_stiff = _cauchy(_drive_along_path(1e4, 1e-3, 20))
        fast_soft = _cauchy(_drive_along_path(1e3, 1e-2, 20))

        self.assertAlmostEqual(
            slow_stiff[0, 0] / fast_soft[0, 0], 1.0, places=10,
        )

    def test_overstress_scales_with_the_rate(self) -> None:
        """Ten times the rate is ten times the overstress above the rate
        independent flow stress."""
        _, _, ref_indep = finite_uniaxial_j2_voce(
            _E, _NU, _Y, _S, _D, _MAX_ALPHA,
        )
        slow = _cauchy(_drive_along_path(self._ETA, 1e-3, 40))[0, 0]
        fast = _cauchy(_drive_along_path(self._ETA, 1e-2, 40))[0, 0]

        self.assertAlmostEqual(
            (fast - ref_indep) / (slow - ref_indep), 10.0, delta=0.15,
        )


class TestBeBarPerzyna(unittest.TestCase):

    def test_overstress_relation_holds_at_the_solution(self) -> None:
        """The converged state satisfies eta / dt * delta_alpha = f.

        The continuous reference only sees the product ``eta * alpha_dot``,
        so this is what pins ``eta`` on its own.
        """
        eta, alpha_dot, num_steps = 5e3, 1e-3, 20
        model = _drive_along_path(eta, alpha_dot, num_steps)
        dt = (_MAX_ALPHA / num_steps) / alpha_dot

        alpha = float(np.asarray(model.xi()[2])[0])
        alpha_prev = float(np.asarray(model.xi_prev()[2])[0])
        delta_alpha = alpha - alpha_prev
        self.assertGreater(delta_alpha, 0.0)

        # the model's yield function is scaled by 2 mu; undo it to compare
        # against the overstress in stress units
        two_mu = _E / (1.0 + _NU)
        yield_fun = two_mu * float(np.asarray(compute_yield_fun(
            model.xi()[0], np.asarray(model.xi()[2]),
            model.parameters.values,
            partial(combined_hardening_fun,
                    hardening_funs=get_hardening_funs()),
        )).reshape(()))

        self.assertAlmostEqual(
            eta / dt * delta_alpha / yield_fun, 1.0, places=6,
        )
        # the stress genuinely sits outside the yield surface
        self.assertGreater(yield_fun, 1.0)

    def test_small_viscosity_recovers_rate_independence(self) -> None:
        """eta -> 0 is the rate independent branch, as a limit of the same
        residual rather than a separate code path.

        This is what the ``(eta / dt) * delta_alpha - f`` form buys over
        ``delta_alpha - (dt / eta) * f``, which divides by zero here.
        """
        _, _, cauchy_ref = finite_uniaxial_j2_voce(
            _E, _NU, _Y, _S, _D, _MAX_ALPHA,
        )
        viscous = _cauchy(_drive_along_path(1e-8, 1e-3, 40))[0, 0]
        absent = _cauchy(_drive_along_path(None, 1e-3, 40))[0, 0]

        self.assertAlmostEqual(viscous / absent, 1.0, places=8)
        self.assertLess(abs(viscous - cauchy_ref) / cauchy_ref, 5e-4)

    def test_rate_independent_by_default(self) -> None:
        """A deck without rate_dependence is untouched by the step size,
        so the rate independent branch never picks up dt."""
        model = _drive_along_path(None, 1e-3, 20)
        self.assertIsNone(model._rate_dependence)

        slow = _cauchy(_drive_along_path(None, 1e-3, 20))[0, 0]
        fast = _cauchy(_drive_along_path(None, 1e-2, 20))[0, 0]
        self.assertAlmostEqual(fast / slow, 1.0, places=12)

    def test_perzyna_selected_from_the_deck(self) -> None:
        model = BeBarElasticPlastic(
            _parameters(100.0), def_type=DefType.FULL_3D)
        self.assertEqual(model._rate_dependence, "perzyna")

    def test_plane_stress_carries_rate_dependence(self) -> None:
        """The out of plane unknown is orthogonal to the flow rule: the
        fourth equation is unchanged and the model still builds and
        solves."""
        model = BeBarElasticPlastic(
            _parameters(5e3), def_type=DefType.PLANE_STRESS)
        model.set_xi_to_init_vals()
        model.gather_global(
            mp_U_from_F(np.diag([1.05, 1.0 / np.sqrt(1.05)])),
            mp_U_from_F(np.eye(2)),
        )
        model.gather_time(StepTime(0.1, 0.0))
        newton_solve(model, max_iters=50)

        model.seed_none()
        model.evaluate_cauchy()
        cauchy = np.asarray(model.Sigma())
        self.assertLess(abs(cauchy[2, 2]) / abs(cauchy[0, 0]), 1e-10)


class TestRateDependenceResolution(unittest.TestCase):

    def test_unknown_law_raises(self) -> None:
        params = _parameters(100.0)
        rate = params.values["plastic"]["flow stress"]["rate_dependence"]
        rate["norton"] = rate.pop("perzyna")

        with self.assertRaises(ValueError) as ctx:
            BeBarElasticPlastic(params, def_type=DefType.FULL_3D)
        self.assertIn("perzyna", str(ctx.exception))

    def test_two_laws_raise(self) -> None:
        params = _parameters(100.0)
        rate = params.values["plastic"]["flow stress"]["rate_dependence"]
        rate["also_perzyna"] = dict(rate["perzyna"])

        with self.assertRaises(ValueError) as ctx:
            BeBarElasticPlastic(params, def_type=DefType.FULL_3D)
        self.assertIn("exactly one", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
