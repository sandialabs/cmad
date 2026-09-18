"""Rate-dependent relations at a material point against a scalar return map.

Uniaxial stress, strain-controlled, backward Euler. The return map is exact
for the small strain models under ``DefType.UNIAXIAL_STRESS`` with J2, so
they match it to within the local Newton tolerance at two strain rates.
The finite deformation models are driven with the return map's lateral
strains at small strain, where their kinematics error is of the order of
the strain and halves with it, and their rate effect, the difference
between the two rates, matches the return map's.
"""
import unittest

import numpy as np
from jax.tree_util import tree_map
from scipy.optimize import brentq

from cmad.models.be_bar_elastic_plastic import BeBarElasticPlastic
from cmad.models.deformation_types import DefType
from cmad.models.elastic_stress import two_mu_scale_factor
from cmad.models.flow_stress import POWER_LAW_OFFSET
from cmad.models.global_fields import StepTime, mp_U_from_F
from cmad.models.hypo_elastic_plastic import HypoElasticPlastic
from cmad.models.nonlinear_solver import newton_solve
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.parameters.parameters import Parameters

_E, _NU = 200e3, 0.3
JOHNSON_COOK = {
    "johnson_cook": {
        "A": 200.0, "B": 300.0, "n": 0.3, "C": 0.014,
        "reference rate": 1e-3,
        "reference temperature": 294.0, "melt temperature": 1793.0,
        "m": 1.03,
    },
}
PERIC = {
    "peric": {
        "initial yield": {"Y": 200.0},
        "hardening": {"voce": {"S": 200.0, "D": 20.0}},
        "eta": 10.0, "epsilon": 0.2,
    },
}
# eta in stress * time. The plastic rate tracks the strain rate closely
# here, so the two rates below differ by ~45 MPa of overstress on ~230 MPa
# of stress -- a rate effect well clear of the tolerances.
PERZYNA = {
    "perzyna": {
        "initial yield": {"Y": 200.0},
        "hardening": {"voce": {"S": 200.0, "D": 20.0}},
        "eta": 500.0,
    },
}
NUM_STEPS = 40
PEAK_STRAIN = 3e-3
STRAIN_RATES = (0.01, 0.1)
MAX_ITERS = 20
# The local Newton's default tolerance on the scaled residual; the models
# scale the yield equation by 2 mu, so the stress agrees to that times
# 2 mu and alpha to that over E.
LOCAL_TOL = 1e-12


def johnson_cook_flow_stress(alpha, rate):
    p = JOHNSON_COOK["johnson_cook"]
    a0 = POWER_LAW_OFFSET
    return (p["A"] + p["B"] * (alpha + a0) ** p["n"]) \
        * (1.0 + p["C"] * np.log(max(rate / p["reference rate"], 1.0))) \
        * (1.0 - a0 ** p["m"])


def peric_flow_stress(alpha, rate):
    p = PERIC["peric"]
    voce = p["hardening"]["voce"]
    sigma_y = p["initial yield"]["Y"] \
        + voce["S"] * (1.0 - np.exp(-voce["D"] * alpha))
    return sigma_y * (1.0 + p["eta"] * rate) ** p["epsilon"]


def perzyna_flow_stress(alpha, rate):
    p = PERZYNA["perzyna"]
    voce = p["hardening"]["voce"]
    sigma_y = p["initial yield"]["Y"] \
        + voce["S"] * (1.0 - np.exp(-voce["D"] * alpha))
    return sigma_y + p["eta"] * rate


def return_map(flow_stress, strains, times):
    """The uniaxial stress, small strain, backward Euler return map.

    Returns ``sigma_11``, ``alpha``, and the axial plastic strain per step.
    """
    sigma = np.zeros(len(strains))
    alpha = np.zeros(len(strains))
    eps_p = np.zeros(len(strains))
    for k in range(1, len(strains)):
        dt = times[k] - times[k - 1]
        alpha[k], eps_p[k] = alpha[k - 1], eps_p[k - 1]
        sigma_tr = _E * (strains[k] - eps_p[k - 1])
        sign = np.sign(sigma_tr)
        if abs(sigma_tr) <= flow_stress(alpha[k - 1], 0.0):
            sigma[k] = sigma_tr
            continue

        def g(dg, a=alpha[k - 1], s=abs(sigma_tr), dt=dt):
            return s - _E * dg - flow_stress(a + dg, dg / dt)

        dg = brentq(g, 0.0, abs(sigma_tr) / _E, xtol=1e-18, rtol=1e-15)
        sigma[k] = sigma_tr - _E * dg * sign
        alpha[k] = alpha[k - 1] + dg
        eps_p[k] = eps_p[k - 1] + dg * sign
    return sigma, alpha, eps_p


def schedule(peak_strain, strain_rate):
    strains = np.linspace(0.0, peak_strain, NUM_STEPS + 1)
    return strains, strains / strain_rate


def parameters(flow_stress_params):
    values = {
        "rotation matrix": np.eye(3),
        "elastic": {"E": _E, "nu": _NU},
        "plastic": {
            "effective stress": {"J2": 0.0},
            "flow stress": flow_stress_params,
        },
    }
    return Parameters(
        values,
        tree_map(lambda a: False, values),
        tree_map(lambda a: None, values),
    )


def drive(model, F, times, alpha_index):
    """Run the point through ``F`` at ``times``.

    Returns the Cauchy stress history, the alpha history, the Newton
    iteration count per step, and the residual norm after each solve.
    """
    num_steps = F.shape[2] - 1
    model.set_xi_to_init_vals()
    cauchy = np.zeros((3, 3, num_steps + 1))
    alpha = np.zeros(num_steps + 1)
    iterations = np.zeros(num_steps + 1, dtype=int)
    residuals = np.zeros(num_steps + 1)
    for step in range(1, num_steps + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step]),
            mp_U_from_F(F[:, :, step - 1]),
        )
        model.gather_time(StepTime(times[step], times[step - 1]))
        iterations[step], _ = newton_solve(model, max_iters=MAX_ITERS)
        model.seed_none()
        model.evaluate()
        residuals[step] = np.linalg.norm(model.C())
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
        alpha[step] = float(model.xi()[alpha_index][0])
        model.advance_xi()
    return cauchy, alpha, iterations, residuals


def run_small_strain(Model, flow_stress_params, flow_stress, strain_rate):
    strains, times = schedule(PEAK_STRAIN, strain_rate)
    sigma_ref, alpha_ref, _ = return_map(flow_stress, strains, times)
    model = Model(parameters(flow_stress_params), DefType.UNIAXIAL_STRESS)
    F = (1.0 + strains)[None, None, :]
    return drive(model, F, times, 1), sigma_ref, alpha_ref


def run_finite(
        Model, alpha_index, flow_stress_params, flow_stress, peak_strain,
        strain_rate,
):
    strains, times = schedule(peak_strain, strain_rate)
    sigma_ref, alpha_ref, eps_p = return_map(flow_stress, strains, times)
    eps_lat = -_NU * (strains - eps_p) - eps_p / 2.0
    F = np.repeat(np.eye(3)[:, :, None], NUM_STEPS + 1, axis=2)
    F[0, 0, :] += strains
    F[1, 1, :] += eps_lat
    F[2, 2, :] += eps_lat
    model = Model(parameters(flow_stress_params), DefType.FULL_3D)
    return drive(model, F, times, alpha_index), sigma_ref, alpha_ref


def stress_bound():
    """The stress agreement the local Newton tolerance permits."""
    return 2.0 * LOCAL_TOL * float(two_mu_scale_factor(
        parameters(JOHNSON_COOK).values))


class TestSmallStrainModelsMatchTheReturnMap(unittest.TestCase):

    def _check(self, flow_stress_params, flow_stress):
        for Model in (SmallElasticPlastic, SmallRateElasticPlastic):
            for strain_rate in STRAIN_RATES:
                (cauchy, alpha, iterations, residuals), sigma_ref, alpha_ref = \
                    run_small_strain(
                        Model, flow_stress_params, flow_stress, strain_rate)
                self.assertLess(iterations.max(), MAX_ITERS)
                self.assertLess(residuals.max(), LOCAL_TOL)
                self.assertLess(
                    np.abs(cauchy[0, 0, :] - sigma_ref).max(), stress_bound())
                self.assertLess(
                    np.abs(alpha - alpha_ref).max(), stress_bound() / _E)
                off_axis = cauchy.copy()
                off_axis[0, 0, :] = 0.0
                self.assertLess(np.abs(off_axis).max(), stress_bound())

    def test_johnson_cook(self) -> None:
        # Both strain rates exceed the reference rate once the point flows,
        # so the rate term is exercised rather than clipped.
        strains, times = schedule(PEAK_STRAIN, STRAIN_RATES[0])
        _, alpha_ref, _ = return_map(johnson_cook_flow_stress, strains, times)
        rates = np.diff(alpha_ref) / np.diff(times)
        plastic = np.flatnonzero(rates > 0.0)[1:]
        self.assertGreater(
            rates[plastic].min(),
            JOHNSON_COOK["johnson_cook"]["reference rate"])
        self._check(JOHNSON_COOK, johnson_cook_flow_stress)

    def test_peric(self) -> None:
        self._check(PERIC, peric_flow_stress)

    def test_perzyna(self) -> None:
        self._check(PERZYNA, perzyna_flow_stress)


class TestFiniteModelsAtSmallStrain(unittest.TestCase):

    def _check(self, Model, alpha_index, flow_stress_params, flow_stress):
        errors = {}
        rate_effect_errors = {}
        for level in (1.0, 0.5):
            sigma = {}
            sigma_ref = {}
            for strain_rate in STRAIN_RATES:
                (cauchy, _, iterations, residuals), sigma_ref[strain_rate], _ = \
                    run_finite(
                        Model, alpha_index, flow_stress_params, flow_stress,
                        level * PEAK_STRAIN, strain_rate)
                self.assertLess(iterations.max(), MAX_ITERS)
                self.assertLess(residuals.max(), LOCAL_TOL)
                sigma[strain_rate] = cauchy[0, 0, :]
                errors[(level, strain_rate)] = \
                    np.abs(sigma[strain_rate] - sigma_ref[strain_rate]).max() \
                    / np.abs(sigma_ref[strain_rate]).max()
            slow, fast = STRAIN_RATES
            return_map_effect = sigma_ref[fast] - sigma_ref[slow]
            model_effect = sigma[fast] - sigma[slow]
            rate_effect_errors[level] = \
                np.abs(model_effect - return_map_effect).max() \
                / np.abs(return_map_effect).max()
        for error in errors.values():
            self.assertLess(error, 1e-2)
        for strain_rate in STRAIN_RATES:
            self.assertLess(
                errors[(0.5, strain_rate)], 0.75 * errors[(1.0, strain_rate)])
        for error in rate_effect_errors.values():
            self.assertLess(error, 0.05)

    def test_hypoelastic_johnson_cook(self) -> None:
        self._check(
            HypoElasticPlastic, 1, JOHNSON_COOK, johnson_cook_flow_stress)

    def test_be_bar_peric(self) -> None:
        self._check(BeBarElasticPlastic, 2, PERIC, peric_flow_stress)

    def test_be_bar_perzyna(self) -> None:
        self._check(BeBarElasticPlastic, 2, PERZYNA, perzyna_flow_stress)


if __name__ == "__main__":
    unittest.main()
