"""Shape contract + Elastic equivalence + J2 IFT-correction tests
for GlobalResidual.for_model COUPLED mode.

Bundles a small toy 3D u-only quasi-static equilibrium GR (mirror
of test_abc_contract.py's _ToyEquilibrium, mode-aware via the
``mode`` arg threaded through residual_fn) and exercises the
COUPLED branch's 2-key dict shape (both 8-arg Newton-running),
the R + dR/dU equivalence between CLOSED_FORM and COUPLED bindings
on a closed-form-capable Elastic model, and a JVP-vs-FD check on
dR/dU at a J2 plastic-loading point (validates the IFT correction
in make_newton_solve's custom_vjp rule).
"""
import unittest
from typing import cast

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.global_residuals import (
    GlobalResidual,
    GlobalResidualMode,
)
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.var_types import VarType
from cmad.parameters.parameters import Parameters
from cmad.typing import PyTreeDict
from tests.support.test_problems import J2AnalyticalProblem


def _tet_barycenter_shapes() -> ShapeFunctionsAtIP:
    N = jnp.array([0.25, 0.25, 0.25, 0.25])
    grad_N = jnp.array([
        [-1., -1., -1.],
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.],
    ])
    return ShapeFunctionsAtIP(N=N, grad_N=grad_N)


def _make_parameters() -> Parameters:
    values = cast(
        PyTreeDict, {"elastic": {"kappa": 100.0, "mu": 50.0}})
    active_flags = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Parameters(values, active_flags, transforms)


def _make_linear_elastic_model() -> Elastic:
    return Elastic(_make_parameters(), def_type=DefType.FULL_3D)


def _make_J2_model() -> SmallElasticPlastic:
    """SmallElasticPlastic with the J2-Voce parameter set from the
    calibration test problem (E=200e3, nu=0.3, Y=200, Voce S=200,
    D=20)."""
    return SmallElasticPlastic(
        J2AnalyticalProblem().J2_parameters, def_type=DefType.FULL_3D)


def _j2_plastic_inputs(model: SmallElasticPlastic):
    """Element-level evaluation point with strain components a few
    times the yield strain (Y/E = 1e-3); the local Newton lands in
    the plastic branch."""
    U = [
        jnp.zeros((4, 3))
        .at[1, 0].set(0.005)
        .at[2, 1].set(0.003)
        .at[3, 2].set(0.002)
    ]
    U_prev = [jnp.zeros((4, 3))]
    params = model.parameters.values
    shapes_ip = [_tet_barycenter_shapes()]
    w = 1.0
    dv = 1.0 / 6.0
    h = 1.0
    ip_set = 0
    return params, U, U_prev, shapes_ip, w, dv, h, ip_set


class _ToyEquilibrium(GlobalResidual):
    """3D u-only quasi-static equilibrium, small-def kinematics.

    Mode-aware via the ``mode`` arg threaded through ``residual_fn``:
    CLOSED_FORM dispatches to ``model.cauchy_closed_form``, COUPLED
    dispatches to ``model.cauchy(xi, xi_prev, ...)``. Mirror of the
    same-named class in ``test_abc_contract.py``.
    """
    def __init__(self) -> None:
        self._is_complex = False
        self.dtype = np.float64
        self._ndims = 3

        self._init_residuals(1)
        self._var_types[0] = VarType.VECTOR
        self._num_eqs[0] = 3
        self.resid_names[0] = "displacement"
        self.var_names[0] = "u"

        def residual_fn(xi, xi_prev, params, U, U_prev,
                        model, mode, shapes_ip, w, dv, h, ip_set):
            U_ip = self.interpolate_global_fields_at_ip(U, shapes_ip)
            U_ip_prev = self.interpolate_global_fields_at_ip(U_prev, shapes_ip)
            if mode == GlobalResidualMode.CLOSED_FORM:
                sigma = model.cauchy_closed_form(params, U_ip, U_ip_prev)
            else:
                sigma = model.cauchy(xi, xi_prev, params, U_ip, U_ip_prev)
            R = (shapes_ip[0].grad_N @ sigma) * w * dv   # (4, 3)
            return [R]

        super().__init__(residual_fn)


def _test_inputs(model: Elastic):
    """Element-level evaluation point: small displacements at two
    nodes/components."""
    U = [jnp.zeros((4, 3)).at[1, 0].set(0.001).at[2, 1].set(0.0005)]
    U_prev = [jnp.zeros((4, 3))]
    params = model.parameters.values
    shapes_ip = [_tet_barycenter_shapes()]
    w = 1.0
    dv = 1.0 / 6.0
    h = 1.0
    ip_set = 0
    return params, U, U_prev, shapes_ip, w, dv, h, ip_set


class TestForModelCoupledShape(unittest.TestCase):
    def test_coupled_returns_two_keys(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        self.assertEqual(
            set(evaluators.keys()),
            {"R", "R_and_dR_dU_and_xi"},
        )

    def test_evaluators_callable_with_9arg_sig(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        params, U, U_prev, shapes_ip, w, dv, h, ip_set = (
            _test_inputs(model))
        xi_prev = [jnp.zeros_like(b) for b in model._init_xi]

        # Both COUPLED evaluators take 9 args:
        # (params, U, U_prev, xi_prev, shapes_ip, w, dv, h, ip_set).
        for key in ("R", "R_and_dR_dU_and_xi"):
            evaluators[key](
                params, U, U_prev, xi_prev,
                shapes_ip, w, dv, h, ip_set,
            )

    def test_R_and_dR_dU_and_xi_returns_triple(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        params, U, U_prev, shapes_ip, w, dv, h, ip_set = (
            _test_inputs(model))
        xi_prev = [jnp.zeros_like(b) for b in model._init_xi]

        R_blocks, dR_dU_blocks, xi = evaluators["R_and_dR_dU_and_xi"](
            params, U, U_prev, xi_prev,
            shapes_ip, w, dv, h, ip_set,
        )
        self.assertEqual(len(R_blocks), 1)
        self.assertEqual(R_blocks[0].shape, (4, 3))
        self.assertEqual(len(dR_dU_blocks), 1)
        self.assertEqual(len(dR_dU_blocks[0]), 1)
        self.assertEqual(dR_dU_blocks[0][0].shape, (4, 3, 4, 3))
        self.assertEqual(len(xi), 1)
        self.assertEqual(xi[0].shape, (6,))


class TestForModelCoupledClosedFormEquivalence(unittest.TestCase):
    """R + dR/dU equivalence between CLOSED_FORM and COUPLED on
    Elastic.

    Elastic is closed-form-capable. The COUPLED local Newton on
    ``C = cauchy - elastic_stress(F, params) = 0`` is linear in
    xi[0] and converges in one iteration for any starting xi_prev,
    yielding ``xi[0]_tensor = elastic_stress(F, params)``. The IFT-
    corrected total dR/dU then equals the direct CLOSED_FORM dR/dU
    to roundoff.
    """
    def test_R_and_dR_dU_match_to_1e_minus_12(self):
        gr_closed = _ToyEquilibrium()
        gr_coupled = _ToyEquilibrium()
        model = _make_linear_elastic_model()

        ev_closed = gr_closed.for_model(
            model, mode=GlobalResidualMode.CLOSED_FORM)
        ev_coupled = gr_coupled.for_model(
            model, mode=GlobalResidualMode.COUPLED)

        params, U, U_prev, shapes_ip, w, dv, h, ip_set = (
            _test_inputs(model))
        xi_prev = [jnp.zeros_like(b) for b in model._init_xi]

        R_closed, dR_dU_closed = ev_closed["R_and_dR_dU"](
            params, U, U_prev, shapes_ip, w, dv, h, ip_set,
        )
        R_coupled, dR_dU_coupled, _xi = ev_coupled[
            "R_and_dR_dU_and_xi"](
            params, U, U_prev, xi_prev,
            shapes_ip, w, dv, h, ip_set,
        )

        self.assertTrue(jnp.allclose(
            R_closed[0], R_coupled[0], rtol=0., atol=1e-12))
        self.assertTrue(jnp.allclose(
            dR_dU_closed[0][0], dR_dU_coupled[0][0],
            rtol=0., atol=1e-12))


class TestForModelCoupledJ2LocalNewton(unittest.TestCase):
    """COUPLED ``R_and_dR_dU_and_xi`` on a J2 plastic-loading point.

    The per-IP local Newton inside the evaluator solves
    ``model._residual(xi, xi_prev, params, U_ip, U_ip_prev) = 0``
    in the plastic branch; the converged xi is returned as the third
    element of the 3-tuple.
    """
    def test_local_newton_converges_to_equilibrium(self):
        gr = _ToyEquilibrium()
        model = _make_J2_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        params, U, U_prev, shapes_ip, w, dv, h, ip_set = (
            _j2_plastic_inputs(model))
        xi_prev = [jnp.zeros_like(b) for b in model._init_xi]

        _, _, xi_solved = evaluators["R_and_dR_dU_and_xi"](
            params, U, U_prev, xi_prev, shapes_ip, w, dv, h, ip_set)

        # xi at equilibrium for the loading: model._residual ≈ 0.
        U_ip = gr.interpolate_global_fields_at_ip(U, shapes_ip)
        U_ip_prev = gr.interpolate_global_fields_at_ip(
            U_prev, shapes_ip)
        residual = model._residual(
            xi_solved, xi_prev, params, U_ip, U_ip_prev)
        self.assertLess(float(jnp.linalg.norm(residual)), 1e-10)

        # Confirm plastic regime: alpha (xi[1]) is positive after the
        # solve (took at least one plastic step).
        self.assertGreater(float(xi_solved[1][0]), 0.0)


class TestForModelCoupledJVPvsFD(unittest.TestCase):
    """Forward-mode JVP correctness on ``dR/dU`` total at a J2
    plastic-loading point. Validates the IFT correction in
    ``make_newton_solve``'s ``custom_vjp`` rule by comparing the
    second return of ``R_and_dR_dU_and_xi`` (AD tangent) against
    central FD on the Newton-running ``R`` evaluator (each call
    re-runs the per-IP Newton, so the FD probe inherits the IFT
    structure)."""

    def test_dR_dU_total_matches_central_fd(self):
        gr = _ToyEquilibrium()
        model = _make_J2_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        params, U, U_prev, shapes_ip, w, dv, h, ip_set = (
            _j2_plastic_inputs(model))
        xi_prev = [jnp.zeros_like(b) for b in model._init_xi]

        _, dR_dU_blocks, _ = evaluators["R_and_dR_dU_and_xi"](
            params, U, U_prev, xi_prev,
            shapes_ip, w, dv, h, ip_set)
        ad_arr = np.asarray(dR_dU_blocks[0][0])  # (4, 3, 4, 3)

        eps = 1e-6
        fd_arr = np.zeros_like(ad_arr)
        for b in range(4):
            for k in range(3):
                U_plus = [U[0].at[b, k].add(eps)]
                U_minus = [U[0].at[b, k].add(-eps)]
                R_plus = evaluators["R"](
                    params, U_plus, U_prev, xi_prev,
                    shapes_ip, w, dv, h, ip_set)
                R_minus = evaluators["R"](
                    params, U_minus, U_prev, xi_prev,
                    shapes_ip, w, dv, h, ip_set)
                fd_arr[:, :, b, k] = (
                    R_plus[0] - R_minus[0]) / (2 * eps)

        self.assertTrue(np.allclose(
            ad_arr, fd_arr, rtol=1e-5, atol=1e-7))


class TestForModelMixedModeBindings(unittest.TestCase):
    """Two ``for_model`` calls on the same GR instance produce
    independent closures — the second binding does not corrupt the
    first. Regression for the late-binding bug where ``self._mode``
    was mutated on the GR instance and the first closure's body
    read the second binding's mode at call time, taking the wrong
    branch.

    Distinguisher: the dict keys differ by mode
    (``{"R", "R_and_dR_dU"}`` for CLOSED_FORM vs
    ``{"R", "R_and_dR_dU_and_xi"}`` for COUPLED), proving each
    closure captured its own mode lexically at ``for_model`` time
    rather than reading a mutable instance attribute. On Elastic the
    R values from the two modes agree to roundoff (per
    ``TestForModelCoupledClosedFormEquivalence``), so the value
    check uses same-mode fresh-GR references to catch any
    cross-contamination of captured params / state.
    """

    def test_two_bindings_same_GR_dont_share_state(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()

        ev_closed = gr.for_model(
            model, mode=GlobalResidualMode.CLOSED_FORM)
        ev_coupled = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)

        # Key-set distinguisher: each closure carries its captured
        # mode's evaluators, not the other's.
        self.assertEqual(
            set(ev_closed.keys()), {"R", "R_and_dR_dU"})
        self.assertEqual(
            set(ev_coupled.keys()), {"R", "R_and_dR_dU_and_xi"})

        params, U, U_prev, shapes_ip, w, dv, h, ip_set = (
            _test_inputs(model))
        xi_zeros = [jnp.zeros_like(b) for b in model._init_xi]

        R_closed = ev_closed["R"](
            params, U, U_prev, shapes_ip, w, dv, h, ip_set)
        R_coupled, _, _ = ev_coupled["R_and_dR_dU_and_xi"](
            params, U, U_prev, xi_zeros,
            shapes_ip, w, dv, h, ip_set)

        # Same-mode fresh-GR references: catches cross-
        # contamination of captured params / state across bindings.
        gr_closed_ref = _ToyEquilibrium()
        gr_coupled_ref = _ToyEquilibrium()
        ev_closed_ref = gr_closed_ref.for_model(
            model, mode=GlobalResidualMode.CLOSED_FORM)
        ev_coupled_ref = gr_coupled_ref.for_model(
            model, mode=GlobalResidualMode.COUPLED)

        R_closed_ref = ev_closed_ref["R"](
            params, U, U_prev, shapes_ip, w, dv, h, ip_set)
        R_coupled_ref, _, _ = ev_coupled_ref[
            "R_and_dR_dU_and_xi"](
            params, U, U_prev, xi_zeros,
            shapes_ip, w, dv, h, ip_set)

        self.assertTrue(np.allclose(
            np.asarray(R_closed[0]),
            np.asarray(R_closed_ref[0]), atol=1e-12))
        self.assertTrue(np.allclose(
            np.asarray(R_coupled[0]),
            np.asarray(R_coupled_ref[0]), atol=1e-12))


if __name__ == "__main__":
    unittest.main()
