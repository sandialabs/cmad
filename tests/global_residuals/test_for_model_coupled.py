"""Shape contract + Elastic equivalence tests for GlobalResidual.for_model COUPLED mode.

Bundles a small toy 3D u-only quasi-static equilibrium GR (mirror
of test_abc_contract.py's _ToyEquilibrium, mode-aware via
self._mode) and exercises the COUPLED branch's 7-key dict shape,
mixed argument signatures (9-arg raw vs 8-arg Newton-running), and
the R + dR/dU equivalence between CLOSED_FORM and COUPLED bindings
on a closed-form-capable Elastic model.
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
from cmad.models.var_types import VarType
from cmad.parameters.parameters import Parameters
from cmad.typing import PyTreeDict


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


class _ToyEquilibrium(GlobalResidual):
    """3D u-only quasi-static equilibrium, small-def kinematics.

    Mode-aware via ``self._mode``: CLOSED_FORM dispatches to
    ``model.cauchy_closed_form``, COUPLED dispatches to
    ``model.cauchy(xi, xi_prev, ...)``. Mirror of the same-named
    class in ``test_abc_contract.py``.
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
                        model, shapes_ip, w, dv, ip_set):
            U_ip = self.interpolate_global_fields_at_ip(U, shapes_ip)
            U_ip_prev = self.interpolate_global_fields_at_ip(U_prev, shapes_ip)
            if self._mode == GlobalResidualMode.CLOSED_FORM:
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
    ip_set = 0
    return params, U, U_prev, shapes_ip, w, dv, ip_set


class TestForModelCoupledShape(unittest.TestCase):
    def test_coupled_returns_seven_keys(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        self.assertEqual(
            set(evaluators.keys()),
            {
                "R", "dR_dU_prev", "dR_dp",
                "dR_dxi", "dR_dxi_prev",
                "dR_dU", "R_and_dR_dU_and_xi",
            },
        )

    def test_raw_evaluators_callable_with_9arg_sig(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        params, U, U_prev, shapes_ip, w, dv, ip_set = (
            _test_inputs(model))
        xi = [jnp.zeros_like(b) for b in model._init_xi]
        xi_prev = [jnp.zeros_like(b) for b in model._init_xi]

        # All 5 raw evaluators take 9 args:
        # (params, U, U_prev, xi, xi_prev, shapes_ip, w, dv, ip_set).
        for key in ("R", "dR_dU_prev", "dR_dp",
                    "dR_dxi", "dR_dxi_prev"):
            evaluators[key](
                params, U, U_prev, xi, xi_prev,
                shapes_ip, w, dv, ip_set,
            )

    def test_total_evaluators_callable_with_8arg_sig(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        params, U, U_prev, shapes_ip, w, dv, ip_set = (
            _test_inputs(model))
        xi_prev = [jnp.zeros_like(b) for b in model._init_xi]

        # dR_dU and R_and_dR_dU_and_xi take 8 args:
        # (params, U, U_prev, xi_prev, shapes_ip, w, dv, ip_set).
        for key in ("dR_dU", "R_and_dR_dU_and_xi"):
            evaluators[key](
                params, U, U_prev, xi_prev,
                shapes_ip, w, dv, ip_set,
            )

    def test_R_and_dR_dU_and_xi_returns_triple(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        params, U, U_prev, shapes_ip, w, dv, ip_set = (
            _test_inputs(model))
        xi_prev = [jnp.zeros_like(b) for b in model._init_xi]

        R_blocks, dR_dU_blocks, xi = evaluators["R_and_dR_dU_and_xi"](
            params, U, U_prev, xi_prev,
            shapes_ip, w, dv, ip_set,
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
        # One GR per mode — ``for_model`` mutates ``self._mode``.
        gr_closed = _ToyEquilibrium()
        gr_coupled = _ToyEquilibrium()
        model = _make_linear_elastic_model()

        ev_closed = gr_closed.for_model(
            model, mode=GlobalResidualMode.CLOSED_FORM)
        ev_coupled = gr_coupled.for_model(
            model, mode=GlobalResidualMode.COUPLED)

        params, U, U_prev, shapes_ip, w, dv, ip_set = (
            _test_inputs(model))
        xi_prev = [jnp.zeros_like(b) for b in model._init_xi]

        R_closed, dR_dU_closed = ev_closed["R_and_dR_dU"](
            params, U, U_prev, shapes_ip, w, dv, ip_set,
        )
        R_coupled, dR_dU_coupled, _xi = ev_coupled[
            "R_and_dR_dU_and_xi"](
            params, U, U_prev, xi_prev,
            shapes_ip, w, dv, ip_set,
        )

        self.assertTrue(jnp.allclose(
            R_closed[0], R_coupled[0], rtol=0., atol=1e-12))
        self.assertTrue(jnp.allclose(
            dR_dU_closed[0][0], dR_dU_coupled[0][0],
            rtol=0., atol=1e-12))


if __name__ == "__main__":
    unittest.main()
