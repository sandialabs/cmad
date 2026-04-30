"""Contract tests for the GlobalResidual ABC via a test-local subclass.

The subclass mirrors a small-deformation quasi-static equilibrium
residual on a linear tetrahedron at its barycenter integration point
and pairs with a real Elastic model in CLOSED_FORM mode. This exercises
the full for_model pipeline — closure-dict keys, capability gating,
AD gradients vs central FD, and one Newton step for a linear problem.
"""
import unittest
from typing import ClassVar, cast

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

    Residual form: ``R[a, i] = grad_N[a, j] * sigma[j, i] * w * dv``
    evaluated at a single barycenter IP of a canonical linear tet.
    Cauchy stress comes from ``model.cauchy_closed_form``
    (CLOSED_FORM-only shape).
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
            sigma = model.cauchy_closed_form(params, U_ip, U_ip_prev)
            R = (shapes_ip[0].grad_N @ sigma) * w * dv   # (4, 3)
            return [R]

        super().__init__(residual_fn)


def _test_inputs(model: Elastic):
    """Build a nontrivial element-level evaluation point with probe
    displacements at two different nodes + components."""
    U = [jnp.zeros((4, 3)).at[1, 0].set(0.001).at[2, 1].set(0.0005)]
    U_prev = [jnp.zeros((4, 3))]
    params = model.parameters.values
    shapes_ip = [_tet_barycenter_shapes()]
    w = 1.0
    dv = 1.0 / 6.0
    ip_set = 0
    return params, U, U_prev, shapes_ip, w, dv, ip_set


class TestGlobalResidualABC(unittest.TestCase):
    def test_for_model_closed_form_keys(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.CLOSED_FORM)
        self.assertEqual(
            set(evaluators.keys()),
            {"R", "R_and_dR_dU", "dR_dU", "dR_dU_prev", "dR_dp"},
        )

    def test_for_model_coupled_raises_not_implemented(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        with self.assertRaises(NotImplementedError) as ctx:
            gr.for_model(model, mode=GlobalResidualMode.COUPLED)
        msg = str(ctx.exception)
        self.assertIn("COUPLED", msg)
        self.assertIn("custom_vjp", msg)

    def test_for_model_closed_form_rejects_incapable_model(self):
        gr = _ToyEquilibrium()

        class _ElasticWithoutClosedForm(Elastic):
            supports_closed_form_cauchy: ClassVar[bool] = False

        model = _ElasticWithoutClosedForm(
            _make_parameters(), def_type=DefType.FULL_3D)

        with self.assertRaises(ValueError) as ctx:
            gr.for_model(model, mode=GlobalResidualMode.CLOSED_FORM)
        msg = str(ctx.exception)
        self.assertIn("CLOSED_FORM", msg)
        self.assertIn("supports_closed_form_cauchy", msg)

    def test_ad_matches_fd_on_dR_dU_closed_form(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.CLOSED_FORM)
        params, U, U_prev, shapes_ip, w, dv, ip_set = (
            _test_inputs(model))

        _, dR_dU = evaluators["R_and_dR_dU"](
            params, U, U_prev, shapes_ip, w, dv, ip_set)
        ad_arr = np.asarray(dR_dU[0][0])          # (4, 3, 4, 3)

        eps = 1e-6
        fd_arr = np.zeros_like(ad_arr)
        for b in range(4):
            for k in range(3):
                U_plus = [U[0].at[b, k].add(eps)]
                U_minus = [U[0].at[b, k].add(-eps)]
                R_plus = evaluators["R"](
                    params, U_plus, U_prev,
                    shapes_ip, w, dv, ip_set)
                R_minus = evaluators["R"](
                    params, U_minus, U_prev,
                    shapes_ip, w, dv, ip_set)
                fd_arr[:, :, b, k] = (R_plus[0] - R_minus[0]) / (2 * eps)

        self.assertTrue(jnp.allclose(
            ad_arr, fd_arr, rtol=1e-5, atol=1e-10))

    def test_newton_step_converges_closed_form(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.CLOSED_FORM)
        params, U, U_prev, shapes_ip, w, dv, ip_set = (
            _test_inputs(model))

        R0_blocks, dR_dU = evaluators["R_and_dR_dU"](
            params, U, U_prev, shapes_ip, w, dv, ip_set)
        R0 = R0_blocks[0]
        K_full = dR_dU[0][0]
        # Free DOFs: node 3. K_ff: (3, 3). R_f: (3,).
        K_ff = K_full[3, :, 3, :]
        R_f = R0[3, :]
        dU_f = -jnp.linalg.solve(K_ff, R_f)

        U_new = [U[0].at[3].add(dU_f)]
        R1_blocks = evaluators["R"](
            params, U_new, U_prev, shapes_ip, w, dv, ip_set)
        self.assertLess(float(jnp.linalg.norm(R1_blocks[0][3, :])), 1e-10)

    def test_dR_dU_standalone_returns_block_pair_structure(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.CLOSED_FORM)
        args = _test_inputs(model)
        result = evaluators["dR_dU"](*args)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0].shape, (4, 3, 4, 3))

    def test_dR_dU_prev_standalone_returns_block_pair_structure(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.CLOSED_FORM)
        args = _test_inputs(model)
        result = evaluators["dR_dU_prev"](*args)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0].shape, (4, 3, 4, 3))

    def test_dR_dp_returns_residual_block_pytree_parallel_to_params(self):
        gr = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.CLOSED_FORM)
        args = _test_inputs(model)
        result = evaluators["dR_dp"](*args)
        self.assertEqual(len(result), 1)
        self.assertIn("elastic", result[0])
        self.assertIn("kappa", result[0]["elastic"])
        self.assertIn("mu", result[0]["elastic"])


if __name__ == "__main__":
    unittest.main()
