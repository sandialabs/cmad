"""Per-element COUPLED kernel tests.

Validates per_element_R_and_K_coupled's shape contract,
local-Newton convergence behavior at every IP, equivalence with
the CLOSED_FORM kernel on a closed-form-capable Elastic model,
and FD-vs-AD on the IP-accumulated dR/dU at a J2 plastic-loading
point (composes the per-IP IFT correction validated at the
evaluator level in tests/global_residuals/test_for_model_coupled.py).
"""
import unittest
from typing import cast

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from cmad.fem.assembly import (
    per_element_R_and_K,
    per_element_R_and_K_coupled,
)
from cmad.fem.interpolants import hex_linear
from cmad.fem.quadrature import hex_quadrature
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
from cmad.typing import JaxArray, PyTreeDict
from tests.support.test_problems import J2AnalyticalProblem

# Hex with reference-frame node coordinates: iso_jac = I, so the
# kernel's physical-frame shape functions equal the reference-frame
# shapes. Lets the equilibrium test reuse hex_linear directly to
# build per-IP shape inputs without re-deriving iso_jac internals.
_HEX_REF_X = jnp.array([
    [-1.0, -1.0, -1.0],
    [+1.0, -1.0, -1.0],
    [+1.0, +1.0, -1.0],
    [-1.0, +1.0, -1.0],
    [-1.0, -1.0, +1.0],
    [+1.0, -1.0, +1.0],
    [+1.0, +1.0, +1.0],
    [-1.0, +1.0, +1.0],
])


def _make_linear_elastic_model() -> Elastic:
    values = cast(
        PyTreeDict, {"elastic": {"kappa": 100.0, "mu": 50.0}})
    active = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Elastic(
        Parameters(values, active, transforms),
        def_type=DefType.FULL_3D,
    )


def _make_J2_model() -> SmallElasticPlastic:
    """SmallElasticPlastic with the J2-Voce parameter set from the
    calibration test problem (E=200e3, nu=0.3, Y=200, Voce S=200,
    D=20). Y/E = 1e-3 sets the yield-strain reference for the
    plastic-loading helper."""
    return SmallElasticPlastic(
        J2AnalyticalProblem().J2_parameters,
        def_type=DefType.FULL_3D,
    )


def _U_elem_plastic_loading() -> JaxArray:
    """U on the reference hex driving strains a few times yield
    (Y/E = 1e-3 for the J2 problem). +x / +y / +z faces extended,
    giving non-trivial plastic activity at every IP."""
    U = jnp.zeros((8, 3))
    U = U.at[1, 0].set(0.005).at[2, 0].set(0.005)
    U = U.at[5, 0].set(0.005).at[6, 0].set(0.005)
    U = U.at[2, 1].set(0.003).at[3, 1].set(0.003)
    U = U.at[6, 1].set(0.003).at[7, 1].set(0.003)
    U = U.at[4, 2].set(0.002).at[5, 2].set(0.002)
    U = U.at[6, 2].set(0.002).at[7, 2].set(0.002)
    return U


def _U_elem_small_elastic() -> JaxArray:
    """U with strain ~0.5x yield — keeps Elastic-vs-COUPLED
    comparison well inside any plastic yield surface."""
    U = jnp.zeros((8, 3))
    U = U.at[1, 0].set(0.0005).at[2, 0].set(0.0005)
    U = U.at[5, 0].set(0.0005).at[6, 0].set(0.0005)
    U = U.at[2, 1].set(0.0003).at[3, 1].set(0.0003)
    U = U.at[6, 1].set(0.0003).at[7, 1].set(0.0003)
    return U


class _ToyEquilibrium(GlobalResidual):
    """3D u-only quasi-static equilibrium, mode-aware via
    ``self._mode``. Mirror of the same-named helper in
    tests/global_residuals/test_for_model_coupled.py — kept local
    so the kernel tests don't reach into another test module."""

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
            U_ip_prev = self.interpolate_global_fields_at_ip(
                U_prev, shapes_ip)
            if self._mode == GlobalResidualMode.CLOSED_FORM:
                sigma = model.cauchy_closed_form(
                    params, U_ip, U_ip_prev)
            else:
                sigma = model.cauchy(
                    xi, xi_prev, params, U_ip, U_ip_prev)
            R = (shapes_ip[0].grad_N @ sigma) * w * dv
            return [R]

        super().__init__(residual_fn)


def _kernel_common_kwargs(quad):
    """Per-element kwargs shared across both kernels — only the
    evaluator + xi_prev_per_ip + unravel_xi differ between modes."""
    return {
        "X_elem": _HEX_REF_X,
        "quad_xi": jnp.asarray(quad.xi),
        "quad_w": jnp.asarray(quad.w),
        "geom_interpolant_fn": hex_linear,
        "per_block_interpolant_fns": [hex_linear],
        "forcing_fns_by_block_idx": {},
        "residual_block_shapes": [(8, 3)],
        "t": 0.0,
    }


def _ref_hex_shapes_at_ip(xi_ip: JaxArray) -> ShapeFunctionsAtIP:
    """Physical-frame shapes at one IP of the reference hex.
    iso_jac = I for X_elem = reference coords, so physical = ref."""
    return hex_linear(xi_ip)


class TestPerElementCoupledShape(unittest.TestCase):
    def test_returns_R_K_xi_with_correct_shapes(self) -> None:
        gr = _ToyEquilibrium()
        model = _make_J2_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        params = model.parameters.values

        flat_init_xi, unravel_xi = ravel_pytree(model._init_xi)
        total_xi_dofs = int(flat_init_xi.shape[0])
        # SmallElasticPlastic FULL_3D: vec_cauchy(6) + alpha(1) = 7
        self.assertEqual(total_xi_dofs, 7)

        quad = hex_quadrature(degree=2)
        nips = int(quad.xi.shape[0])
        # Hex degree-2: ceil((2+1)/2) = 2 points per direction → 8
        self.assertEqual(nips, 8)

        U = [_U_elem_plastic_loading()]
        U_prev = [jnp.zeros((8, 3))]
        xi_prev_per_ip = jnp.zeros((nips, total_xi_dofs))

        R_blocks, dR_dU_blocks, xi_solved_per_ip = (
            per_element_R_and_K_coupled(
                U_elem=U,
                U_prev_elem=U_prev,
                params=params,
                xi_prev_per_ip=xi_prev_per_ip,
                R_and_dR_dU_and_xi_evaluator=evaluators[
                    "R_and_dR_dU_and_xi"],
                unravel_xi=unravel_xi,
                **_kernel_common_kwargs(quad),
            )
        )

        self.assertEqual(len(R_blocks), 1)
        self.assertEqual(R_blocks[0].shape, (8, 3))
        self.assertEqual(len(dR_dU_blocks), 1)
        self.assertEqual(len(dR_dU_blocks[0]), 1)
        self.assertEqual(dR_dU_blocks[0][0].shape, (8, 3, 8, 3))
        self.assertEqual(xi_solved_per_ip.shape, (8, 7))


class TestPerElementCoupledLocalEquilibrium(unittest.TestCase):
    """The kernel's per-IP local Newton drives
    ``model._residual(xi, xi_prev, params, U_ip, U_ip_prev) → 0``.
    Verify the converged xi at each IP (returned as
    ``xi_solved_per_ip``) actually satisfies that residual to the
    Newton's tolerance."""

    def test_xi_solved_satisfies_local_residual_at_each_ip(
            self) -> None:
        gr = _ToyEquilibrium()
        model = _make_J2_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        params = model.parameters.values

        flat_init_xi, unravel_xi = ravel_pytree(model._init_xi)
        total_xi_dofs = int(flat_init_xi.shape[0])
        quad = hex_quadrature(degree=2)
        nips = int(quad.xi.shape[0])

        U = [_U_elem_plastic_loading()]
        U_prev = [jnp.zeros((8, 3))]
        xi_prev_per_ip = jnp.zeros((nips, total_xi_dofs))

        _, _, xi_solved_per_ip = per_element_R_and_K_coupled(
            U_elem=U,
            U_prev_elem=U_prev,
            params=params,
            xi_prev_per_ip=xi_prev_per_ip,
            R_and_dR_dU_and_xi_evaluator=evaluators[
                "R_and_dR_dU_and_xi"],
            unravel_xi=unravel_xi,
            **_kernel_common_kwargs(quad),
        )

        for ip_idx in range(nips):
            shapes_ip = [_ref_hex_shapes_at_ip(
                jnp.asarray(quad.xi[ip_idx]))]
            U_ip = gr.interpolate_global_fields_at_ip(U, shapes_ip)
            U_ip_prev = gr.interpolate_global_fields_at_ip(
                U_prev, shapes_ip)

            xi_blocks = unravel_xi(xi_solved_per_ip[ip_idx])
            xi_prev_blocks = unravel_xi(xi_prev_per_ip[ip_idx])
            residual = model._residual(
                xi_blocks, xi_prev_blocks,
                params, U_ip, U_ip_prev,
            )
            self.assertLess(
                float(jnp.linalg.norm(residual)), 1e-10)

        # Sanity: at least one IP in the plastic regime (alpha > 0)
        # so the test isn't accidentally elastic-only.
        alpha_per_ip = xi_solved_per_ip[:, 6]
        self.assertTrue(
            bool(jnp.any(alpha_per_ip > 0.0)),
            "expected at least one IP in plastic regime",
        )


class TestPerElementCoupledClosedFormEquivalence(unittest.TestCase):
    """Elastic in COUPLED matches the CLOSED_FORM kernel's R + K
    on the same per-element problem. Elastic's local Newton is
    linear in xi[0] and converges in one iteration → IFT-corrected
    total dR/dU equals the direct CLOSED_FORM derivative; per-IP
    equivalence is 1e-12 (verified in test_for_model_coupled.py),
    kernel-level loosens by ~one order to absorb sum-over-8-IPs
    accumulation noise."""

    def test_R_and_K_match_to_1e_minus_10(self) -> None:
        # One GR per mode — for_model mutates self._mode, so we
        # need two instances to keep both bindings live.
        gr_closed = _ToyEquilibrium()
        gr_coupled = _ToyEquilibrium()
        model = _make_linear_elastic_model()
        ev_closed = gr_closed.for_model(
            model, mode=GlobalResidualMode.CLOSED_FORM)
        ev_coupled = gr_coupled.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        params = model.parameters.values

        flat_init_xi, unravel_xi = ravel_pytree(model._init_xi)
        total_xi_dofs = int(flat_init_xi.shape[0])
        # Elastic FULL_3D: vec_cauchy only, no plastic state.
        self.assertEqual(total_xi_dofs, 6)

        quad = hex_quadrature(degree=2)
        nips = int(quad.xi.shape[0])

        U = [_U_elem_small_elastic()]
        U_prev = [jnp.zeros((8, 3))]
        xi_prev_per_ip = jnp.zeros((nips, total_xi_dofs))
        common = _kernel_common_kwargs(quad)

        R_closed, dR_dU_closed = per_element_R_and_K(
            U_elem=U,
            U_prev_elem=U_prev,
            params=params,
            R_and_dR_dU_evaluator=ev_closed["R_and_dR_dU"],
            **common,
        )
        R_coupled, dR_dU_coupled, _ = per_element_R_and_K_coupled(
            U_elem=U,
            U_prev_elem=U_prev,
            params=params,
            xi_prev_per_ip=xi_prev_per_ip,
            R_and_dR_dU_and_xi_evaluator=ev_coupled[
                "R_and_dR_dU_and_xi"],
            unravel_xi=unravel_xi,
            **common,
        )

        self.assertTrue(jnp.allclose(
            R_closed[0], R_coupled[0], rtol=0., atol=1e-10))
        self.assertTrue(jnp.allclose(
            dR_dU_closed[0][0], dR_dU_coupled[0][0],
            rtol=0., atol=1e-10))


class TestPerElementCoupledJVPvsFD(unittest.TestCase):
    """Forward-mode AD on the kernel's accumulated dR/dU at a J2
    plastic-loading point matches central-difference FD across all
    24 (basis_fn, component) U indices. The per-IP IFT correction
    is validated at the evaluator level in test_for_model_coupled.py;
    this test composes that correctness across the
    sum-over-8-IPs that the kernel performs."""

    def test_dR_dU_total_matches_central_fd(self) -> None:
        gr = _ToyEquilibrium()
        model = _make_J2_model()
        evaluators = gr.for_model(
            model, mode=GlobalResidualMode.COUPLED)
        params = model.parameters.values

        _, unravel_xi = ravel_pytree(model._init_xi)
        quad = hex_quadrature(degree=2)
        nips = int(quad.xi.shape[0])
        total_xi_dofs = 7

        U_arr = _U_elem_plastic_loading()
        U_prev = [jnp.zeros((8, 3))]
        xi_prev_per_ip = jnp.zeros((nips, total_xi_dofs))
        common = _kernel_common_kwargs(quad)

        def kernel_R_and_K(U_arr_in: JaxArray):
            return per_element_R_and_K_coupled(
                U_elem=[U_arr_in],
                U_prev_elem=U_prev,
                params=params,
                xi_prev_per_ip=xi_prev_per_ip,
                R_and_dR_dU_and_xi_evaluator=evaluators[
                    "R_and_dR_dU_and_xi"],
                unravel_xi=unravel_xi,
                **common,
            )

        _R, dR_dU_blocks, _ = kernel_R_and_K(U_arr)
        ad_arr = np.asarray(dR_dU_blocks[0][0])  # (8, 3, 8, 3)

        eps = 1e-6
        fd_arr = np.zeros_like(ad_arr)
        for b in range(8):
            for k in range(3):
                R_plus, _, _ = kernel_R_and_K(
                    U_arr.at[b, k].add(eps))
                R_minus, _, _ = kernel_R_and_K(
                    U_arr.at[b, k].add(-eps))
                fd_arr[:, :, b, k] = (
                    R_plus[0] - R_minus[0]) / (2 * eps)

        self.assertTrue(np.allclose(
            ad_arr, fd_arr, rtol=1e-5, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
