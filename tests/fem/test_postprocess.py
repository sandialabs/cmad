"""Tests for :mod:`cmad.fem.postprocess`.

Verifies CLOSED_FORM and COUPLED dispatch in
:func:`evaluate_cauchy_at_ips`, and the natural pipe-through to
:func:`cmad.io.results.ip_average_to_element` for element-averaged
stress output.
"""
import unittest

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEState, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.fem.postprocess import evaluate_cauchy_at_ips
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.io.results import ip_average_to_element
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.elastic_stress import isotropic_linear_elastic_cauchy_stress
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.models.var_types import get_vector_from_sym_tensor
from cmad.parameters.parameters import Parameters
from tests.support.test_problems import J2AnalyticalProblem


def _elastic_parameters(kappa: float = 100.0, mu: float = 50.0) -> Parameters:
    values = {"elastic": {"kappa": kappa, "mu": mu}}
    active = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Parameters(values, active, transforms)


def _build_elastic_problem(mode: GlobalResidualMode):
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    dof_map = build_dof_map(
        mesh, [layout], [], components_by_field={"u": 3},
    )
    gr = SmallDispEquilibrium(ndims=3)
    model = Elastic(_elastic_parameters(), def_type=DefType.FULL_3D)
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": mode},
    )


def _build_plastic_problem():
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    dof_map = build_dof_map(
        mesh, [layout], [], components_by_field={"u": 3},
    )
    gr = SmallDispEquilibrium(ndims=3)
    model = SmallRateElasticPlastic(
        J2AnalyticalProblem().J2_parameters,
        def_type=DefType.FULL_3D,
    )
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.COUPLED},
    )


def _U_uniaxial_x(mesh, eps: float) -> np.ndarray:
    """Linear-ramp displacement: u_x = eps * x (others zero).

    For a Q1 hex on [0, 1]^3, ∂u_x/∂x = eps uniformly across the
    element, so cauchy is uniform across IPs.
    """
    n_nodes = mesh.nodes.shape[0]
    U = np.zeros(n_nodes * 3)
    U[0::3] = eps * mesh.nodes[:, 0]
    return U


def _analytical_cauchy_uniaxial_6vec(eps: float) -> np.ndarray:
    """Closed-form linear-elastic cauchy 6-vec (cmad-internal order)
    for uniaxial-x small strain ``ε``."""
    F = jnp.eye(3) + jnp.diag(jnp.array([eps, 0.0, 0.0]))
    sigma = isotropic_linear_elastic_cauchy_stress(
        F, {"elastic": {"kappa": 100.0, "mu": 50.0}},
    )
    return np.asarray(get_vector_from_sym_tensor(sigma, 3))


def _mises(cauchy_6vec: np.ndarray) -> float:
    sxx, sxy, sxz, syy, syz, szz = cauchy_6vec
    tr = (sxx + syy + szz) / 3.0
    dev = np.array([
        [sxx - tr, sxy, sxz],
        [sxy, syy - tr, syz],
        [sxz, syz, szz - tr],
    ])
    return float(np.sqrt(1.5 * np.sum(dev ** 2)))


class TestCauchyAtIpsClosedForm(unittest.TestCase):

    def test_uniaxial_strain_matches_analytical_at_every_ip(self):
        eps = 0.01
        fe_problem = _build_elastic_problem(GlobalResidualMode.CLOSED_FORM)
        fe_state = FEState.from_problem(fe_problem)
        fe_state.U_history[0] = _U_uniaxial_x(fe_problem.mesh, eps)

        cauchy = evaluate_cauchy_at_ips(fe_problem, fe_state, 0, "all")
        n_elems, n_ips, ncomp = cauchy.shape
        self.assertEqual((n_elems, ncomp), (1, 6))
        expected = _analytical_cauchy_uniaxial_6vec(eps)
        for e in range(n_elems):
            for p in range(n_ips):
                np.testing.assert_allclose(
                    cauchy[e, p], expected, rtol=1e-12, atol=1e-12,
                )

    def test_zero_displacement_returns_zero_cauchy(self):
        fe_problem = _build_elastic_problem(GlobalResidualMode.CLOSED_FORM)
        fe_state = FEState.from_problem(fe_problem)
        cauchy = evaluate_cauchy_at_ips(fe_problem, fe_state, 0, "all")
        np.testing.assert_allclose(cauchy, 0.0, atol=1e-12)


class TestCauchyAtIpsCoupledDispatch(unittest.TestCase):
    """COUPLED-mode dispatch threads xi through the postprocess path:
    Elastic._cauchy_fn returns xi[0] as a sym tensor, so injecting the
    analytical 6-vec into xi[0] and reading back via
    evaluate_cauchy_at_ips reproduces the CLOSED_FORM result."""

    def test_xi_carrying_cauchy_round_trips_to_closed_form_value(self):
        eps = 0.01
        fe_problem_cf = _build_elastic_problem(
            GlobalResidualMode.CLOSED_FORM,
        )
        fe_state_cf = FEState.from_problem(fe_problem_cf)
        fe_state_cf.U_history[0] = _U_uniaxial_x(
            fe_problem_cf.mesh, eps,
        )
        cauchy_cf = evaluate_cauchy_at_ips(
            fe_problem_cf, fe_state_cf, 0, "all",
        )

        fe_problem_c = _build_elastic_problem(GlobalResidualMode.COUPLED)
        fe_state_c = FEState.from_problem(fe_problem_c)
        fe_state_c.U_history[0] = _U_uniaxial_x(
            fe_problem_c.mesh, eps,
        )
        # Elastic FULL_3D: _init_xi = [zeros(6)] -> total 6 xi DOFs;
        # xi[..., :6] is the full sym-tensor cauchy slot.
        analytical = _analytical_cauchy_uniaxial_6vec(eps)
        xi = fe_state_c.xi_history_by_block["all"][0]
        xi[..., :6] = analytical
        cauchy_c = evaluate_cauchy_at_ips(
            fe_problem_c, fe_state_c, 0, "all",
        )
        np.testing.assert_allclose(
            cauchy_c, cauchy_cf, rtol=1e-12, atol=1e-12,
        )


class TestCauchyAtIpsPlasticAboveYield(unittest.TestCase):
    """SmallRateElasticPlastic FULL_3D: xi pytree is [vec_cauchy(6),
    alpha(1)] → 7 flat dofs. Inject a uniaxial 300 MPa stress (Y=200
    for the J2 fixture) and verify Mises > Y at every IP, and that
    the unravel_xi closure correctly handles the 2-block pytree."""

    def test_uniaxial_300mpa_returns_mises_above_yield(self):
        fe_problem = _build_plastic_problem()
        fe_state = FEState.from_problem(fe_problem)
        sigma_xx = 300.0
        analytical = np.array(
            [sigma_xx, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        xi = fe_state.xi_history_by_block["all"][0]
        xi[..., :6] = analytical
        cauchy = evaluate_cauchy_at_ips(
            fe_problem, fe_state, 0, "all",
        )
        n_elems, n_ips, ncomp = cauchy.shape
        self.assertEqual(ncomp, 6)
        self.assertGreater(n_ips, 0)
        Y = 200.0
        for e in range(n_elems):
            for p in range(n_ips):
                self.assertGreater(_mises(cauchy[e, p]), Y)


class TestCauchyAtIpsComposeIpAverage(unittest.TestCase):

    def test_pipe_through_ip_average_recovers_uniform_cauchy(self):
        # Uniaxial-strain U on a single Q1 hex on [0,1]^3 produces a
        # uniform cauchy across IPs; ip_average_to_element collapses
        # the IP axis and the per-element value equals the analytical
        # cauchy.
        eps = 0.01
        fe_problem = _build_elastic_problem(GlobalResidualMode.CLOSED_FORM)
        fe_state = FEState.from_problem(fe_problem)
        fe_state.U_history[0] = _U_uniaxial_x(fe_problem.mesh, eps)
        cauchy_ip = evaluate_cauchy_at_ips(
            fe_problem, fe_state, 0, "all",
        )
        cauchy_elem = ip_average_to_element(
            cauchy_ip, fe_problem.geometry_cache, "all",
        )
        n_elems = cauchy_ip.shape[0]
        self.assertEqual(cauchy_elem.shape, (n_elems, 6))
        expected = _analytical_cauchy_uniaxial_6vec(eps)
        for e in range(n_elems):
            np.testing.assert_allclose(
                cauchy_elem[e], expected, rtol=1e-12, atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
