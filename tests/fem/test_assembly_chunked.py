"""The assembly in chunks of elements matches the assembly of a block at
once: ``assemble_global``, ``assemble_element_tangent``, and
``assemble_global_residual`` at a plastic state of the mixed u-p cube, the
gradient of a trajectory cost through the chunked assembly, and the input
file key. The chunked assembly on several devices is in
``test_sharding.py``."""
import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from cmad.fem.assembly import (
    assemble_element_tangent,
    assemble_global,
    assemble_global_residual,
)
from cmad.fem.driver import build_fe_quasistatic_trajectory
from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.sharding import place_element_leaves
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.io.deck import apply_deck_defaults
from cmad.io.schema import validate_deck
from tests.fem.test_fem_fd_checks import _build_fe_problem_2x2x2, _make_J2_model
from tests.fem.test_sharding import drive_mixed_problem
from tests.io.test_schema_fe import _minimal_fe_deck

# Seven elements per chunk on the eight element cube: two chunks, the
# second one padded with six copies of the last element.
_CHUNK = 7


def _chunked(fe_problem: FEProblem, chunk: int) -> FEProblem:
    return dataclasses.replace(fe_problem, elements_per_chunk=chunk)


class TestChunkedAssembly(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        (cls.fe_problem, cls.params, cls.U, cls.U_prev, cls.xi,
         cls.xi_prev, cls.step_time) = drive_mixed_problem()
        cls.chunked = _chunked(cls.fe_problem, _CHUNK)

    def _xi_prev(self, fe_problem: FEProblem) -> dict[str, jax.Array]:
        return place_element_leaves(
            {b: jnp.asarray(x) for b, x in self.xi_prev.items()}, fe_problem,
        )

    def test_padding(self) -> None:
        self.assertEqual(self.chunked.n_elems_padded_by_block, {"all": 14})
        self.assertEqual(self.fe_problem.n_elems_padded_by_block, {"all": 8})
        xi = self._xi_prev(self.chunked)["all"]
        self.assertEqual(xi.shape[0], 14)

    def test_assemble_global_matches(self) -> None:
        outputs = []
        for fe_problem in (self.fe_problem, self.chunked):
            K, R, xi, scale = jax.jit(
                lambda arrays, xi_prev, fe_problem=fe_problem: assemble_global(
                    fe_problem, arrays, self.params, self.U, self.U_prev,
                    self.step_time, xi_prev_by_block=xi_prev,
                ),
            )(fe_problem.kernel_arrays, self._xi_prev(fe_problem))
            n = fe_problem.n_elems_by_block["all"]
            outputs.append((
                np.asarray(K.data), np.asarray(R), np.asarray(xi["all"][:n]),
                float(scale),
            ))
        (K_ref, R_ref, xi_ref, scale_ref), (K_c, R_c, xi_c, scale_c) = outputs
        np.testing.assert_allclose(K_c, K_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(R_c, R_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(xi_c, xi_ref, rtol=1e-13, atol=1e-15)
        self.assertAlmostEqual(scale_c, scale_ref, delta=1e-13 * scale_ref)

    def test_element_tangent_and_residual_match(self) -> None:
        outputs = []
        for fe_problem in (self.fe_problem, self.chunked):
            K_elem, R, _, _ = jax.jit(
                lambda arrays, xi_prev, fe_problem=fe_problem: (
                    assemble_element_tangent(
                        fe_problem, arrays, self.params, self.U, self.U_prev,
                        self.step_time, xi_prev_by_block=xi_prev,
                    )
                ),
            )(fe_problem.kernel_arrays, self._xi_prev(fe_problem))
            R_only = jax.jit(
                lambda arrays, xi_prev, fe_problem=fe_problem: (
                    assemble_global_residual(
                        fe_problem, arrays, self.params, self.U, self.U_prev,
                        self.step_time, xi_prev_by_block=xi_prev,
                    )
                ),
            )(fe_problem.kernel_arrays, self._xi_prev(fe_problem))
            n = fe_problem.n_elems_by_block["all"]
            outputs.append((
                [[np.asarray(K_rs[:n]) for K_rs in K_r] for K_r in K_elem["all"]],
                np.asarray(R), np.asarray(R_only),
            ))
        (K_ref, R_ref, R_only_ref), (K_c, R_c, R_only_c) = outputs
        for K_r_ref, K_r_c in zip(K_ref, K_c, strict=True):
            for K_rs_ref, K_rs_c in zip(K_r_ref, K_r_c, strict=True):
                np.testing.assert_allclose(K_rs_c, K_rs_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(R_c, R_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(R_only_c, R_only_ref, rtol=1e-13, atol=1e-13)

    def test_trajectory_gradient_matches(self) -> None:
        """The gradient through the checkpointed chunk scan, on the J2
        cube of the FD checks (its boundary conditions trace) over an
        elastic and a plastic step."""
        model = _make_J2_model()
        unchunked = _build_fe_problem_2x2x2(
            model, GlobalResidualMode.COUPLED, 2e-3,
        )
        t_schedule = jnp.asarray([0.0, 0.4, 1.0])
        grads = []
        for fe_problem in (unchunked, _chunked(unchunked, _CHUNK)):
            n_dofs = fe_problem.dof_map.num_total_dofs
            state = FEState.from_problem(fe_problem)
            xi_init = place_element_leaves(
                {"all": jnp.asarray(state.xi_at(0, "all"))}, fe_problem,
            )
            trajectory = build_fe_quasistatic_trajectory(
                fe_problem,
                nonlinear_solver_settings={
                    "max iters": 30, "abs tol": 1e-10, "rel tol": 1e-10,
                },
            )

            def cost(params, fe_problem=fe_problem, trajectory=trajectory,
                     xi_init=xi_init, n_dofs=n_dofs):
                U_steps, _, _, _, _, _ = trajectory(
                    fe_problem.kernel_arrays, {"all": params},
                    (jnp.zeros(n_dofs), xi_init), t_schedule,
                )
                return jnp.sum(U_steps ** 2)

            grads.append(jax.jit(jax.grad(cost))(model.parameters.values))
        ref, chunked = (ravel_pytree(g)[0] for g in grads)
        np.testing.assert_allclose(
            np.asarray(chunked), np.asarray(ref), rtol=1e-10, atol=1e-14,
        )

    def test_deck_key(self) -> None:
        deck = _minimal_fe_deck()
        deck["discretization"]["elements per chunk"] = 1000
        validate_deck(apply_deck_defaults(deck), "primal")
        deck["discretization"]["elements per chunk"] = 0
        with self.assertRaises(ValueError):
            validate_deck(apply_deck_defaults(deck), "primal")
