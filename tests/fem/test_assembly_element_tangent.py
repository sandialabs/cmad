"""``assemble_element_tangent`` against ``assemble_global``.

The per element tangent blocks, summed onto the deduplicated pattern,
reproduce the assembled ``K``; the element matvec built from them
reproduces ``K @ x``; ``R`` and ``xi_solved`` are the same arrays. Checked
on a mixed u-p problem at a plastic state (two residual blocks, so the
``(r, s)`` block layout is exercised) and on a two element block elastic
mesh (so the per block emit order is exercised).
"""
import unittest

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.assembly import (
    assemble_element_tangent,
    assemble_global,
    params_by_block_from_models,
)
from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.mesh import StructuredHexMesh
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.deformation_types import DefType
from cmad.models.global_fields import StepTime
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from tests.fem.test_assembly_coupled import (
    _build_fe_problem,
    _split_into_two_blocks,
)
from tests.fem.test_mixed_up_plastic import (
    MAX_ALPHA,
    NUM_DRIVE_STEPS,
    _build_mixed_fe,
)
from tests.support.test_problems import J2AnalyticalProblem


def _element_matvec(
        fe_problem: FEProblem, K_elem_by_block, x: NDArray[np.floating],
) -> NDArray[np.floating]:
    """``K @ x`` from the per element blocks, scattered in numpy."""
    arrays = fe_problem.kernel_arrays
    y = np.zeros(fe_problem.dof_map.num_total_dofs)
    for block, K_blocks in K_elem_by_block.items():
        eqs = [np.asarray(e) for e in arrays.r_scatter_eq_by_block[block]]
        for r, K_r in enumerate(K_blocks):
            for s, K_rs in enumerate(K_r):
                np.add.at(y, eqs[r], np.einsum(
                    "eij,ej->ei", np.asarray(K_rs), x[eqs[s]]))
    return y


class TestElementTangentMatchesAssembled(unittest.TestCase):

    def _check(self, fe_problem, U, U_prev, xi_prev, step_time) -> None:
        params = params_by_block_from_models(fe_problem)
        arrays = fe_problem.kernel_arrays
        K_bcoo, R, xi, _ = assemble_global(
            fe_problem, arrays, params, U, U_prev, step_time,
            xi_prev_by_block=xi_prev,
        )
        K_elem_by_block, R_elem, xi_elem, _ = assemble_element_tangent(
            fe_problem, arrays, params, U, U_prev, step_time,
            xi_prev_by_block=xi_prev,
        )
        np.testing.assert_array_equal(np.asarray(R_elem), np.asarray(R))
        self.assertEqual(set(xi_elem), set(xi))
        for block in xi:
            np.testing.assert_array_equal(
                np.asarray(xi_elem[block]), np.asarray(xi[block]),
            )
        self.assertEqual(list(K_elem_by_block), list(fe_problem.evaluators_by_block))
        for block, K_blocks in K_elem_by_block.items():
            eqs = arrays.r_scatter_eq_by_block[block]
            for r, K_r in enumerate(K_blocks):
                for s, K_rs in enumerate(K_r):
                    self.assertEqual(
                        K_rs.shape,
                        (eqs[r].shape[0], eqs[r].shape[1], eqs[s].shape[1]),
                    )

        # The blocks raveled in emit order and summed through the dedup
        # scatter are the deduplicated data.
        raw = np.concatenate([
            np.asarray(K_rs).ravel()
            for K_blocks in K_elem_by_block.values()
            for K_r in K_blocks for K_rs in K_r
        ])
        K_data = np.asarray(K_bcoo.data)
        dedup = np.zeros(K_data.shape[0])
        np.add.at(dedup, np.asarray(arrays.coo_dedup_scatter), raw)
        scale = np.abs(K_data).max()
        np.testing.assert_allclose(dedup, K_data, rtol=0, atol=1e-14 * scale)

        # The element matvec is the assembled matvec.
        n = fe_problem.dof_map.num_total_dofs
        x = np.random.default_rng(0).standard_normal(n)
        y_ref = np.asarray(K_bcoo @ jnp.asarray(x))
        y = _element_matvec(fe_problem, K_elem_by_block, x)
        np.testing.assert_allclose(
            y, y_ref, rtol=0, atol=1e-13 * np.abs(y_ref).max(),
        )
        self.assertGreater(np.abs(y_ref).max(), 0.0)

    def test_mixed_up_plastic_state(self) -> None:
        # The uniaxial cube of tests/fem/test_mixed_up_plastic.py, pulled
        # in two load steps of the same size that test uses (it takes five),
        # so the state checked is plastic and the tangent carries the
        # return map terms in every (r, s) block. Plasticity is asserted
        # below, not assumed.
        problem = J2AnalyticalProblem()
        stress_mask = np.zeros((3, 3))
        stress_mask[0, 0] = 1.0
        _, strain, _ = problem.analytical_solution(
            stress_mask, MAX_ALPHA, num_steps=2,
        )
        axial_strain = float(strain[0, 0, -1])
        model = SmallElasticPlastic(
            problem.J2_parameters, def_type=DefType.FULL_3D,
        )
        fe_problem = _build_mixed_fe(model)
        params = params_by_block_from_models(fe_problem)
        state = FEState.from_problem(fe_problem)
        U = state.U_at(0)
        xi = {"all": state.xi_at(0, "all")}
        t_prev = 0.0
        for step in range(1, 3):
            U_prev, xi_prev = U, xi
            t = axial_strain * step / NUM_DRIVE_STEPS
            U, xi = fe_newton_solve(
                fe_problem, params, U_prev=U_prev, t=t, t_prev=t_prev,
                xi_prev_by_block=xi_prev,
            )
            t_prev = t
        alpha_slot = int(np.asarray(model._init_xi[0]).shape[0])
        self.assertGreater(
            float(np.asarray(xi["all"])[..., alpha_slot].max()), 0.0,
        )
        self._check(
            fe_problem, U, U_prev, xi_prev, StepTime(t=t, t_prev=t_prev),
        )

    def test_two_element_blocks(self) -> None:
        mesh = _split_into_two_blocks(
            StructuredHexMesh(lengths=(1.0, 1.0, 1.0), divisions=(2, 2, 2)),
        )
        fe_problem = _build_fe_problem(
            mesh, {"left": GlobalResidualMode.CLOSED_FORM,
                   "right": GlobalResidualMode.COUPLED},
        )
        n = fe_problem.dof_map.num_total_dofs
        rng = np.random.default_rng(1)
        U = 0.01 * rng.standard_normal(n)
        U_prev = np.zeros(n)
        state = FEState.from_problem(fe_problem)
        xi_prev = {"right": state.xi_at(0, "right")}
        self._check(fe_problem, U, U_prev, xi_prev, StepTime(1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
