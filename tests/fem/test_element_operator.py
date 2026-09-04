"""``ElementOperator`` against ``AssembledOperator``.

The element operator is built from ``assemble_element_tangent``, the
assembled one from ``assemble_global`` on the enforced data. At a plastic
state of the mixed u-p cube (two fields, so every field block is
exercised) they must agree on: the global matvec, the raw matvec, the
diagonal, every field block matvec both ways, the block diagonals, and
the embedded residual. A two element block elastic mesh covers the per
block emit order. One Newton step through ``fe_newton_solve`` with
``operator: element`` must reproduce the assembled step to the solver
tolerance.
"""
import unittest

import jax.numpy as jnp
import numpy as np

from cmad.fem.assembly import (
    assemble_element_tangent,
    assemble_global,
    params_by_block_from_models,
)
from cmad.fem.fe_problem import FEState
from cmad.fem.mesh import StructuredHexMesh
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.fem.sparse_solve import (
    AssembledOperator,
    ElementOperator,
    _embedded_bc_enforce,
    _embedded_residual,
)
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.global_fields import StepTime
from tests.fem.test_assembly_coupled import (
    _build_fe_problem,
    _split_into_two_blocks,
)
from tests.fem.test_mixed_up_plastic import _JAX_BLOCK_CHEBYSHEV_SETTINGS
from tests.fem.test_sharding import drive_mixed_problem


def _operators(fe_problem, params, U, U_prev, xi_prev, step_time):
    """``(assembled, element, K_bcoo)`` at one state."""
    arrays = fe_problem.kernel_arrays
    presc = arrays.prescribed_indices
    K_bcoo, R, _, _ = assemble_global(
        fe_problem, arrays, params, U, U_prev, step_time,
        xi_prev_by_block=xi_prev,
    )
    K, _ = _embedded_bc_enforce(K_bcoo, presc)
    assembled = AssembledOperator(
        K, arrays.embedded_sparsity, arrays.block_sparsity,
    )
    K_elem, R_elem, _, _ = assemble_element_tangent(
        fe_problem, arrays, params, U, U_prev, step_time,
        xi_prev_by_block=xi_prev,
    )
    np.testing.assert_array_equal(np.asarray(R_elem), np.asarray(R))
    element = ElementOperator(
        K_elem, arrays.r_scatter_eq_by_block, fe_problem.field_idx_per_block,
        fe_problem.dof_map.block_offsets, presc,
        fe_problem.dof_map.num_total_dofs,
    )
    return assembled, element, K_bcoo


def _assert_close(actual, reference, rel=1e-12):
    reference = np.asarray(reference)
    scale = float(np.abs(reference).max())
    assert scale > 0.0
    np.testing.assert_allclose(
        np.asarray(actual), reference, rtol=0, atol=rel * scale,
    )


class TestElementOperatorMatchesAssembled(unittest.TestCase):

    def _check(self, fe_problem, params, U, U_prev, xi_prev, step_time):
        assembled, element, K_bcoo = _operators(
            fe_problem, params, U, U_prev, xi_prev, step_time,
        )
        n = fe_problem.dof_map.num_total_dofs
        rng = np.random.default_rng(3)
        x = jnp.asarray(rng.standard_normal(n))
        _assert_close(element.matvec(x), assembled.matvec(x))
        _assert_close(element.raw_matvec(x), K_bcoo @ x)
        _assert_close(element.diagonal(), assembled.diagonal())
        presc = fe_problem.kernel_arrays.prescribed_indices
        presc_vals = jnp.asarray(rng.standard_normal(presc.shape[0]))
        r_assembled = _embedded_residual(
            x, lambda v: K_bcoo @ v, jnp.asarray(U), presc, presc_vals,
            assembled.diagonal()[presc],
        )
        r_element = _embedded_residual(
            x, element.raw_matvec, jnp.asarray(U), presc, presc_vals,
            element.diagonal()[presc],
        )
        _assert_close(r_element, r_assembled)
        if fe_problem.kernel_arrays.block_sparsity is None:
            return
        offs = element.field_offsets
        self.assertEqual(offs, assembled.field_offsets)
        for i in range(element.num_fields):
            _assert_close(element.block_diagonal(i), assembled.block_diagonal(i))
            for j in range(element.num_fields):
                x_j = x[offs[j]:offs[j + 1]]
                for transpose in (False, True):
                    with self.subTest(i=i, j=j, transpose=transpose):
                        _assert_close(
                            element.block_matvec(i, j, x_j, transpose=transpose),
                            assembled.block_matvec(i, j, x_j, transpose=transpose),
                        )

    def test_mixed_up_plastic_state(self) -> None:
        fe_problem, params, U, U_prev, _xi, xi_prev, step_time = (
            drive_mixed_problem()
        )
        self._check(fe_problem, params, U, U_prev, xi_prev, step_time)

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
        state = FEState.from_problem(fe_problem)
        xi_prev = {"right": state.xi_at(0, "right")}
        self._check(
            fe_problem, params_by_block_from_models(fe_problem), U,
            np.zeros(n), xi_prev, StepTime(1.0, 0.0),
        )


class TestNewtonStepWithElementOperator(unittest.TestCase):

    def test_matches_assembled_to_solver_tolerance(self) -> None:
        fe_problem, params, _U, U_prev, _xi, xi_prev, step_time = (
            drive_mixed_problem()
        )
        results = {}
        for operator in ("assembled", "element"):
            settings = {**_JAX_BLOCK_CHEBYSHEV_SETTINGS, "operator": operator}
            U_new, xi_new = fe_newton_solve(
                fe_problem, params, U_prev=U_prev, t=float(step_time.t),
                t_prev=float(step_time.t_prev), xi_prev_by_block=xi_prev,
                linear_solver_settings=settings,
            )
            results[operator] = (np.asarray(U_new), np.asarray(xi_new["all"]))
        _assert_close(results["element"][0], results["assembled"][0], 1e-8)
        _assert_close(results["element"][1], results["assembled"][1], 1e-8)
        # The comparison means nothing if the step did not move U.
        self.assertGreater(
            float(np.abs(results["assembled"][0] - np.asarray(U_prev)).max()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
