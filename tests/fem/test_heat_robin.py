"""Surface conditions on a heated bar, each against its exact linear
profile: convection and radiation (Robin) and a prescribed flux, plus the
Robin tangent against central differences of the residual and the two
tangent representations against each other."""
import unittest
from collections.abc import Sequence
from typing import Any

import numpy as np
from jax import numpy as jnp
from scipy.optimize import brentq

from cmad.fem.assembly import (
    assemble_global,
    assemble_global_residual,
    params_by_block_from_models,
)
from cmad.fem.bcs import DirichletBC, NeumannBC, RobinBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map, dof_physical_coords
from cmad.fem.driver import fe_quasistatic_drive
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.global_residuals.heat_transfer import HeatTransfer
from cmad.models.conduction import Conduction
from cmad.models.global_fields import StepTime
from tests.fem._mms_helpers import make_conduction_parameters

_K = 10.0
_T0 = 1000.0
_T_INF = 300.0
_H = 25.0
_EMISSIVITY = 0.8
_SIGMA_B = 5.67e-8
_Q_IN = 500.0

_ELEMENT_SOLVER: dict[str, Any] = {
    "type": "cg", "operator": "element", "preconditioner": {"type": "jacobi"},
    "max iters": 500, "rel tol": 1.0e-12,
}


def _bar(
        neumann_bcs: Sequence[NeumannBC] = (),
        robin_bcs: Sequence[RobinBC] = (),
) -> FEProblem:
    """A unit bar of 8 hexes along x, ``T = T0`` at ``x = 0``, the other
    faces insulated unless a condition is given on ``xmax_sides``."""
    mesh = StructuredHexMesh(lengths=(1.0, 1.0, 1.0), divisions=(8, 1, 1))
    layout = GlobalFieldLayout(name="T", finite_element=Q1_HEX)
    dof_map = build_dof_map(
        mesh, [layout],
        [DirichletBC(["xmin_sides"], "T", dofs=(0,), values=[_T0])],
        components_by_field={"T": 1},
    )
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=HeatTransfer(ndims=3),
        models_by_block={"all": Conduction(make_conduction_parameters(_K))},
        neumann_bcs=list(neumann_bcs),
        robin_bcs=list(robin_bcs),
    )


def _solve(
        fe_problem: FEProblem, **solver_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(x, T, newton_iterations)`` of one step at ``t = 1``."""
    state, _, status = fe_quasistatic_drive(
        fe_problem, [0.0, 1.0], **solver_kwargs,
    )
    assert status.converged, status.failure_message()
    coords, eq = dof_physical_coords(
        fe_problem.mesh, fe_problem.dof_map, "T",
    )
    T = np.asarray(state.U_at(-1))[eq[:, 0]]
    return coords[:, 0], T, status.iters_per_step[0]


def _convection() -> RobinBC:
    return RobinBC(
        ["xmax_sides"], "T", lambda T, coords, t: _H * (T - _T_INF),
    )


def _radiation() -> RobinBC:
    return RobinBC(
        ["xmax_sides"], "T",
        lambda T, coords, t: _EMISSIVITY * _SIGMA_B * (T ** 4 - _T_INF ** 4),
    )


def _tangent_and_fd(fe_problem: FEProblem, U: np.ndarray) -> tuple[float, float]:
    """``(max |K v - fd|, max |K v|)`` for a random direction ``v`` at ``U``,
    the difference being central, of the assembled ``K`` against the
    residual only assembly."""
    fe_arrays = fe_problem.kernel_arrays
    params = params_by_block_from_models(fe_problem)
    step_time = StepTime(1.0, 0.0)
    U_jax = jnp.asarray(U)
    K, _, _, _ = assemble_global(
        fe_problem, fe_arrays, params, U_jax, U_jax, step_time,
    )
    v = jnp.asarray(np.random.default_rng(0).standard_normal(U.shape[0]))
    eps = 1.0e-3

    def residual(U_at):
        return assemble_global_residual(
            fe_problem, fe_arrays, params, U_at, U_jax, step_time,
        )

    fd = (residual(U_jax + eps * v) - residual(U_jax - eps * v)) / (2 * eps)
    Kv = K @ v
    return (
        float(np.max(np.abs(np.asarray(Kv - fd)))),
        float(np.max(np.abs(np.asarray(Kv)))),
    )


class TestHeatSurfaceConditions(unittest.TestCase):
    def test_convection_bar(self) -> None:
        # k (T0 - T_L) / L = h (T_L - T_inf) at the cooled end.
        T_end = (_K * _T0 + _H * _T_INF) / (_K + _H)
        x, T, iterations = _solve(_bar(robin_bcs=[_convection()]))
        np.testing.assert_allclose(T, _T0 + (T_end - _T0) * x, rtol=1e-10)
        self.assertEqual(iterations, 1)

    def test_convection_bar_element_operator(self) -> None:
        T_end = (_K * _T0 + _H * _T_INF) / (_K + _H)
        x, T, _ = _solve(
            _bar(robin_bcs=[_convection()]),
            linear_solver_settings=_ELEMENT_SOLVER,
        )
        np.testing.assert_allclose(T, _T0 + (T_end - _T0) * x, rtol=1e-8)

    def test_radiation_bar(self) -> None:
        # k (T0 - T_L) / L = eps sigma_B (T_L^4 - T_inf^4) at the end.
        T_end = brentq(
            lambda TL: _K * (_T0 - TL)
            - _EMISSIVITY * _SIGMA_B * (TL ** 4 - _T_INF ** 4),
            _T_INF, _T0,
        )
        x, T, iterations = _solve(_bar(robin_bcs=[_radiation()]))
        np.testing.assert_allclose(T, _T0 + (T_end - _T0) * x, rtol=1e-8)
        self.assertLessEqual(iterations, 10, f"{iterations} iterations")

    def test_flux_bar(self) -> None:
        # Heat entering at x = 1 leaves at x = 0: T = T0 + q x / k.
        x, T, _ = _solve(
            _bar(neumann_bcs=[NeumannBC(["xmax_sides"], "T", [_Q_IN])]),
        )
        np.testing.assert_allclose(T, _T0 + _Q_IN * x / _K, rtol=1e-10)

    def test_side_tangent_is_placed_where_the_side_residual_is(self) -> None:
        # The per element side tangent is exact by AD; this checks that
        # it lands at the right pattern positions, in the right order,
        # by differencing the assembled residual.
        for make_bc in (_convection, _radiation):
            fe_problem = _bar(robin_bcs=[make_bc()])
            coords, eq = dof_physical_coords(
                fe_problem.mesh, fe_problem.dof_map, "T",
            )
            U = np.zeros(fe_problem.dof_map.num_total_dofs)
            U[eq[:, 0]] = 800.0 + 100.0 * coords[:, 0]
            difference, scale = _tangent_and_fd(fe_problem, U)
            self.assertLess(
                difference / scale, 1e-6,
                f"{make_bc.__name__}: {difference} against {scale}",
            )


if __name__ == "__main__":
    unittest.main()
