"""Uniaxial tension check for the mixed (u-p) plastic SmallDispEquilibrium.

A cube in uniaxial tension reproduces the J2 + Voce analytic axial stress, with
vanishing lateral stress and pressure p = -sigma_axial / 3, for both small
strain plastic models.
"""
import unittest
from collections.abc import Callable, Mapping

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEProblem, FEState, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.fem.postprocess import evaluate_cauchy_at_ips
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.deformation_types import DefType
from cmad.models.model import Model
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.typing import JaxArray
from tests.support.test_problems import J2AnalyticalProblem

MAX_ALPHA = 0.05
NUM_DRIVE_STEPS = 5


def _axial_displacement(
        coords: NDArray[np.floating] | JaxArray, t: float | JaxArray,
) -> JaxArray:
    return jnp.full((np.asarray(coords).shape[0], 1), t)


def _uniaxial_dbcs() -> list[DirichletBC]:
    return [
        DirichletBC(
            sideset_names=["xmin_sides"], field_name="u", dofs=(0,),
            values=None,
        ),
        DirichletBC(
            sideset_names=["ymin_sides"], field_name="u", dofs=(1,),
            values=None,
        ),
        DirichletBC(
            sideset_names=["zmin_sides"], field_name="u", dofs=(2,),
            values=None,
        ),
        DirichletBC(
            sideset_names=["xmax_sides"], field_name="u", dofs=(0,),
            values=_axial_displacement,
        ),
    ]


def _build_mixed_fe(model: Model) -> FEProblem:
    mesh = StructuredHexMesh(lengths=(1.0, 1.0, 1.0), divisions=(2, 2, 2))
    layouts = [
        GlobalFieldLayout(name="u", finite_element=Q1_HEX),
        GlobalFieldLayout(name="p", finite_element=Q1_HEX),
    ]
    dof_map = build_dof_map(
        mesh, layouts, _uniaxial_dbcs(),
        components_by_field={"u": 3, "p": 1},
    )
    gr = SmallDispEquilibrium(ndims=3, mixed=True)
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.COUPLED},
    )


_BLOCK_SOLVER_SETTINGS = {
    "type": "gmres",
    "preconditioner": {
        "type": "block", "inner": "amg",
        "diagonal_block": "schur", "coupling": "lower",
    },
    "rtol": 1.0e-10, "max iters": 20, "restart": 120,
}


class TestMixedUpPlastic(unittest.TestCase):

    def _run(
            self, model_cls: Callable[..., Model],
            linear_solver_settings: dict | None = None,
    ) -> None:
        problem = J2AnalyticalProblem()
        stress_mask = np.zeros((3, 3))
        stress_mask[0, 0] = 1.0
        stress, strain, _ = problem.analytical_solution(
            stress_mask, MAX_ALPHA, num_steps=2,
        )
        axial_strain = float(strain[0, 0, -1])
        sigma_axial = float(stress[0, 0, -1])

        model = model_cls(problem.J2_parameters, def_type=DefType.FULL_3D)
        fe_problem = _build_mixed_fe(model)
        params = params_by_block_from_models(fe_problem)

        state = FEState.from_problem(fe_problem)
        U_solved: NDArray[np.floating] | JaxArray = state.U_at(0)
        xi_prev: Mapping[str, NDArray[np.floating] | JaxArray] = {
            "all": state.xi_at(0, "all"),
        }
        for step in range(1, NUM_DRIVE_STEPS + 1):
            t = axial_strain * step / NUM_DRIVE_STEPS
            U_solved, xi_solved = fe_newton_solve(
                fe_problem, params, U_prev=U_solved, t=t,
                xi_prev_by_block=xi_prev,
                linear_solver_settings=linear_solver_settings,
            )
            state.append(U_solved, xi_solved, t)
            xi_prev = xi_solved

        cauchy = evaluate_cauchy_at_ips(
            fe_problem, state, NUM_DRIVE_STEPS, "all",
        )
        np.testing.assert_allclose(cauchy[..., 0], sigma_axial, rtol=1e-5)
        self.assertLess(
            float(np.max(np.abs(cauchy[..., 3]))), 1e-4 * sigma_axial,
        )
        self.assertLess(
            float(np.max(np.abs(cauchy[..., 5]))), 1e-4 * sigma_axial,
        )
        p = np.asarray(U_solved)[fe_problem.dof_map.block_offsets[1]:]
        np.testing.assert_allclose(p, -sigma_axial / 3.0, rtol=1e-5)

    def test_small_elastic_plastic(self) -> None:
        self._run(SmallElasticPlastic)

    def test_small_rate_elastic_plastic(self) -> None:
        self._run(SmallRateElasticPlastic)

    def test_small_elastic_plastic_block_solver(self) -> None:
        self._run(SmallElasticPlastic, _BLOCK_SOLVER_SETTINGS)

    def test_small_rate_elastic_plastic_block_solver(self) -> None:
        self._run(SmallRateElasticPlastic, _BLOCK_SOLVER_SETTINGS)


if __name__ == "__main__":
    unittest.main()
