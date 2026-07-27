"""Smoke test: 2D plane strain plasticity solves in both formulations.

Drives a small structured quad mesh past yield under an in-plane
displacement ramp and checks that the global and per-IP (COUPLED) Newton
solves converge for both the displacement and the mixed (u-p) plane
strain Mechanics, with each small strain plastic model. This is the one
surface the elastic MMS does not cover: the elastic MMS exercises 2D
assembly but only CLOSED_FORM (no local solve), while the COUPLED local
solve is otherwise tested only in 3D. The bar here is convergence plus
stress sanity (finite displacement, plasticity entered, finite stress),
not an analytic stress.
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
from cmad.fem.finite_element import Q1_QUAD
from cmad.fem.mesh import StructuredQuadMesh
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.fem.postprocess import evaluate_cauchy_at_ips
from cmad.global_residuals.mechanics import Mechanics
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.deformation_types import DefType
from cmad.models.model import Model
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.small_rate_elastic_plastic import SmallRateElasticPlastic
from cmad.typing import JaxArray
from tests.support.test_problems import J2AnalyticalProblem

# J2 yield strain is Y/E = 200/200e3 = 1e-3; ramp to ~5x that over 4
# steps so the IPs are well into the plastic regime by the final step.
_NUM_STEPS = 4
_MAX_AXIAL_STRAIN = 5e-3


def _u_x_at_t(
        coords: NDArray[np.floating] | JaxArray, t: float | JaxArray,
) -> JaxArray:
    return jnp.full((np.asarray(coords).shape[0], 1), t)


def _plane_strain_dbcs() -> list[DirichletBC]:
    """Symmetry on -x / -y edges, a t-ramped u_x on +x, +y free."""
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
            sideset_names=["xmax_sides"], field_name="u", dofs=(0,),
            values=_u_x_at_t,
        ),
    ]


def _build_fe(model: Model, mixed: bool) -> FEProblem:
    mesh = StructuredQuadMesh(lengths=(1.0, 1.0), divisions=(2, 2))
    if mixed:
        layouts = [
            GlobalFieldLayout(name="u", finite_element=Q1_QUAD),
            GlobalFieldLayout(name="p", finite_element=Q1_QUAD),
        ]
        components = {"u": 2, "p": 1}
        gr = Mechanics(ndims=2, mixed=True)
    else:
        layouts = [GlobalFieldLayout(name="u", finite_element=Q1_QUAD)]
        components = {"u": 2}
        gr = Mechanics(ndims=2)
    dof_map = build_dof_map(
        mesh, layouts, _plane_strain_dbcs(), components_by_field=components,
    )
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.COUPLED},
    )


class TestPlaneStrainPlastic2D(unittest.TestCase):

    def _run(self, model_cls: Callable[..., Model], mixed: bool) -> None:
        model = model_cls(
            J2AnalyticalProblem().J2_parameters, def_type=DefType.PLANE_STRAIN,
        )
        fe_problem = _build_fe(model, mixed)
        params = params_by_block_from_models(fe_problem)

        state = FEState.from_problem(fe_problem)
        U_solved: NDArray[np.floating] | JaxArray = state.U_at(0)
        xi_prev: Mapping[str, NDArray[np.floating] | JaxArray] = {
            "all": state.xi_at(0, "all"),
        }
        for step in range(1, _NUM_STEPS + 1):
            t = _MAX_AXIAL_STRAIN * step / _NUM_STEPS
            U_solved, xi_solved = fe_newton_solve(
                fe_problem, params, U_prev=U_solved, t=t,
                xi_prev_by_block=xi_prev,
            )
            state.append(U_solved, xi_solved, t)
            xi_prev = xi_solved

        # Global Newton produced a finite displacement field.
        self.assertTrue(np.all(np.isfinite(np.asarray(U_solved))))

        # The COUPLED local solve drove at least one IP into plasticity.
        # alpha (scalar hardening) is xi var 1, just past the 6-component
        # var 0 (plastic strain / unrotated cauchy), so its flat offset is
        # the size of var 0.
        alpha_idx = int(model._num_eqs[0])
        alpha = state.xi_at(_NUM_STEPS, "all")[..., alpha_idx]
        self.assertGreater(float(np.max(alpha)), 0.0)

        # Recovered stress is finite and nonzero.
        cauchy = np.asarray(
            evaluate_cauchy_at_ips(fe_problem, state, _NUM_STEPS, "all"),
        )
        self.assertTrue(np.all(np.isfinite(cauchy)))
        self.assertGreater(float(np.max(np.abs(cauchy))), 0.0)

    def test_displacement_small(self) -> None:
        self._run(SmallElasticPlastic, mixed=False)

    def test_displacement_small_rate(self) -> None:
        self._run(SmallRateElasticPlastic, mixed=False)

    def test_mixed_small(self) -> None:
        self._run(SmallElasticPlastic, mixed=True)

    def test_mixed_small_rate(self) -> None:
        self._run(SmallRateElasticPlastic, mixed=True)


if __name__ == "__main__":
    unittest.main()
