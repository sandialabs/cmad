"""Smoke test: 2D finite plane stress plasticity.

Drives a small quad mesh past yield with
:class:`~cmad.models.be_bar_elastic_plastic.BeBarElasticPlastic` at
``def_type=PLANE_STRESS``. Beyond the material point coverage in
``tests/models/test_be_bar_plane_stress.py``, this is the path where the
out of plane stretch reaches the global residual: it is a local unknown,
so ``deformation_gradient`` carries it into the ``cof(F)`` of the finite
Piola-Kirchhoff map, and the traced local solve starts from the model's
elastic predictor rather than from the previous state.

Uniform biaxial stretching would leave a homogeneous state; the mesh is
pulled in x with the transverse direction free, so the elements thin
through the plane by different amounts as they yield.
"""
import unittest
from collections.abc import Mapping

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
from cmad.models.be_bar_elastic_plastic import BeBarElasticPlastic
from cmad.models.deformation_types import DefType
from cmad.typing import JaxArray
from tests.support.test_problems import J2AnalyticalProblem

# Yield is near 0.1% nominal strain (Y/E = 200/200e3); ramp to 5% over 10
# finite steps so the elements are well into plasticity by the last one.
_NUM_STEPS = 10
_MAX_AXIAL_DISP = 0.05


def _u_x_at_t(
        coords: NDArray[np.floating] | JaxArray, t: float | JaxArray,
) -> JaxArray:
    return jnp.full((np.asarray(coords).shape[0], 1), t)


def _build_fe(model: BeBarElasticPlastic) -> FEProblem:
    mesh = StructuredQuadMesh(lengths=(1.0, 1.0), divisions=(2, 2))
    dbcs = [
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
    dof_map = build_dof_map(
        mesh, [GlobalFieldLayout(name="u", finite_element=Q1_QUAD)], dbcs,
        components_by_field={"u": 2},
    )
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=Mechanics(ndims=2),
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.COUPLED},
    )


class TestPlaneStressFinitePlastic2D(unittest.TestCase):

    def test_converges_and_thins(self) -> None:
        model = BeBarElasticPlastic(
            J2AnalyticalProblem().J2_parameters,
            def_type=DefType.PLANE_STRESS,
        )
        fe_problem = _build_fe(model)
        params = params_by_block_from_models(fe_problem)

        state = FEState.from_problem(fe_problem)
        U_solved: NDArray[np.floating] | JaxArray = state.U_at(0)
        xi_prev: Mapping[str, NDArray[np.floating] | JaxArray] = {
            "all": state.xi_at(0, "all"),
        }
        for step in range(1, _NUM_STEPS + 1):
            t = _MAX_AXIAL_DISP * step / _NUM_STEPS
            U_solved, xi_solved = fe_newton_solve(
                fe_problem, params, U_prev=U_solved, t=t,
                xi_prev_by_block=xi_prev,
            )
            state.append(U_solved, xi_solved, t)
            xi_prev = xi_solved

        self.assertTrue(np.all(np.isfinite(np.asarray(U_solved))))

        # xi is [zeta (6), Ie (1), alpha (1), out of plane stretch (1)].
        alpha_idx = int(model._num_eqs[0]) + int(model._num_eqs[1])
        xi_final = state.xi_at(_NUM_STEPS, "all")
        alpha = xi_final[..., alpha_idx]
        self.assertGreater(float(np.max(alpha)), 0.0)

        # Pulled in x, the sheet thins: the solved stretch drops below 1.
        stretch = np.asarray(xi_final[..., alpha_idx + 1])
        self.assertTrue(np.all(stretch > 0.0))
        self.assertLess(float(np.max(stretch)), 1.0)

        # The plane stress condition holds at every integration point.
        cauchy = np.asarray(
            evaluate_cauchy_at_ips(fe_problem, state, _NUM_STEPS, "all"),
        )
        self.assertTrue(np.all(np.isfinite(cauchy)))
        scale = float(np.max(np.abs(cauchy)))
        self.assertGreater(scale, 0.0)
        self.assertLess(float(np.max(np.abs(cauchy[..., 2, 2]))) / scale, 1e-8)


if __name__ == "__main__":
    unittest.main()
