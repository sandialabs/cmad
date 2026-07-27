"""Smoke test: 2D finite plane strain plasticity, mixed (u-p) form.

Drives a small quad mesh past yield with the finite deformation
:class:`~cmad.models.be_bar_elastic_plastic.BeBarElasticPlastic` at
``def_type=PLANE_STRAIN``, mixed and COUPLED. Checks convergence and
stress sanity (finite displacement, plasticity entered, finite stress);
finite plane strain J2 has no simple closed form to check against, since
F_33 = 1 leaves a nonzero sigma_zz.
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

# Yield is near 0.1% nominal strain (Y/E = 200/200e3); ramp the +x face
# to 5% displacement over 10 finite steps so the IPs are well into
# plasticity by the final step.
_NUM_STEPS = 10
_MAX_AXIAL_DISP = 0.05


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


def _build_mixed_fe(model: BeBarElasticPlastic) -> FEProblem:
    mesh = StructuredQuadMesh(lengths=(1.0, 1.0), divisions=(2, 2))
    layouts = [
        GlobalFieldLayout(name="u", finite_element=Q1_QUAD),
        GlobalFieldLayout(name="p", finite_element=Q1_QUAD),
    ]
    dof_map = build_dof_map(
        mesh, layouts, _plane_strain_dbcs(),
        components_by_field={"u": 2, "p": 1},
    )
    gr = Mechanics(ndims=2, mixed=True)
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.COUPLED},
    )


class TestPlaneStrainFinitePlastic2D(unittest.TestCase):

    def test_mixed_converges_and_yields(self) -> None:
        model = BeBarElasticPlastic(
            J2AnalyticalProblem().J2_parameters,
            def_type=DefType.PLANE_STRAIN,
        )
        fe_problem = _build_mixed_fe(model)
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

        # Global Newton produced a finite displacement-pressure field.
        self.assertTrue(np.all(np.isfinite(np.asarray(U_solved))))

        # The COUPLED local solve drove at least one IP into plasticity.
        # alpha (scalar hardening) is xi var 2, after the 6-component zeta
        # (var 0) and the scalar Ie (var 1), so its flat offset is their
        # combined size.
        alpha_idx = int(model._num_eqs[0]) + int(model._num_eqs[1])
        alpha = state.xi_at(_NUM_STEPS, "all")[..., alpha_idx]
        self.assertGreater(float(np.max(alpha)), 0.0)

        # Recovered stress is finite and nonzero.
        cauchy = np.asarray(
            evaluate_cauchy_at_ips(fe_problem, state, _NUM_STEPS, "all"),
        )
        self.assertTrue(np.all(np.isfinite(cauchy)))
        self.assertGreater(float(np.max(np.abs(cauchy))), 0.0)


if __name__ == "__main__":
    unittest.main()
