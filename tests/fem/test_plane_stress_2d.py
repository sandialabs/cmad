"""Plane stress uniaxial tension check in 2D.

A plane stress square pulled in x with a free lateral edge has the
homogeneous solution sigma_xx = E * eps_xx, sigma_yy = 0, sigma_zz = 0.
The local solve enforces sigma_zz = 0 via the out of plane stretch (a
local COUPLED unknown, never in the momentum balance), and Q1 captures
the constant strain field exactly, so this checks the 2D plane stress
path against the analytic stress with no residual or model change.

The 2D cauchy is the in-plane block (xx, xy, yy) -- sigma_xx and
sigma_yy at indices 0 and 2. sigma_zz (zero here, enforced by the local
solve) is out of plane and not in the in-plane output.
"""
import unittest

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEState, build_fe_problem
from cmad.fem.finite_element import Q1_QUAD
from cmad.fem.mesh import StructuredQuadMesh
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.fem.postprocess import evaluate_cauchy_at_ips
from cmad.global_residuals.mechanics import Mechanics
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.typing import JaxArray
from tests.fem._mms_helpers import make_elastic_parameters

_KAPPA = 100.0
_MU = 50.0
_EPS_XX = 1e-3


def _u_x_at_t(
        coords: NDArray[np.floating] | JaxArray, t: float | JaxArray,
) -> JaxArray:
    return jnp.full((np.asarray(coords).shape[0], 1), t)


class TestPlaneStressUniaxial2D(unittest.TestCase):

    def test_uniaxial_matches_E_times_strain(self) -> None:
        mesh = StructuredQuadMesh(lengths=(1.0, 1.0), divisions=(2, 2))
        layout = GlobalFieldLayout(name="u", finite_element=Q1_QUAD)
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
            mesh, [layout], dbcs, components_by_field={"u": 2},
        )
        gr = Mechanics(ndims=2)
        # PLANE_STRESS is not wired for CLOSED_FORM (the out of plane
        # stretch needs the local solve), so run COUPLED.
        model = Elastic(
            make_elastic_parameters(_KAPPA, _MU),
            def_type=DefType.PLANE_STRESS,
        )
        fe_problem = build_fe_problem(
            mesh=mesh, dof_map=dof_map, gr=gr,
            models_by_block={"all": model},
            modes_by_block={"all": GlobalResidualMode.COUPLED},
        )

        params = params_by_block_from_models(fe_problem)
        state = FEState.from_problem(fe_problem)
        U, xi = fe_newton_solve(
            fe_problem, params, U_prev=state.U_at(0), t=_EPS_XX,
            xi_prev_by_block={"all": state.xi_at(0, "all")},
        )
        state.append(U, xi, _EPS_XX)

        cauchy = np.asarray(
            evaluate_cauchy_at_ips(fe_problem, state, 1, "all"),
        )
        # Young's modulus from (kappa, mu).
        E = 9.0 * _KAPPA * _MU / (3.0 * _KAPPA + _MU)

        np.testing.assert_allclose(
            cauchy[..., 0], E * _EPS_XX, rtol=1e-6,
        )
        self.assertLess(float(np.max(np.abs(cauchy[..., 2]))), 1e-6)


if __name__ == "__main__":
    unittest.main()
