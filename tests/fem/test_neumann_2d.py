"""Edge traction patch test for 2D Neumann BCs.

A constant traction ``t̄ = (p, 0)`` on the +x edge of a unit square, with
symmetry BCs on the -x edge (``u_x = 0``) and -y edge (``u_y = 0``).
Equilibrium gives a uniform uniaxial stress ``sigma_xx = p`` (the +y edge
is free, so ``sigma_yy = 0``), which is the direct check of the edge
length factor in the Neumann side integral: a wrong ``ds`` changes the
total applied force, hence ``sigma_xx``. Run on a quad mesh and a tri
mesh (``quad_to_tri_split``), so both edge lifts are exercised. Q1 / P1
capture the constant stress field exactly, so the recovered ``sigma_xx``
matches ``p`` to solver tolerance.
"""
import unittest

import numpy as np
from numpy.typing import NDArray

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.bcs import DirichletBC, NeumannBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import FEProblem, FEState, build_fe_problem
from cmad.fem.finite_element import P1_TRI, Q1_QUAD
from cmad.fem.mesh import Mesh, StructuredQuadMesh, quad_to_tri_split
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.fem.postprocess import evaluate_cauchy_at_ips
from cmad.global_residuals.mechanics import Mechanics
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from tests.fem._mms_helpers import make_elastic_parameters

_KAPPA = 100.0
_MU = 50.0
_P = 10.0


def _build_fe_problem(mesh: Mesh) -> FEProblem:
    if mesh.element_family == ElementFamily.QUAD_LINEAR:
        fe = Q1_QUAD
    elif mesh.element_family == ElementFamily.TRI_LINEAR:
        fe = P1_TRI
    else:
        raise ValueError(f"unsupported element family {mesh.element_family}")
    layout = GlobalFieldLayout(name="u", finite_element=fe)
    dbcs = [
        DirichletBC(
            sideset_names=["xmin_sides"], field_name="u", dofs=(0,),
            values=None,
        ),
        DirichletBC(
            sideset_names=["ymin_sides"], field_name="u", dofs=(1,),
            values=None,
        ),
    ]
    nbc = NeumannBC(
        sideset_names=["xmax_sides"], field_name="u", values=[_P, 0.0],
    )
    dof_map = build_dof_map(
        mesh, [layout], dbcs, components_by_field={"u": 2},
    )
    gr = Mechanics(ndims=2)
    # COUPLED so the cauchy state variable is solved and readable by
    # evaluate_cauchy_at_ips (the closed form path leaves xi unsolved).
    model = Elastic(
        make_elastic_parameters(_KAPPA, _MU), def_type=DefType.PLANE_STRAIN,
    )
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.COUPLED},
        neumann_bcs=(nbc,),
    )


def _solve_cauchy(mesh: Mesh) -> NDArray[np.floating]:
    fe_problem = _build_fe_problem(mesh)
    params = params_by_block_from_models(fe_problem)
    state = FEState.from_problem(fe_problem)
    U, xi = fe_newton_solve(
        fe_problem, params, U_prev=state.U_at(0), t=1.0,
        xi_prev_by_block={"all": state.xi_at(0, "all")},
    )
    state.append(U, xi, 1.0)
    return np.asarray(evaluate_cauchy_at_ips(fe_problem, state, 1, "all"))


class TestNeumannEdgeTraction2D(unittest.TestCase):

    def _check_uniaxial(self, mesh: Mesh) -> None:
        cauchy = _solve_cauchy(mesh)
        # 2D cauchy is the in-plane block (xx, xy, yy).
        np.testing.assert_allclose(cauchy[..., 0], _P, rtol=1e-6)
        self.assertLess(float(np.max(np.abs(cauchy[..., 1]))), 1e-6 * _P)
        self.assertLess(float(np.max(np.abs(cauchy[..., 2]))), 1e-6 * _P)

    def test_quad(self) -> None:
        mesh = StructuredQuadMesh(lengths=(1.0, 1.0), divisions=(2, 2))
        self._check_uniaxial(mesh)

    def test_tri(self) -> None:
        quad_mesh = StructuredQuadMesh(lengths=(1.0, 1.0), divisions=(2, 2))
        self._check_uniaxial(quad_to_tri_split(quad_mesh))


if __name__ == "__main__":
    unittest.main()
