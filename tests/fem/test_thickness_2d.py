"""Out of plane thickness on a 2D mesh.

A 2D mesh integrates over area, so without a thickness everything with
units of force comes out per unit of it. ``discretization: thickness``
scales the volume element, and the same factor has to reach the Neumann
side integral, where a traction acts over an edge length times the
thickness.

Two checks cover that. Under prescribed displacement the reaction is
proportional to the thickness while the displacement field is untouched,
since the factor is uniform on the residual. Under a prescribed traction
the displacement field is untouched instead, because the applied force
and the internal one pick up the same factor. The second fails if only
the volume measure is scaled.
"""
import unittest

import numpy as np
from numpy.typing import NDArray

from cmad.fem.assembly import (
    assemble_global_residual,
    params_by_block_from_models,
)
from cmad.fem.bcs import DirichletBC, NeumannBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEProblem, FEState, build_fe_problem
from cmad.fem.finite_element import Q1_QUAD
from cmad.fem.mesh import StructuredQuadMesh
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.global_residuals.mechanics import Mechanics
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.global_fields import StepTime
from tests.fem._mms_helpers import make_elastic_parameters

_KAPPA = 100.0
_MU = 50.0
_TRACTION = 10.0
_STRETCH = 0.01


def _build(thickness: float | None, *, traction: bool) -> FEProblem:
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
    ]
    nbcs: tuple[NeumannBC, ...] = ()
    if traction:
        nbcs = (
            NeumannBC(
                sideset_names=["xmax_sides"], field_name="u",
                values=[_TRACTION, 0.0],
            ),
        )
    else:
        dbcs.append(
            DirichletBC(
                sideset_names=["xmax_sides"], field_name="u", dofs=(0,),
                values=lambda coords, t: np.full(
                    (np.asarray(coords).shape[0], 1), _STRETCH * t,
                ),
            ),
        )
    dof_map = build_dof_map(
        mesh, [GlobalFieldLayout(name="u", finite_element=Q1_QUAD)], dbcs,
        components_by_field={"u": 2},
    )
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=Mechanics(ndims=2),
        models_by_block={"all": Elastic(
            make_elastic_parameters(_KAPPA, _MU),
            def_type=DefType.PLANE_STRAIN,
        )},
        modes_by_block={"all": GlobalResidualMode.COUPLED},
        neumann_bcs=nbcs,
        thickness=thickness,
    )


def _solve(
        thickness: float | None, *, traction: bool,
) -> tuple[NDArray[np.floating], float]:
    """Return the displacement field and the reaction on the fixed edge."""
    fe_problem = _build(thickness, traction=traction)
    params = params_by_block_from_models(fe_problem)
    state = FEState.from_problem(fe_problem)
    U, _xi = fe_newton_solve(
        fe_problem, params, U_prev=state.U_at(0), t=1.0,
        xi_prev_by_block={"all": state.xi_at(0, "all")},
    )
    # At the converged solution the residual holds the internal force on
    # the constrained equations, which is the reaction there. This is the
    # same quantity fe_load_match reports.
    R = np.asarray(assemble_global_residual(
        fe_problem, fe_problem.kernel_arrays, params, U, state.U_at(0),
        StepTime(t=1.0, t_prev=0.0),
        {"all": state.xi_at(0, "all")},
    ))
    eqs = fe_problem.dof_map.dirichlet_eqs_for_component("xmin_sides", "u", 0)
    return np.asarray(U), float(np.sum(R[eqs]))


class TestThickness2D(unittest.TestCase):

    def test_prescribed_displacement_scales_the_reaction(self) -> None:
        U_thin, reaction_thin = _solve(1.0, traction=False)
        U_thick, reaction_thick = _solve(2.5, traction=False)

        np.testing.assert_allclose(U_thick, U_thin, atol=1e-12)
        self.assertGreater(abs(reaction_thin), 1e-8)
        np.testing.assert_allclose(
            reaction_thick, 2.5 * reaction_thin, rtol=1e-9,
        )

    def test_prescribed_traction_is_thickness_independent(self) -> None:
        U_thin, _reaction = _solve(1.0, traction=True)
        U_thick, _reaction = _solve(2.5, traction=True)

        self.assertGreater(float(np.max(np.abs(U_thin))), 1e-8)
        np.testing.assert_allclose(U_thick, U_thin, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
