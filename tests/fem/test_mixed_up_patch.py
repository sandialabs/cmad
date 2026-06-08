"""Constant-strain patch test for the mixed (u-p) SmallDispEquilibrium.

A linear displacement ``u = A·x`` with zero body force is an exact
equilibrium solution: the strain (and stress) is uniform, so
``div(sigma) = 0``. A consistent element reproduces it to roundoff. For
the mixed formulation the pressure is the constant ``p = -kappa·tr(eps)``,
so ``grad(p) = 0`` makes the stabilization term zero and the constraint
pins ``p = -hydro``. The patch test exercises the full two-block assembly
and the indefinite two-field solve end to end.
"""
import unittest

import numpy as np

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEState, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from tests.fem._mms_helpers import l2_h1_errors, make_elastic_parameters

_KAPPA = 100.0
_MU = 50.0
# A uniform (symmetric) displacement gradient -> uniform strain.
_A = np.array([
    [0.020, 0.010, 0.000],
    [0.010, -0.010, 0.000],
    [0.000, 0.000, -0.005],
])


class TestMixedUpPatch(unittest.TestCase):
    def test_constant_strain_patch(self) -> None:
        mesh = StructuredHexMesh(
            lengths=(1.0, 1.0, 1.0), divisions=(3, 3, 3),
        )
        layout_u = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
        layout_p = GlobalFieldLayout(name="p", finite_element=Q1_HEX)
        bc = DirichletBC(
            sideset_names=[
                "xmin_sides", "xmax_sides",
                "ymin_sides", "ymax_sides",
                "zmin_sides", "zmax_sides",
            ],
            field_name="u",
            dofs=(0, 1, 2),
            values=lambda coords, _t: np.asarray(coords) @ _A.T,
        )
        dof_map = build_dof_map(
            mesh, [layout_u, layout_p], [bc],
            components_by_field={"u": 3, "p": 1},
        )
        gr = SmallDispEquilibrium(ndims=3, mixed=True)
        elastic = Elastic(
            make_elastic_parameters(_KAPPA, _MU), def_type=DefType.FULL_3D,
        )
        fe_problem = build_fe_problem(
            mesh=mesh,
            dof_map=dof_map,
            gr=gr,
            models_by_block={"all": elastic},
        )

        state = FEState.from_problem(fe_problem)
        params_by_block = params_by_block_from_models(fe_problem)
        U_solved, _ = fe_newton_solve(
            fe_problem, params_by_block, U_prev=state.U_at(0), t=1.0,
        )

        # Displacement reproduced exactly: u = A·x, grad u = A.
        L2, H1 = l2_h1_errors(
            fe_problem, U_solved,
            u_exact=lambda coords: _A @ np.asarray(coords),
            grad_u_exact=lambda _coords: _A,
        )
        self.assertLess(L2, 1e-9, f"L2 {L2}")
        self.assertLess(H1, 1e-9, f"H1 {H1}")

        # Pressure is the uniform p = -kappa·tr(eps).
        p_exact = -_KAPPA * float(np.trace(_A))
        p_dofs = np.asarray(U_solved)[dof_map.block_offsets[1]:]
        self.assertTrue(
            np.allclose(p_dofs, p_exact, atol=1e-9),
            f"p in [{p_dofs.min()}, {p_dofs.max()}], expected {p_exact}",
        )


if __name__ == "__main__":
    unittest.main()
