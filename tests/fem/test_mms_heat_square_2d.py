"""Manufactured solution convergence for heat transfer on the unit square.

The manufactured temperature vanishes on the boundary, giving homogeneous
Dirichlet on the four side sets; the source is ``-div(k grad T)`` of it.
Quad sweep ``N in {4, 8, 16}`` and the tri split at ``N in {4, 8}``, L2
rate at least 1.9 and H1 rate at least 0.9.
"""
import unittest
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sympy import pi, sin, symbols

from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import P1_TRI, Q1_QUAD
from cmad.fem.mesh import Mesh, StructuredQuadMesh, quad_to_tri_split
from cmad.global_residuals.heat_transfer import HeatTransfer
from cmad.models.conduction import Conduction
from cmad.typing import JaxArray
from tests.fem._mms_helpers import (
    build_heat_mms_callables,
    make_conduction_parameters,
    solve_and_measure,
)

_K = 16.0


def _build_fe_problem(
        mesh: Mesh,
        source_fn: Callable[
            [NDArray[np.floating] | JaxArray, float],
            NDArray[np.floating] | JaxArray,
        ],
) -> FEProblem:
    if mesh.element_family == ElementFamily.QUAD_LINEAR:
        fe = Q1_QUAD
    elif mesh.element_family == ElementFamily.TRI_LINEAR:
        fe = P1_TRI
    else:
        raise ValueError(f"unsupported element family {mesh.element_family}")
    layout = GlobalFieldLayout(name="T", finite_element=fe)
    bc = DirichletBC(
        sideset_names=[
            "xmin_sides", "xmax_sides", "ymin_sides", "ymax_sides",
        ],
        field_name="T",
        dofs=(0,),
        values=None,
    )
    dof_map = build_dof_map(
        mesh, [layout], [bc], components_by_field={"T": 1},
    )
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=HeatTransfer(ndims=2),
        models_by_block={"all": Conduction(make_conduction_parameters(_K))},
        forcing_fns_by_block_idx={0: source_fn},
    )


class TestMmsHeatSquare2D(unittest.TestCase):

    source_fn: Callable[
        [NDArray[np.floating] | JaxArray, float],
        NDArray[np.floating] | JaxArray,
    ]
    T_exact: Callable[..., NDArray[np.floating]]
    grad_T_exact: Callable[..., NDArray[np.floating]]

    @classmethod
    def setUpClass(cls) -> None:
        x, y = symbols("x y", real=True)
        T_sym = sin(pi * x) * sin(pi * y)
        cls.source_fn, cls.T_exact, cls.grad_T_exact = (
            build_heat_mms_callables(T_sym, (x, y), _K)
        )

    def _solve_and_measure(self, mesh: Mesh) -> tuple[float, float]:
        fe_problem = _build_fe_problem(mesh, type(self).source_fn)
        return solve_and_measure(
            fe_problem, type(self).T_exact, type(self).grad_T_exact,
        )

    def test_quad_convergence_rates(self) -> None:
        Ns = (4, 8, 16)
        L2_errs: list[float] = []
        H1_errs: list[float] = []
        for N in Ns:
            mesh = StructuredQuadMesh(lengths=(1.0, 1.0), divisions=(N, N))
            L2, H1 = self._solve_and_measure(mesh)
            L2_errs.append(L2)
            H1_errs.append(H1)
        L2_rates = [
            np.log2(L2_errs[i] / L2_errs[i + 1]) for i in range(len(Ns) - 1)
        ]
        H1_rates = [
            np.log2(H1_errs[i] / H1_errs[i + 1]) for i in range(len(Ns) - 1)
        ]
        for r in L2_rates:
            self.assertGreaterEqual(r, 1.9, f"L2 rates {L2_rates}")
        for r in H1_rates:
            self.assertGreaterEqual(r, 0.9, f"H1 rates {H1_rates}")

    def test_tri_convergence_rates(self) -> None:
        Ns = (4, 8)
        L2_errs: list[float] = []
        H1_errs: list[float] = []
        for N in Ns:
            quad_mesh = StructuredQuadMesh(
                lengths=(1.0, 1.0), divisions=(N, N),
            )
            L2, H1 = self._solve_and_measure(quad_to_tri_split(quad_mesh))
            L2_errs.append(L2)
            H1_errs.append(H1)
        L2_rate = float(np.log2(L2_errs[0] / L2_errs[1]))
        H1_rate = float(np.log2(H1_errs[0] / H1_errs[1]))
        self.assertGreaterEqual(L2_rate, 1.9, f"L2 rate {L2_rate}")
        self.assertGreaterEqual(H1_rate, 0.9, f"H1 rate {H1_rate}")


if __name__ == "__main__":
    unittest.main()
