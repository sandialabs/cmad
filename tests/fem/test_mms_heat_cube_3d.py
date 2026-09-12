"""Manufactured solution convergence for heat transfer on the unit cube.

The manufactured temperature vanishes on the boundary, giving homogeneous
Dirichlet on the six face side sets; the source is ``-div(k grad T)`` of
it. Hex sweep ``N in {4, 8, 16}`` and the tet split at ``N in {8, 16}``,
L2 rate at least 1.9 and H1 rate at least 0.9, as for the mechanics MMS.
The tet sweep starts at 8 because the split of the 4 cubed mesh is not
yet asymptotic for this problem: measured L2 rates 1.82 from 4 to 8 and
1.95 from 8 to 16, H1 rates 0.94 and 0.98.
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
from cmad.fem.finite_element import P1_TET, Q1_HEX
from cmad.fem.mesh import Mesh, StructuredHexMesh, hex_to_tet_split
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
    if mesh.element_family == ElementFamily.HEX_LINEAR:
        fe = Q1_HEX
    elif mesh.element_family == ElementFamily.TET_LINEAR:
        fe = P1_TET
    else:
        raise ValueError(f"unsupported element family {mesh.element_family}")
    layout = GlobalFieldLayout(name="T", finite_element=fe)
    bc = DirichletBC(
        sideset_names=[
            "xmin_sides", "xmax_sides",
            "ymin_sides", "ymax_sides",
            "zmin_sides", "zmax_sides",
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
        gr=HeatTransfer(ndims=3),
        models_by_block={"all": Conduction(make_conduction_parameters(_K))},
        forcing_fns_by_block_idx={0: source_fn},
    )


class TestMmsHeatCube3D(unittest.TestCase):

    source_fn: Callable[
        [NDArray[np.floating] | JaxArray, float],
        NDArray[np.floating] | JaxArray,
    ]
    T_exact: Callable[..., NDArray[np.floating]]
    grad_T_exact: Callable[..., NDArray[np.floating]]

    @classmethod
    def setUpClass(cls) -> None:
        x, y, z = symbols("x y z", real=True)
        T_sym = sin(pi * x) * sin(pi * y) * sin(pi * z)
        cls.source_fn, cls.T_exact, cls.grad_T_exact = (
            build_heat_mms_callables(T_sym, (x, y, z), _K)
        )

    def _solve_and_measure(self, mesh: Mesh) -> tuple[float, float]:
        fe_problem = _build_fe_problem(mesh, type(self).source_fn)
        return solve_and_measure(
            fe_problem, type(self).T_exact, type(self).grad_T_exact,
        )

    def test_hex_convergence_rates(self) -> None:
        Ns = (4, 8, 16)
        L2_errs: list[float] = []
        H1_errs: list[float] = []
        for N in Ns:
            mesh = StructuredHexMesh(
                lengths=(1.0, 1.0, 1.0), divisions=(N, N, N),
            )
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

    def test_tet_convergence_rates(self) -> None:
        Ns = (8, 16)
        L2_errs: list[float] = []
        H1_errs: list[float] = []
        for N in Ns:
            hex_mesh = StructuredHexMesh(
                lengths=(1.0, 1.0, 1.0), divisions=(N, N, N),
            )
            L2, H1 = self._solve_and_measure(hex_to_tet_split(hex_mesh))
            L2_errs.append(L2)
            H1_errs.append(H1)
        L2_rate = float(np.log2(L2_errs[0] / L2_errs[1]))
        H1_rate = float(np.log2(H1_errs[0] / H1_errs[1]))
        self.assertGreaterEqual(L2_rate, 1.9, f"L2 rate {L2_rate}")
        self.assertGreaterEqual(H1_rate, 0.9, f"H1 rate {H1_rate}")


if __name__ == "__main__":
    unittest.main()
