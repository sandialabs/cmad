"""Manufactured-solution convergence regression for SmallDispEquilibrium.

Verifies that :func:`cmad.fem.nonlinear_solver.fe_newton_solve` over a
structured hex mesh (and its hex-to-tet split) converges at the expected
linear-element rates against a smooth manufactured solution: L² rate
``>= 1.9`` and H¹ rate ``>= 0.9`` on each consecutive-N ratio.

The manufactured displacement vanishes on the boundary of the unit
cube, giving homogeneous Dirichlet on the union of all six face
side sets. The body force is derived symbolically (via sympy) from
``-div(sigma(u_exact))`` using the same isotropic small-strain Cauchy
formula as :class:`cmad.models.elastic.Elastic` —
``sigma = kappa * tr(epsilon) * I + 2 * mu * dev(epsilon)``,
``epsilon = (grad u + (grad u)^T) / 2``,
``dev(epsilon) = epsilon - tr(epsilon) / 3 * I`` — then lambdified into
a JAX callable so the FE pipeline sees the matching analytical source.

Hex sweep ``N ∈ {4, 8, 16}`` (two consecutive ratios) establishes the
rate; tet sweep ``N ∈ {4, 8}`` via ``hex_to_tet_split`` (one ratio)
confirms it. Tet sweep is shallower because a 6-tet split of an 8³ hex
mesh already produces 3072 tets, which keeps the test inside a typical
pytest budget while still validating the tet assembly path end-to-end.

The MMS scaffolding (symbolic → callables, error norms, Newton drive)
lives in :mod:`tests.fem._mms_helpers` so other manufactured-solution
tests can reuse it; only the cube-specific FE-problem build (clamped
on all six faces) stays here.
"""
import unittest
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sympy import Matrix, pi, sin, symbols

from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import P1_TET, Q1_HEX
from cmad.fem.mesh import (
    Mesh,
    StructuredHexMesh,
    hex_to_tet_split,
)
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.typing import JaxArray
from tests.fem._mms_helpers import (
    build_mms_callables,
    make_elastic_parameters,
    solve_and_measure,
)

_KAPPA = 100.0
_MU = 50.0


def _build_fe_problem(
        mesh: Mesh,
        body_force_fn: Callable[
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
    layout = GlobalFieldLayout(name="u", finite_element=fe)
    bc = DirichletBC(
        sideset_names=[
            "xmin_sides", "xmax_sides",
            "ymin_sides", "ymax_sides",
            "zmin_sides", "zmax_sides",
        ],
        field_name="u",
        dofs=(0, 1, 2),
        values=None,
    )
    dof_map = build_dof_map(
        mesh, [layout], [bc], components_by_field={"u": 3},
    )
    gr = SmallDispEquilibrium(ndims=3)
    elastic = Elastic(
        make_elastic_parameters(_KAPPA, _MU), def_type=DefType.FULL_3D,
    )
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=gr,
        models_by_block={"all": elastic},
        forcing_fns_by_block_idx={0: body_force_fn},
    )


class TestMmsCube3D(unittest.TestCase):

    body_force_fn: Callable[
        [NDArray[np.floating] | JaxArray, float],
        NDArray[np.floating] | JaxArray,
    ]
    u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]]
    grad_u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]]

    @classmethod
    def setUpClass(cls) -> None:
        x, y, z = symbols("x y z", real=True)
        profile = sin(pi * x) * sin(pi * y) * sin(pi * z)
        u_sym = Matrix([profile, profile, profile])
        (
            cls.body_force_fn,
            cls.u_exact,
            cls.grad_u_exact,
            _,
        ) = build_mms_callables(u_sym, (x, y, z), _KAPPA, _MU)

    def _solve_and_measure(self, mesh: Mesh) -> tuple[float, float]:
        fe_problem = _build_fe_problem(mesh, type(self).body_force_fn)
        L2, H1, n_iters = solve_and_measure(
            fe_problem, type(self).u_exact, type(self).grad_u_exact,
        )
        self.assertLessEqual(
            n_iters, 2,
            "linear elastic + closed-form Cauchy should converge in one "
            f"Newton iteration; got {n_iters}",
        )
        return L2, H1

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
            np.log2(L2_errs[i] / L2_errs[i + 1])
            for i in range(len(Ns) - 1)
        ]
        H1_rates = [
            np.log2(H1_errs[i] / H1_errs[i + 1])
            for i in range(len(Ns) - 1)
        ]
        for r in L2_rates:
            self.assertGreaterEqual(r, 1.9, f"L2 rates {L2_rates}")
        for r in H1_rates:
            self.assertGreaterEqual(r, 0.9, f"H1 rates {H1_rates}")

    def test_tet_convergence_rates(self) -> None:
        Ns = (4, 8)
        L2_errs: list[float] = []
        H1_errs: list[float] = []
        for N in Ns:
            hex_mesh = StructuredHexMesh(
                lengths=(1.0, 1.0, 1.0), divisions=(N, N, N),
            )
            tet_mesh = hex_to_tet_split(hex_mesh)
            L2, H1 = self._solve_and_measure(tet_mesh)
            L2_errs.append(L2)
            H1_errs.append(H1)
        L2_rate = float(np.log2(L2_errs[0] / L2_errs[1]))
        H1_rate = float(np.log2(H1_errs[0] / H1_errs[1]))
        self.assertGreaterEqual(L2_rate, 1.9, f"L2 rate {L2_rate}")
        self.assertGreaterEqual(H1_rate, 0.9, f"H1 rate {H1_rate}")


if __name__ == "__main__":
    unittest.main()
