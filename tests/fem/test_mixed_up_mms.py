"""Convergence (MMS) check for the mixed (u-p) SmallDispEquilibrium.

The manufactured displacement is divergence-free, so the exact pressure
is zero and the grad(p) stabilization stays consistent: no manufactured
source is needed for it, and ``build_mms_callables`` produces the right
body force on its own (the ``kappa*tr(eps)`` term drops out). The mixed
solve recovers ``u`` at the optimal rates for linear elements (L2 >= 1.9,
H1 >= 0.9) under refinement.
"""
import unittest
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sympy import Matrix, pi, sin, symbols

from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import Mesh, StructuredHexMesh
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from tests.fem._mms_helpers import (
    build_mms_callables,
    make_elastic_parameters,
    solve_and_measure,
)

_KAPPA = 100.0
_MU = 50.0
_FACES = [
    "xmin_sides", "xmax_sides", "ymin_sides",
    "ymax_sides", "zmin_sides", "zmax_sides",
]


def _build_mixed(
        mesh: Mesh,
        body_force_fn: Callable[..., NDArray[np.floating]],
        u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]],
) -> FEProblem:
    """Mixed u-p FE problem with u = u_exact prescribed on every face."""
    def values(
            coords: NDArray[np.floating], _t: float,
    ) -> NDArray[np.floating]:
        return np.stack([u_exact(c) for c in np.asarray(coords)])

    bc = DirichletBC(
        sideset_names=_FACES, field_name="u", dofs=(0, 1, 2), values=values,
    )
    layouts = [
        GlobalFieldLayout(name="u", finite_element=Q1_HEX),
        GlobalFieldLayout(name="p", finite_element=Q1_HEX),
    ]
    dof_map = build_dof_map(
        mesh, layouts, [bc], components_by_field={"u": 3, "p": 1},
    )
    elastic = Elastic(
        make_elastic_parameters(_KAPPA, _MU), def_type=DefType.FULL_3D,
    )
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=SmallDispEquilibrium(ndims=3, mixed=True),
        models_by_block={"all": elastic},
        forcing_fns_by_block_idx={0: body_force_fn},
    )


class TestMixedUpMms(unittest.TestCase):

    body_force_fn: Callable[..., NDArray[np.floating]]
    u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]]
    grad_u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]]

    @classmethod
    def setUpClass(cls) -> None:
        x, y, z = symbols("x y z", real=True)
        # Each component is independent of its own coordinate -> div u = 0.
        u_sym = Matrix([
            sin(pi * y) * sin(pi * z),
            sin(pi * z) * sin(pi * x),
            sin(pi * x) * sin(pi * y),
        ])
        cls.body_force_fn, cls.u_exact, cls.grad_u_exact, _ = (
            build_mms_callables(u_sym, (x, y, z), _KAPPA, _MU)
        )

    def test_convergence_rates(self) -> None:
        Ns = (4, 8, 16)
        L2s: list[float] = []
        H1s: list[float] = []
        for N in Ns:
            mesh = StructuredHexMesh(
                lengths=(1.0, 1.0, 1.0), divisions=(N, N, N),
            )
            fe = _build_mixed(
                mesh, type(self).body_force_fn, type(self).u_exact,
            )
            L2, H1 = solve_and_measure(
                fe, type(self).u_exact, type(self).grad_u_exact,
            )
            L2s.append(L2)
            H1s.append(H1)
        L2_rates = [np.log2(L2s[i] / L2s[i + 1]) for i in range(len(Ns) - 1)]
        H1_rates = [np.log2(H1s[i] / H1s[i + 1]) for i in range(len(Ns) - 1)]
        for r in L2_rates:
            self.assertGreaterEqual(r, 1.9, f"L2 rates {L2_rates}")
        for r in H1_rates:
            self.assertGreaterEqual(r, 0.9, f"H1 rates {H1_rates}")


if __name__ == "__main__":
    unittest.main()
