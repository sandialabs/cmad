"""Surface-flux MMS convergence regression for SmallDispEquilibrium.

Same manufactured ``u`` and body force as
:mod:`tests.fem.test_mms_cube_3d`, but replaces the homogeneous
Dirichlet clamp on two faces (``xmax_sides`` and ``ymax_sides``) with
analytic Neumann tractions ``t̄ = sigma(u_exact)·n̂`` derived from the
same symbolic stress. The four remaining cube faces keep homogeneous
Dirichlet, which is consistent with ``u_exact = 0`` on the entire
boundary of the unit cube.

The two NBCs declare separate sidesets (one ``xmax_sides``, one
``ymax_sides``) so the test exercises cross-NBC superposition: each
NBC scatters its contribution into R independently and the assembly
sums them at shared edges and corners. Two distinct
``(family, local_side_id)`` lift entries are exercised in one solve,
covering the lift-table-correctness ↔ scatter-correctness coupling
that single-face tests miss.

Convergence thresholds match the volume MMS (L² rate ``>= 1.9``,
H¹ rate ``>= 0.9``). Hex and tet both sweep ``N ∈ {8, 16}`` (one
asymptotic ratio per family). Tets are generated via
:func:`cmad.fem.mesh.hex_to_tet_split`.
"""
import unittest
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from sympy import Matrix, lambdify, pi, sin, symbols

from cmad.fem.bcs import DirichletBC, NeumannBC
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


def _build_traction_callable(
        sigma_sym: Matrix,
        coord_syms: Sequence[Any],
        n_hat: Sequence[float],
) -> Callable[
    [NDArray[np.floating] | JaxArray, float],
    NDArray[np.floating] | JaxArray,
]:
    """Lambdify ``sigma·n̂`` for an axis-aligned outward unit normal.

    Returns a callable matching the
    :class:`cmad.fem.bcs.NeumannBC` callable contract: input
    ``coords`` of shape ``(N_pts, 3)``, output traction of shape
    ``(N_pts, 3)``. The lambdified expression is per-point; vmap
    lifts it across the leading axis so the per-side evaluator can
    call it with any side-IP count.
    """
    t_sym = sigma_sym @ Matrix(list(n_hat))
    t_callable = lambdify(tuple(coord_syms), t_sym, modules="jax")

    def traction_fn(
            coords: NDArray[np.floating] | JaxArray, _t: float,
    ) -> NDArray[np.floating] | JaxArray:
        def at_point(coord: JaxArray) -> JaxArray:
            return jnp.asarray(
                t_callable(coord[0], coord[1], coord[2]),
            ).reshape(-1)
        return jax.vmap(at_point)(jnp.asarray(coords))

    return traction_fn


def _build_fe_problem(
        mesh: Mesh,
        body_force_fn: Callable[
            [NDArray[np.floating] | JaxArray, float],
            NDArray[np.floating] | JaxArray,
        ],
        traction_xmax_fn: Callable[
            [NDArray[np.floating] | JaxArray, float],
            NDArray[np.floating] | JaxArray,
        ],
        traction_ymax_fn: Callable[
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
    dbc = DirichletBC(
        sideset_names=[
            "xmin_sides", "ymin_sides",
            "zmin_sides", "zmax_sides",
        ],
        field_name="u",
        dofs=(0, 1, 2),
        values=None,
    )
    nbc_xmax = NeumannBC(
        sideset_names=["xmax_sides"],
        field_name="u",
        values=traction_xmax_fn,
    )
    nbc_ymax = NeumannBC(
        sideset_names=["ymax_sides"],
        field_name="u",
        values=traction_ymax_fn,
    )
    dof_map = build_dof_map(
        mesh, [layout], [dbc], components_by_field={"u": 3},
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
        neumann_bcs=(nbc_xmax, nbc_ymax),
    )


class TestMmsCube3DNeumann(unittest.TestCase):

    body_force_fn: Callable[
        [NDArray[np.floating] | JaxArray, float],
        NDArray[np.floating] | JaxArray,
    ]
    traction_xmax_fn: Callable[
        [NDArray[np.floating] | JaxArray, float],
        NDArray[np.floating] | JaxArray,
    ]
    traction_ymax_fn: Callable[
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
            sigma_sym,
        ) = build_mms_callables(u_sym, (x, y, z), _KAPPA, _MU)
        cls.traction_xmax_fn = _build_traction_callable(
            sigma_sym, (x, y, z), [1.0, 0.0, 0.0],
        )
        cls.traction_ymax_fn = _build_traction_callable(
            sigma_sym, (x, y, z), [0.0, 1.0, 0.0],
        )

    def _solve_and_measure(self, mesh: Mesh) -> tuple[float, float]:
        fe_problem = _build_fe_problem(
            mesh,
            type(self).body_force_fn,
            type(self).traction_xmax_fn,
            type(self).traction_ymax_fn,
        )
        L2, H1, n_iters = solve_and_measure(
            fe_problem, type(self).u_exact, type(self).grad_u_exact,
        )
        self.assertLessEqual(
            n_iters, 2,
            "linear elastic + closed-form Cauchy + analytic traction "
            f"should converge in one Newton iteration; got {n_iters}",
        )
        return L2, H1

    def test_hex_convergence_rates(self) -> None:
        Ns = (8, 16)
        L2_errs: list[float] = []
        H1_errs: list[float] = []
        for N in Ns:
            mesh = StructuredHexMesh(
                lengths=(1.0, 1.0, 1.0), divisions=(N, N, N),
            )
            L2, H1 = self._solve_and_measure(mesh)
            L2_errs.append(L2)
            H1_errs.append(H1)
        L2_rate = float(np.log2(L2_errs[0] / L2_errs[1]))
        H1_rate = float(np.log2(H1_errs[0] / H1_errs[1]))
        self.assertGreaterEqual(L2_rate, 1.9, f"L2 rate {L2_rate}")
        self.assertGreaterEqual(H1_rate, 0.9, f"H1 rate {H1_rate}")

    def test_tet_convergence_rates(self) -> None:
        Ns = (8, 16)
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
