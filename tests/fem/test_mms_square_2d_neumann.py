"""Edge traction MMS convergence for 2D plane strain Neumann BCs.

The 2D analog of :mod:`tests.fem.test_mms_cube_3d_neumann`: the plane
strain manufactured solution of :mod:`tests.fem.test_mms_square_2d`, but
the +x and +y edges carry analytic Neumann tractions ``t̄ = sigma . n``
(from the manufactured stress) instead of the homogeneous Dirichlet
clamp. The -x and -y edges keep homogeneous Dirichlet, consistent with
``u_exact = 0`` on the whole unit square boundary. Two NBCs on separate
sidesets exercise superposition across the two NBCs and both edge lifts
in one solve. L2 rate ``>= 1.9`` and H1 rate ``>= 0.9`` on quad and (via
``quad_to_tri_split``) tri.
"""
import unittest
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from sympy import Matrix, pi, sin, symbols

from cmad.fem.bcs import DirichletBC, NeumannBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import P1_TRI, Q1_QUAD
from cmad.fem.mesh import Mesh, StructuredQuadMesh, quad_to_tri_split
from cmad.global_residuals.mechanics import Mechanics
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.typing import JaxArray
from tests.fem._mms_helpers import (
    build_plane_strain_mms_callables,
    make_elastic_parameters,
    solve_and_measure,
)

_KAPPA = 100.0
_MU = 50.0

_ForcingFn = Callable[
    [NDArray[np.floating] | JaxArray, float | JaxArray],
    NDArray[np.floating] | JaxArray,
]


def _build_traction_callable(
        sigma_fn: Callable[[JaxArray], JaxArray],
        n_hat: Sequence[float],
) -> _ForcingFn:
    """Wrap ``sigma_fn(X) . n_hat`` into the NeumannBC callable contract.

    ``sigma_fn`` is the single point in-plane stress from
    :func:`build_plane_strain_mms_callables`; vmap lifts ``sigma . n``
    across the leading point axis so the side evaluator can call it at
    any side IP count.
    """
    n_vec = jnp.asarray(n_hat)

    def traction_fn(
            coords: NDArray[np.floating] | JaxArray,
            _t: float | JaxArray,
    ) -> NDArray[np.floating] | JaxArray:
        return jax.vmap(lambda c: sigma_fn(c) @ n_vec)(jnp.asarray(coords))

    return traction_fn


def _build_fe_problem(
        mesh: Mesh,
        body_force_fn: _ForcingFn,
        traction_xmax_fn: _ForcingFn,
        traction_ymax_fn: _ForcingFn,
) -> FEProblem:
    if mesh.element_family == ElementFamily.QUAD_LINEAR:
        fe = Q1_QUAD
    elif mesh.element_family == ElementFamily.TRI_LINEAR:
        fe = P1_TRI
    else:
        raise ValueError(f"unsupported element family {mesh.element_family}")
    layout = GlobalFieldLayout(name="u", finite_element=fe)
    dbc = DirichletBC(
        sideset_names=["xmin_sides", "ymin_sides"],
        field_name="u",
        dofs=(0, 1),
        values=None,
    )
    nbc_xmax = NeumannBC(
        sideset_names=["xmax_sides"], field_name="u", values=traction_xmax_fn,
    )
    nbc_ymax = NeumannBC(
        sideset_names=["ymax_sides"], field_name="u", values=traction_ymax_fn,
    )
    dof_map = build_dof_map(
        mesh, [layout], [dbc], components_by_field={"u": 2},
    )
    gr = Mechanics(ndims=2)
    elastic = Elastic(
        make_elastic_parameters(_KAPPA, _MU), def_type=DefType.PLANE_STRAIN,
    )
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=gr,
        models_by_block={"all": elastic},
        forcing_fns_by_block_idx={0: body_force_fn},
        neumann_bcs=(nbc_xmax, nbc_ymax),
    )


class TestMmsSquare2DNeumann(unittest.TestCase):

    body_force_fn: _ForcingFn
    traction_xmax_fn: _ForcingFn
    traction_ymax_fn: _ForcingFn
    u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]]
    grad_u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]]

    @classmethod
    def setUpClass(cls) -> None:
        x, y = symbols("x y", real=True)
        profile = sin(pi * x) * sin(pi * y)
        u_sym = Matrix([profile, profile])
        (
            cls.body_force_fn,
            cls.u_exact,
            cls.grad_u_exact,
            sigma_fn,
        ) = build_plane_strain_mms_callables(u_sym, (x, y), _KAPPA, _MU)
        cls.traction_xmax_fn = _build_traction_callable(sigma_fn, [1.0, 0.0])
        cls.traction_ymax_fn = _build_traction_callable(sigma_fn, [0.0, 1.0])

    def _solve_and_measure(self, mesh: Mesh) -> tuple[float, float]:
        fe_problem = _build_fe_problem(
            mesh,
            type(self).body_force_fn,
            type(self).traction_xmax_fn,
            type(self).traction_ymax_fn,
        )
        return solve_and_measure(
            fe_problem, type(self).u_exact, type(self).grad_u_exact,
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
            tri_mesh = quad_to_tri_split(quad_mesh)
            L2, H1 = self._solve_and_measure(tri_mesh)
            L2_errs.append(L2)
            H1_errs.append(H1)
        L2_rate = float(np.log2(L2_errs[0] / L2_errs[1]))
        H1_rate = float(np.log2(H1_errs[0] / H1_errs[1]))
        self.assertGreaterEqual(L2_rate, 1.9, f"L2 rate {L2_rate}")
        self.assertGreaterEqual(H1_rate, 0.9, f"H1 rate {H1_rate}")


if __name__ == "__main__":
    unittest.main()
