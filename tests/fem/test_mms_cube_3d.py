"""Manufactured-solution convergence regression for SmallDispEquilibrium.

Verifies that :func:`cmad.fem.nonlinear_solver.fe_newton_solve` over a
structured hex mesh (and its hex-to-tet split) converges at the expected
linear-element rates against a smooth manufactured solution: L² rate
``>= 1.9`` and H¹ rate ``>= 0.9`` on each consecutive-N ratio.

The manufactured displacement vanishes on the boundary of the unit
cube, giving homogeneous Dirichlet on the union of all six face
nodesets. The body force is derived symbolically (via sympy) from
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
"""
import unittest
from collections.abc import Callable
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import sympy
from jax.tree_util import tree_map
from numpy.typing import NDArray
from sympy import Matrix, eye, lambdify, pi, sin, symbols

from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import FEProblem, FEState, build_fe_problem
from cmad.fem.interpolants import hex_linear, tet_linear
from cmad.fem.mesh import (
    Mesh,
    StructuredHexMesh,
    hex_to_tet_split,
)
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.fem.quadrature import hex_quadrature, tet_quadrature
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray

_KAPPA = 100.0
_MU = 50.0


def _make_parameters() -> Parameters:
    values = {"elastic": {"kappa": _KAPPA, "mu": _MU}}
    active_flags = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Parameters(values, active_flags, transforms)


def _build_mms_callables() -> tuple[
    Callable[[JaxArray, float], JaxArray],
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
]:
    """Symbolically derive ``(body_force_fn, u_exact, grad_u_exact)``.

    ``u_exact = sin(pi*x) * sin(pi*y) * sin(pi*z) * (1, 1, 1)`` vanishes
    on the boundary of the unit cube. ``b = -div(sigma(u_exact))`` with
    sigma matching
    :func:`cmad.models.elastic_stress.isotropic_linear_elastic_cauchy_stress`.
    The body-force callable lambdifies into ``jax`` so the FE pipeline's
    vmap/jit can trace it; the exact-solution callables lambdify into
    ``numpy`` since they're invoked from the python error-norm loop.
    """
    x, y, z = symbols("x y z", real=True)
    profile = sin(pi * x) * sin(pi * y) * sin(pi * z)
    u_sym = Matrix([profile, profile, profile])
    grad_u_sym = u_sym.jacobian([x, y, z])
    eps_sym = (grad_u_sym + grad_u_sym.T) / 2
    tr_eps = eps_sym.trace()
    dev_eps_sym = eps_sym - (tr_eps / 3) * eye(3)
    sigma_sym = _KAPPA * tr_eps * eye(3) + 2 * _MU * dev_eps_sym
    coord_syms = (x, y, z)
    b_sym = sympy.simplify(Matrix([
        -sum(sigma_sym[i, j].diff(coord_syms[j]) for j in range(3))
        for i in range(3)
    ]))

    b_callable = lambdify((x, y, z), b_sym, modules="jax")
    u_callable = lambdify((x, y, z), u_sym, modules="numpy")
    grad_u_callable = lambdify((x, y, z), grad_u_sym, modules="numpy")

    def body_force_fn(coords: JaxArray, _t: float) -> JaxArray:
        return jnp.asarray(
            b_callable(coords[0], coords[1], coords[2]),
        ).reshape(-1)

    def u_exact(coords: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.asarray(
            u_callable(coords[0], coords[1], coords[2]),
        ).reshape(-1)

    def grad_u_exact(coords: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.asarray(
            grad_u_callable(coords[0], coords[1], coords[2]),
        )

    return body_force_fn, u_exact, grad_u_exact


def _add_boundary_nodeset(mesh: Mesh) -> Mesh:
    """Augment ``mesh`` with ``"all_boundary_nodes"``: union of 6 faces.

    S3 forward-compat handoff recipe (a): nodeset unions are constructed
    at fixture level until a ``union_nodesets`` mesh helper materializes.
    """
    boundary = np.unique(np.concatenate([
        mesh.node_sets["xmin_nodes"], mesh.node_sets["xmax_nodes"],
        mesh.node_sets["ymin_nodes"], mesh.node_sets["ymax_nodes"],
        mesh.node_sets["zmin_nodes"], mesh.node_sets["zmax_nodes"],
    ]))
    return replace(
        mesh,
        node_sets={**mesh.node_sets, "all_boundary_nodes": boundary},
    )


def _build_fe_problem(
        mesh: Mesh,
        body_force_fn: Callable[[JaxArray, float], JaxArray],
) -> FEProblem:
    n_nodes = int(mesh.nodes.shape[0])
    layout = GlobalFieldLayout(
        name="u", num_basis_fns=n_nodes, num_dofs_per_basis_fn=3,
    )
    bc = DirichletBC(
        nodeset_name="all_boundary_nodes",
        field_name="u",
        dofs=(0, 1, 2),
        values=None,
    )
    dof_map = build_dof_map(mesh, [layout], [bc])
    if mesh.element_family == ElementFamily.HEX_LINEAR:
        num_basis_fns = 8
    elif mesh.element_family == ElementFamily.TET_LINEAR:
        num_basis_fns = 4
    else:
        raise ValueError(f"unsupported element family {mesh.element_family}")
    gr = SmallDispEquilibrium(num_basis_fns=num_basis_fns, ndims=3)
    elastic = Elastic(_make_parameters(), def_type=DefType.FULL_3D)
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=gr,
        models_by_block={"all": elastic},
        forcing_fns_by_block_idx={0: body_force_fn},
    )


def _l2_h1_errors(
        fe_problem: FEProblem,
        U_solved: NDArray[np.floating],
        u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        grad_u_exact: Callable[
            [NDArray[np.floating]], NDArray[np.floating],
        ],
) -> tuple[float, float]:
    """L² and H¹ errors against ``u_exact`` via degree-4 quadrature.

    Pure-numpy element-loop (test-only path, no jit). Reference-frame
    shape functions are precomputed once per quadrature rule; the
    isoparametric Jacobian and physical-frame gradient are evaluated
    with numpy for each ``(elem, ip)`` pair.
    """
    mesh = fe_problem.mesh
    dof_map = fe_problem.dof_map
    fam = mesh.element_family
    if fam == ElementFamily.HEX_LINEAR:
        norm_quad = hex_quadrature(degree=4)
        interpolant = hex_linear
    else:
        norm_quad = tet_quadrature(degree=4)
        interpolant = tet_linear

    nips = norm_quad.xi.shape[0]
    nnodes = mesh.connectivity.shape[1]
    N_ref = np.empty((nips, nnodes), dtype=np.float64)
    grad_N_ref = np.empty((nips, nnodes, 3), dtype=np.float64)
    for ip in range(nips):
        sh = interpolant(jnp.asarray(norm_quad.xi[ip]))
        N_ref[ip] = np.asarray(sh.N)
        grad_N_ref[ip] = np.asarray(sh.grad_N)
    w = np.asarray(norm_quad.w, dtype=np.float64)

    layout = dof_map.field_layouts[0]
    block_off = int(dof_map.block_offsets[0])
    ndofs = layout.num_dofs_per_basis_fn

    L2_sq = 0.0
    H1_grad_sq = 0.0

    for elem_idx in range(mesh.connectivity.shape[0]):
        node_ids = mesh.connectivity[elem_idx]
        X_elem = mesh.nodes[node_ids]
        eq = (
            block_off
            + node_ids[:, None] * ndofs
            + np.arange(ndofs)[None, :]
        ).ravel()
        U_elem = U_solved[eq].reshape(-1, ndofs)

        for ip in range(nips):
            iso_jac = X_elem.T @ grad_N_ref[ip]
            iso_jac_det = float(np.linalg.det(iso_jac))
            grad_N_phys = grad_N_ref[ip] @ np.linalg.inv(iso_jac)
            dv = iso_jac_det * float(w[ip])

            coords_ip = N_ref[ip] @ X_elem
            u_h_ip = N_ref[ip] @ U_elem
            grad_u_h_ip = U_elem.T @ grad_N_phys

            u_ex_ip = u_exact(coords_ip)
            grad_u_ex_ip = grad_u_exact(coords_ip)

            L2_sq += float(np.sum((u_h_ip - u_ex_ip) ** 2)) * dv
            H1_grad_sq += float(
                np.sum((grad_u_h_ip - grad_u_ex_ip) ** 2),
            ) * dv

    return float(np.sqrt(L2_sq)), float(np.sqrt(L2_sq + H1_grad_sq))


class TestMmsCube3D(unittest.TestCase):

    body_force_fn: Callable[[JaxArray, float], JaxArray]
    u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]]
    grad_u_exact: Callable[[NDArray[np.floating]], NDArray[np.floating]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.body_force_fn, cls.u_exact, cls.grad_u_exact = (
            _build_mms_callables()
        )

    def _solve_and_measure(self, mesh: Mesh) -> tuple[float, float]:
        fe_problem = _build_fe_problem(mesh, type(self).body_force_fn)
        state = FEState.from_problem(fe_problem)
        U_solved, n_iters, _ = fe_newton_solve(
            fe_problem, U_prev=state.U_at(0), t=1.0,
        )
        self.assertLessEqual(
            n_iters, 2,
            "linear elastic + closed-form Cauchy should converge in one "
            f"Newton iteration; got {n_iters}",
        )
        return _l2_h1_errors(
            fe_problem, U_solved,
            type(self).u_exact, type(self).grad_u_exact,
        )

    def test_hex_convergence_rates(self) -> None:
        Ns = (4, 8, 16)
        L2_errs: list[float] = []
        H1_errs: list[float] = []
        for N in Ns:
            mesh = _add_boundary_nodeset(StructuredHexMesh(
                lengths=(1.0, 1.0, 1.0), divisions=(N, N, N),
            ))
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
            hex_mesh = _add_boundary_nodeset(StructuredHexMesh(
                lengths=(1.0, 1.0, 1.0), divisions=(N, N, N),
            ))
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
