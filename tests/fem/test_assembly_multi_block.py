"""Multi-residual-block scatter regression for assemble_element_block.

A 1-tet mesh paired with a mock 2-block GR (var_names=["u", "p"],
num_eqs=[3, 1]) and a mock R_and_dR_dU evaluator returning known
per-IP constants. Verifies the nested-loop scatter:

  - per-block R contributions land in the correct global eq ranges
    via ``gr.var_names[r]`` → field_idx lookup;
  - the (r, s) tangent scatter populates all four (uu, up, pu, pp)
    sub-blocks with correct rows / cols / values, including the
    off-diagonal blocks that single-block tests miss.

Both fields share the same per-element basis-fn count (4, the linear
tet's connectivity size) and only differ in num_eqs (3 for u, 1 for
p). The structural asymmetry tested here (different num_eqs → all
four K sub-blocks have distinct shapes) is the mistake-prone part of
the scatter that single-block tests miss.

The test bypasses ``for_model`` and constructs
``FEProblem.evaluators_by_block`` manually so the mock evaluator's
constants are deterministic and the scatter is the only thing under
test. A 1-element / 1-IP setup makes per-IP contributions equal the
final accumulated values (no quadrature scaling).
"""
import unittest
from typing import cast

import jax.numpy as jnp
import numpy as np

from cmad.fem.assembly import assemble_element_block
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEProblem
from cmad.fem.interpolants import tet_linear
from cmad.fem.mesh import ElementFamily, Mesh
from cmad.fem.quadrature import tet_quadrature
from cmad.global_residuals.global_residual import GlobalResidual
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.model import Model
from cmad.models.var_types import VarType
from cmad.typing import GREvaluators, ResidualFnGR


_R_U_PER_IP = 1.0
_R_P_PER_IP = 2.0
_K_UU_PER_IP = 10.0
_K_UP_PER_IP = 20.0
_K_PU_PER_IP = 30.0
_K_PP_PER_IP = 40.0


class _TwoBlockMockGR(GlobalResidual):
    """Mock GR with 2 residual blocks: u (vector, 3 eqs), p (scalar, 1 eq).

    residual_fn is a no-op stub — this GR is paired with a hand-built
    evaluators dict in the test, not via :meth:`for_model`.
    """
    def __init__(self) -> None:
        self._is_complex = False
        self.dtype = float
        self._ndims = 3

        self._init_residuals(2)
        self._var_types[0] = VarType.VECTOR
        self._num_eqs[0] = 3
        self._num_basis_fns[0] = 4
        self.resid_names[0] = "displacement"
        self.var_names[0] = "u"
        self._var_types[1] = VarType.SCALAR
        self._num_eqs[1] = 1
        self._num_basis_fns[1] = 4
        self.resid_names[1] = "pressure"
        self.var_names[1] = "p"
        self._init_element_dof_layout()

        super().__init__(cast(ResidualFnGR, lambda *args: []))


class _MockModel:
    """Minimal Model stand-in exposing only the attributes assembly reads."""

    class _Params:
        values: dict[str, object] = {}

    parameters = _Params()
    _init_xi = [jnp.zeros(0)]


def _mock_R_and_dR_dU(
        xi, xi_prev, params, U, U_prev, shapes_ip, w, dv, ip_set,
):
    R_u = jnp.ones((4, 3)) * _R_U_PER_IP
    R_p = jnp.ones((4, 1)) * _R_P_PER_IP
    K_uu = jnp.ones((4, 3, 4, 3)) * _K_UU_PER_IP
    K_up = jnp.ones((4, 3, 4, 1)) * _K_UP_PER_IP
    K_pu = jnp.ones((4, 1, 4, 3)) * _K_PU_PER_IP
    K_pp = jnp.ones((4, 1, 4, 1)) * _K_PP_PER_IP
    return [R_u, R_p], [[K_uu, K_up], [K_pu, K_pp]]


def _build_unit_tet_mesh() -> Mesh:
    """Single linear tet at the unit-tet vertices."""
    nodes = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.intp)
    return Mesh(
        nodes=nodes,
        connectivity=connectivity,
        element_family=ElementFamily.TET_LINEAR,
        element_blocks={"all": np.array([0], dtype=np.intp)},
        node_sets={},
        side_sets={},
    )


def _build_fe_problem() -> FEProblem:
    mesh = _build_unit_tet_mesh()
    n_nodes = int(mesh.nodes.shape[0])
    layout_u = GlobalFieldLayout(
        name="u", num_basis_fns=n_nodes, num_dofs_per_basis_fn=3,
    )
    layout_p = GlobalFieldLayout(
        name="p", num_basis_fns=n_nodes, num_dofs_per_basis_fn=1,
    )
    dof_map = build_dof_map(mesh, [layout_u, layout_p], [])
    gr = _TwoBlockMockGR()
    evaluators: GREvaluators = {"R_and_dR_dU": _mock_R_and_dR_dU}
    return FEProblem(
        mesh=mesh,
        dof_map=dof_map,
        gr=gr,
        models_by_block=cast(dict[str, Model], {"all": _MockModel()}),
        modes_by_block={"all": GlobalResidualMode.CLOSED_FORM},
        evaluators_by_block={"all": evaluators},
        forcing_fns_by_block_idx=None,
        assembly_quadrature={
            ElementFamily.TET_LINEAR: tet_quadrature(degree=1),
        },
        interpolant_fn={ElementFamily.TET_LINEAR: tet_linear},
    )


class TestAssemblyMultiBlock(unittest.TestCase):
    """Verify multi-residual-block scatter in assemble_element_block."""

    def setUp(self) -> None:
        self.fe_problem = _build_fe_problem()
        n_dofs = self.fe_problem.dof_map.num_total_dofs
        self.U = np.zeros(n_dofs)
        self.U_prev = np.zeros(n_dofs)
        # u-eqs: 0..11 (4 nodes * 3 dofs). p-eqs: 12..15.
        self.n_dofs_u = 12
        self.n_dofs_p = 4

    def test_R_block_scatter_lands_in_correct_field_eqs(self) -> None:
        _, _, _, R = assemble_element_block(
            self.fe_problem, "all", self.U, self.U_prev, t=0.0,
        )
        self.assertEqual(R.shape, (self.n_dofs_u + self.n_dofs_p,))
        np.testing.assert_allclose(R[:self.n_dofs_u], _R_U_PER_IP)
        np.testing.assert_allclose(R[self.n_dofs_u:], _R_P_PER_IP)

    def test_K_scatter_populates_all_four_block_pairs(self) -> None:
        rows, cols, vals, _ = assemble_element_block(
            self.fe_problem, "all", self.U, self.U_prev, t=0.0,
        )
        u_rows = rows < self.n_dofs_u
        u_cols = cols < self.n_dofs_u
        p_rows = ~u_rows
        p_cols = ~u_cols

        uu_vals = vals[u_rows & u_cols]
        up_vals = vals[u_rows & p_cols]
        pu_vals = vals[p_rows & u_cols]
        pp_vals = vals[p_rows & p_cols]

        self.assertEqual(uu_vals.shape, (self.n_dofs_u * self.n_dofs_u,))
        self.assertEqual(up_vals.shape, (self.n_dofs_u * self.n_dofs_p,))
        self.assertEqual(pu_vals.shape, (self.n_dofs_p * self.n_dofs_u,))
        self.assertEqual(pp_vals.shape, (self.n_dofs_p * self.n_dofs_p,))

        np.testing.assert_allclose(uu_vals, _K_UU_PER_IP)
        np.testing.assert_allclose(up_vals, _K_UP_PER_IP)
        np.testing.assert_allclose(pu_vals, _K_PU_PER_IP)
        np.testing.assert_allclose(pp_vals, _K_PP_PER_IP)

    def test_K_off_diagonal_uses_correct_row_col_eq_ranges(self) -> None:
        """up entries: u-eq rows, p-eq cols. pu entries: p-eq rows, u-eq cols."""
        rows, cols, _, _ = assemble_element_block(
            self.fe_problem, "all", self.U, self.U_prev, t=0.0,
        )
        u_rows = rows < self.n_dofs_u
        u_cols = cols < self.n_dofs_u
        p_rows = ~u_rows
        p_cols = ~u_cols

        up_rows = rows[u_rows & p_cols]
        up_cols = cols[u_rows & p_cols]
        self.assertTrue((up_rows >= 0).all() and
                        (up_rows < self.n_dofs_u).all())
        self.assertTrue((up_cols >= self.n_dofs_u).all() and
                        (up_cols < self.n_dofs_u + self.n_dofs_p).all())

        pu_rows = rows[p_rows & u_cols]
        pu_cols = cols[p_rows & u_cols]
        self.assertTrue((pu_rows >= self.n_dofs_u).all() and
                        (pu_rows < self.n_dofs_u + self.n_dofs_p).all())
        self.assertTrue((pu_cols >= 0).all() and
                        (pu_cols < self.n_dofs_u).all())


if __name__ == "__main__":
    unittest.main()
