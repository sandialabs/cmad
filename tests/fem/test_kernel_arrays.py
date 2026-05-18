"""Tests for :mod:`cmad.fem.kernel_arrays`.

:func:`build_fe_kernel_arrays` must produce index arrays consistent with
the assembly layer's index-derivation helpers: the carrier's U-gather
and R-scatter eq arrays are checked against :func:`_element_eq_indices`,
and the deduped COO ``(rows, cols)`` plus the dedup scatter against
:func:`assembled_coo_dedup`, on hex and tet meshes.
"""
import unittest
from typing import ClassVar, cast

import jax.numpy as jnp
import numpy as np

from cmad.fem.assembly import (
    _element_eq_indices,
    assembled_coo_dedup,
)
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import FEProblem
from cmad.fem.finite_element import P1_TET, Q1_HEX, FiniteElement
from cmad.fem.mesh import Mesh, StructuredHexMesh, hex_to_tet_split
from cmad.fem.quadrature import (
    QuadratureRule,
    hex_quadrature,
    quad_quadrature,
    tet_quadrature,
    tri_quadrature,
)
from cmad.global_residuals.global_residual import GlobalResidual
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.model import Model
from cmad.models.var_types import VarType
from cmad.typing import GREvaluators, JaxArray, ResidualFnGR


class _DisplacementMockGR(GlobalResidual):
    """Single-residual-block GR: one vector displacement field, 3 eqs.

    residual_fn is a stub — the kernel-array test inspects only index
    arrays and never assembles, so the GR is paired with a hand-built
    evaluators dict.
    """

    def __init__(self) -> None:
        self._is_complex = False
        self.dtype = float
        self._ndims = 3
        self._init_residuals(1)
        self._var_types[0] = VarType.VECTOR
        self._num_eqs[0] = 3
        self.resid_names[0] = "displacement"
        self.var_names[0] = "u"
        super().__init__(cast(ResidualFnGR, lambda *args: []))


class _MockModel:
    """Minimal Model stand-in exposing only what FEProblem reads."""

    class _Params:
        values: ClassVar[dict[str, object]] = {}

    parameters = _Params()
    _init_xi: ClassVar[list[JaxArray]] = [jnp.zeros(0)]


def _build_fe_problem(
        mesh: Mesh,
        finite_element: FiniteElement,
        family: ElementFamily,
        assembly_quad: QuadratureRule,
        side_quad: QuadratureRule,
) -> FEProblem:
    """Single-field FEProblem over ``mesh`` (no BCs, mock GR)."""
    layout = GlobalFieldLayout(name="u", finite_element=finite_element)
    dof_map = build_dof_map(
        mesh, [layout], [], components_by_field={"u": 3},
    )
    block_names = list(mesh.element_blocks.keys())
    evaluators: GREvaluators = {"R_and_dR_dU": lambda *args: ([], [])}
    return FEProblem(
        mesh=mesh,
        dof_map=dof_map,
        gr=_DisplacementMockGR(),
        models_by_block=cast(
            dict[str, Model], {b: _MockModel() for b in block_names},
        ),
        modes_by_block={
            b: GlobalResidualMode.CLOSED_FORM for b in block_names
        },
        evaluators_by_block={b: evaluators for b in block_names},
        forcing_fns_by_block_idx=None,
        assembly_quadrature={family: assembly_quad},
        neumann_bcs=(),
        side_quadrature={family: side_quad},
    )


def _build_hex_fe_problem() -> FEProblem:
    return _build_fe_problem(
        StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2)),
        Q1_HEX, ElementFamily.HEX_LINEAR,
        hex_quadrature(degree=2), quad_quadrature(degree=2),
    )


def _build_tet_fe_problem() -> FEProblem:
    return _build_fe_problem(
        hex_to_tet_split(StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))),
        P1_TET, ElementFamily.TET_LINEAR,
        tet_quadrature(degree=1), tri_quadrature(degree=2),
    )


class TestKernelArraysMatchDerivation(unittest.TestCase):
    """build_fe_kernel_arrays index arrays vs the derivation helpers."""

    def _check(self, fe_problem: FEProblem) -> None:
        ka = fe_problem.kernel_arrays
        mesh = fe_problem.mesh
        dof_map = fe_problem.dof_map

        # COO (rows, cols) + dedup scatter: carrier vs assembled_coo_dedup.
        rows, cols, scatter = assembled_coo_dedup(fe_problem)
        np.testing.assert_array_equal(np.asarray(ka.coo_rows), rows)
        np.testing.assert_array_equal(np.asarray(ka.coo_cols), cols)
        np.testing.assert_array_equal(
            np.asarray(ka.coo_dedup_scatter), scatter,
        )

        # geometry cache / embedded sparsity: same objects, not copies.
        self.assertIs(ka.geometry_cache, fe_problem.geometry_cache)
        self.assertIs(ka.embedded_sparsity, fe_problem.embedded_sparsity)
        np.testing.assert_array_equal(
            np.asarray(ka.prescribed_indices),
            np.asarray(dof_map.prescribed_indices),
        )

        for block_name in fe_problem.evaluators_by_block:
            conn = mesh.connectivity[mesh.element_blocks[block_name]]
            n_elems = conn.shape[0]

            # U-gather: the carrier's per-field eq arrays vs the
            # _element_eq_indices derivation, reshaped to the
            # (n_elems, dofs_per_element, dofs_per_basis_fn) layout
            # _gather_element_U indexes the flat global U with.
            u_gather_eqs = ka.u_gather_eq_by_block[block_name]
            self.assertEqual(
                len(u_gather_eqs), len(dof_map.field_layouts),
            )
            for field_idx, eq in enumerate(u_gather_eqs):
                ndofs = int(dof_map.num_dofs_per_basis_fn[field_idx])
                eq_ref = _element_eq_indices(
                    conn, dof_map, field_idx=field_idx,
                ).reshape(n_elems, -1, ndofs)
                np.testing.assert_array_equal(np.asarray(eq), eq_ref)

            # R-scatter: the carrier's eq arrays vs _element_eq_indices.
            r_scatter_eqs = ka.r_scatter_eq_by_block[block_name]
            self.assertEqual(
                len(r_scatter_eqs), fe_problem.gr.num_residuals,
            )
            for r, eq in enumerate(r_scatter_eqs):
                eq_ref = _element_eq_indices(
                    conn, dof_map,
                    field_idx=fe_problem.field_idx_per_block[r],
                )
                np.testing.assert_array_equal(np.asarray(eq), eq_ref)

    def test_hex(self) -> None:
        self._check(_build_hex_fe_problem())

    def test_tet(self) -> None:
        self._check(_build_tet_fe_problem())


if __name__ == "__main__":
    unittest.main()
