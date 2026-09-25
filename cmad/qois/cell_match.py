"""Squared field mismatch integrated over the mesh's cells, or a region of
them; in 2D the cells are the measured surface itself.

Shared by the QoIs that integrate the square of a field over cells:
:func:`cell_arrays_and_measure` evaluates the field's shape values and
the weighted Jacobian determinants at the points of a rule exact for the
square of a linear field, whatever rule the assembly uses, and gives the
region's measure; :func:`cell_squared_mismatch` gathers the field on
each cell, interpolates the mismatch to those points, and integrates its
square.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.assembly import _gather_element_U
from cmad.fem.element_family import ElementFamily
from cmad.fem.precompute import precompute_block_geometry
from cmad.fem.quadrature import (
    QuadratureRule,
    hex_quadrature,
    quad_quadrature,
    tet_quadrature,
    tri_quadrature,
)
from cmad.fem.sharding import pad_element_leaves, shard_element_leaves
from cmad.qois.fe_match_term import SquaredMismatch
from cmad.typing import JaxArray

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem
    from cmad.fem.kernel_arrays import FEKernelArrays

# Exact for the square of a linear field.
_SQUARED_FIELD_QUADRATURE: dict[ElementFamily, QuadratureRule] = {
    ElementFamily.HEX_LINEAR: hex_quadrature(degree=2),
    ElementFamily.TET_LINEAR: tet_quadrature(degree=2),
    ElementFamily.QUAD_LINEAR: quad_quadrature(degree=2),
    ElementFamily.TRI_LINEAR: tri_quadrature(degree=2),
}


@dataclass(frozen=True)
class CellIntegrationArrays:
    """One element block's arrays for integrating the square of a field:
    ``N``, ``(n_ip, n_basis_fns)``, the field's shape values at the
    integration points, and ``weighted_iso_jac_det``,
    ``(n_elems_padded, n_ip)``, the Jacobian determinant times the
    quadrature weight, zero on the cells outside the region and on the
    padding, sharded as the kernel arrays are."""

    N: JaxArray
    weighted_iso_jac_det: JaxArray


def cell_arrays_and_measure(
        fe_problem: FEProblem,
        field_name: str,
        roi: NDArray[np.intp] | None,
) -> tuple[dict[str, CellIntegrationArrays], float]:
    """The arrays for ``field_name`` by element block and the measure of
    the region ``roi`` (global element indices, the whole mesh when
    ``None``)."""
    r = list(fe_problem.gr.var_names).index(field_name)
    cache = precompute_block_geometry(
        fe_problem.mesh, _SQUARED_FIELD_QUADRATURE,
        fe_problem.field_layouts_per_block, fe_problem.thickness,
    )
    keep = np.ones(fe_problem.mesh.connectivity.shape[0], dtype=bool)
    if roi is not None:
        keep[:] = False
        keep[roi] = True

    arrays: dict[str, CellIntegrationArrays] = {}
    measure = 0.0
    for block, elems in fe_problem.mesh.element_blocks.items():
        block_cache = cache[block]
        weighted = (
            block_cache.per_elem.iso_jac_det
            * block_cache.shared.quad_w[None, :]
            * jnp.asarray(keep[elems], dtype=jnp.float64)[:, None]
        )
        measure += float(jnp.sum(weighted))
        weighted = pad_element_leaves(
            weighted, fe_problem.n_elems_padded_by_block[block], zero=True,
        )
        if fe_problem.device_mesh is not None:
            weighted = shard_element_leaves(weighted, fe_problem.device_mesh)
        arrays[block] = CellIntegrationArrays(
            N=block_cache.shared.field_N_per_block[r],
            weighted_iso_jac_det=weighted,
        )
    if measure <= 0.0:
        raise ValueError(
            f"the region of interest for field '{field_name}' selects no "
            "elements, so the mismatch has nothing to average over"
        )
    return arrays, measure


def cell_squared_mismatch(
        arrays: dict[str, CellIntegrationArrays],
        field_idx: int,
        fe_arrays: FEKernelArrays,
        data_flat: JaxArray | None = None,
) -> SquaredMismatch:
    """``mismatch(U, step)``: the squared difference between field
    ``field_idx`` of ``U`` and the data at match time ``step``, integrated
    over the region; the square of the field itself without
    ``data_flat``, ``step`` then unused."""
    def _mismatch(U: JaxArray, step: int | JaxArray) -> JaxArray:
        U_data = None if data_flat is None else data_flat[step]
        total = jnp.zeros(())
        for block, block_arrays in arrays.items():
            diff = _gather_element_U(U, fe_arrays, block)[field_idx]
            if U_data is not None:
                diff = diff - _gather_element_U(
                    U_data, fe_arrays, block,
                )[field_idx]
            diff_at_ip = jnp.einsum("pa,eak->epk", block_arrays.N, diff)
            diff_sq = jnp.sum(diff_at_ip * diff_at_ip, axis=-1)
            total = total + jnp.sum(
                diff_sq * block_arrays.weighted_iso_jac_det,
            )
        return total

    return _mismatch
