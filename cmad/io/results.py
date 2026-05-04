"""Result-variable types and helpers for time-stepped Exodus I/O.

VarType-aware component naming and storage permutation between cmad's
internal sym-tensor vec order ``[xx, xy, xz, yy, yz, zz]`` and the
Exodus / Paraview convention ``[xx, yy, zz, xy, xz, yz]``. SCALAR /
VECTOR / TENSOR pass through; only SYM_TENSOR is permuted.

IP→element and global-field→element volume-weighted reductions live
here too; both rely on the per-element-block geometry cache populated
by :func:`cmad.fem.precompute.precompute_block_geometry` at FEProblem
build time.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.precompute import (
    BlockIPGeometryCache,
    compute_ip_quadrature_weights,
)
from cmad.models.var_types import VarType
from cmad.typing import JaxArray

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem


@dataclass(frozen=True)
class FieldSpec:
    """Declares a result field for Exodus I/O.

    ``name`` is the cmad-side root name (``"displacement"``,
    ``"cauchy"``, etc.); the writer derives Exodus-side decorated
    component names via :func:`component_names`. ``var_type`` controls
    component count and naming (SCALAR -> bare; VECTOR -> _x/_y/_z;
    SYM_TENSOR -> Exodus order _xx/_yy/_zz/_xy/_xz/_yz; TENSOR ->
    row-major _ij). Whether a spec is interpreted as nodal or per-
    element data is determined by which kwarg slot it is passed under,
    not by an attribute on the spec.
    """
    name: str
    var_type: VarType


@dataclass(frozen=True)
class ExodusResults:
    """Time-stepped result variables read from an Exodus file.

    ``time`` shape ``(n_steps,)``. ``nodal[name]`` shape
    ``(n_steps, n_nodes, *components)``;
    ``element[block][name]`` shape
    ``(n_steps, n_elems_in_block, *components)``. Component axes are
    in cmad-internal order — SYM_TENSOR has been un-permuted from
    disk, so callers see ``[xx, xy, xz, yy, yz, zz]``.
    """
    time: NDArray[np.floating]
    nodal: dict[str, NDArray[np.floating]]
    element: dict[str, dict[str, NDArray[np.floating]]]


# Exodus / Paraview convention for sym-tensor disk order.
# cmad's internal flat order is [xx, xy, xz, yy, yz, zz] (3D),
# [xx, xy, yy] (2D), [xx] (1D).
_SYM_INTERNAL_TO_EXODUS: dict[int, tuple[int, ...]] = {
    1: (0,),
    3: (0, 2, 1),
    6: (0, 3, 5, 1, 2, 4),
}
_SYM_EXODUS_TO_INTERNAL: dict[int, tuple[int, ...]] = {
    1: (0,),
    3: (0, 2, 1),
    6: (0, 3, 4, 1, 5, 2),
}

_VECTOR_SUFFIXES: dict[int, tuple[str, ...]] = {
    1: ("_x",),
    2: ("_x", "_y"),
    3: ("_x", "_y", "_z"),
}
_SYM_TENSOR_SUFFIXES_EXODUS: dict[int, tuple[str, ...]] = {
    1: ("_xx",),
    2: ("_xx", "_yy", "_xy"),
    3: ("_xx", "_yy", "_zz", "_xy", "_xz", "_yz"),
}
_TENSOR_SUFFIXES: dict[int, tuple[str, ...]] = {
    1: ("_xx",),
    2: ("_xx", "_xy", "_yx", "_yy"),
    3: (
        "_xx", "_xy", "_xz",
        "_yx", "_yy", "_yz",
        "_zx", "_zy", "_zz",
    ),
}


def component_names(spec: FieldSpec, ndims: int) -> tuple[str, ...]:
    """Exodus-side decorated component names, in disk order.

    SCALAR returns ``(name,)``; VECTOR appends ``_x``/``_y``/``_z``
    truncated to ``ndims``; SYM_TENSOR uses Exodus order
    ``_xx``/``_yy``/``_zz``/``_xy``/``_xz``/``_yz`` (cmad's internal
    flat order is ``[xx, xy, xz, yy, yz, zz]``; the writer applies a
    permutation when laying components out on disk); TENSOR appends 9
    indices in row-major ``_ij``.

    Component order in the returned tuple is the disk order — for
    SYM_TENSOR specifically, this is *not* the order of the internal
    flat vector.
    """
    if spec.var_type == VarType.SCALAR:
        return (spec.name,)
    if spec.var_type == VarType.VECTOR:
        suffixes = _VECTOR_SUFFIXES[ndims]
    elif spec.var_type == VarType.SYM_TENSOR:
        suffixes = _SYM_TENSOR_SUFFIXES_EXODUS[ndims]
    elif spec.var_type == VarType.TENSOR:
        suffixes = _TENSOR_SUFFIXES[ndims]
    else:
        raise ValueError(f"Unknown var_type: {spec.var_type}")
    return tuple(spec.name + s for s in suffixes)


def to_exodus_storage(
        values: NDArray[np.floating] | JaxArray,
        var_type: VarType,
) -> NDArray[np.floating] | JaxArray:
    """Permute the trailing component axis from internal -> Exodus.

    SCALAR / VECTOR / TENSOR pass through unchanged (TENSOR's row-
    major ``_ij`` matches both representations). SYM_TENSOR permutes
    the last axis from cmad's internal order to Exodus order:
    3D: ``[xx, xy, xz, yy, yz, zz] -> [xx, yy, zz, xy, xz, yz]``;
    2D: ``[xx, xy, yy] -> [xx, yy, xy]``;
    1D: ``[xx] -> [xx]``.
    """
    if var_type != VarType.SYM_TENSOR:
        return values
    n_comp = values.shape[-1]
    if n_comp not in _SYM_INTERNAL_TO_EXODUS:
        raise ValueError(
            f"SYM_TENSOR component count {n_comp} not in "
            f"{sorted(_SYM_INTERNAL_TO_EXODUS)}"
        )
    perm = list(_SYM_INTERNAL_TO_EXODUS[n_comp])
    if isinstance(values, np.ndarray):
        return values[..., perm]
    return jnp.asarray(values)[..., jnp.asarray(perm)]


def from_exodus_storage(
        values: NDArray[np.floating] | JaxArray,
        var_type: VarType,
) -> NDArray[np.floating] | JaxArray:
    """Inverse of :func:`to_exodus_storage`."""
    if var_type != VarType.SYM_TENSOR:
        return values
    n_comp = values.shape[-1]
    if n_comp not in _SYM_EXODUS_TO_INTERNAL:
        raise ValueError(
            f"SYM_TENSOR component count {n_comp} not in "
            f"{sorted(_SYM_EXODUS_TO_INTERNAL)}"
        )
    perm = list(_SYM_EXODUS_TO_INTERNAL[n_comp])
    if isinstance(values, np.ndarray):
        return values[..., perm]
    return jnp.asarray(values)[..., jnp.asarray(perm)]


def ip_average_to_element(
        values_per_ip: NDArray[np.floating] | JaxArray,
        geometry_cache: dict[str, BlockIPGeometryCache],
        block_name: str,
) -> NDArray[np.floating]:
    """Volume-weighted IP -> element reduction.

    ``values_per_ip`` shape ``(n_elems, n_ip, *components)``; returns
    ``(n_elems, *components)`` -- the integration-measure-weighted
    mean ``sum_p (det J · w · v) / sum_p (det J · w)``. Component
    axes are preserved unchanged. Pulls the per-element-IP measure
    from :func:`cmad.fem.precompute.compute_ip_quadrature_weights`.
    """
    weights = compute_ip_quadrature_weights(geometry_cache)[block_name]
    values = np.asarray(values_per_ip)
    if values.shape[:2] != weights.shape:
        raise ValueError(
            f"values_per_ip leading shape {values.shape[:2]} does not "
            f"match cached weights shape {weights.shape} for block "
            f"'{block_name}'"
        )
    while weights.ndim < values.ndim:
        weights = weights[..., None]
    weighted = values * weights
    numerator = weighted.sum(axis=1)
    denominator = weights.sum(axis=1)
    return numerator / denominator


def volume_average_global_field(
        U_global: NDArray[np.floating] | JaxArray,
        fe_problem: "FEProblem",
        block_name: str,
        field_name: str,
) -> NDArray[np.floating]:
    """Per-element volume-integral average of a global FE field.

    Resolves ``field_name`` against ``fe_problem.gr.var_names`` to
    locate the residual block carrying the field, evaluates that
    block's reference shape values (cached at
    ``geometry_cache[block_name].shared.field_N_per_block[r]``)
    against the per-element gathered ``U`` slice for the
    corresponding dof-map field, then chains through
    :func:`ip_average_to_element` for the integration-measure-
    weighted mean.

    Returns ``(n_elems_in_block, n_components)`` where
    ``n_components = get_num_eqs(var_type, ndims)`` for the field.

    For Q1 vertex-anchored fields on undistorted hexes this matches
    the simple vertex average; on distorted hexes (and for higher-
    order or non-vertex bases when they land) the volume integral
    and the simple average diverge — this helper does the integral.
    """
    from cmad.fem.assembly import _gather_element_U

    var_names = fe_problem.gr.var_names
    matches = [r for r, name in enumerate(var_names) if name == field_name]
    if not matches:
        raise ValueError(
            f"field '{field_name}' not bound to any residual block "
            f"in gr.var_names ({list(var_names)})"
        )
    if len(matches) > 1:
        raise ValueError(
            f"field '{field_name}' bound to multiple residual blocks "
            f"{matches}; ambiguous"
        )
    r = matches[0]

    mesh = fe_problem.mesh
    dof_map = fe_problem.dof_map
    elem_indices = mesh.element_blocks[block_name]
    connectivity_block = mesh.connectivity[elem_indices]

    field_idx = fe_problem.field_idx_per_block[r]
    U_per_field = _gather_element_U(U_global, dof_map, connectivity_block)
    U_elem_field = U_per_field[field_idx]

    block_cache = fe_problem.geometry_cache[block_name]
    field_N = block_cache.shared.field_N_per_block[r]

    U_at_ips = jnp.einsum("pa,eak->epk", field_N, U_elem_field)
    return ip_average_to_element(
        U_at_ips, fe_problem.geometry_cache, block_name,
    )
