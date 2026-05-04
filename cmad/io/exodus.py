"""Exodus II mesh I/O for cmad's FE workflow.

Reads ``.exo`` / ``.g`` mesh files into :class:`cmad.fem.mesh.Mesh` and
writes mesh skeletons via :class:`ExodusWriter`. Implementation uses
:mod:`netCDF4` directly — no SEACAS / ``libexodus`` runtime dep.

Read-side schema mapping
------------------------

- **Coordinates**: combined ``coord`` with shape ``(num_dim, num_nodes)``
  is preferred (older convention; what meshio writes); falls back to
  separate ``coordx`` / ``coordy`` / ``coordz`` vectors.
- **Element blocks**: per-block ``connect{i}`` (1-based node IDs),
  ``eb_prop1`` (block IDs), optional ``eb_names``. Block names default
  to ``f"block_{id}"`` when ``eb_names`` is missing or its slot is empty.
- **Element type**: per-block ``connect{i}.elem_type`` string attribute,
  case-insensitive. Accepted: ``HEX`` / ``HEX8`` -> ``HEX_LINEAR``;
  ``TETRA`` / ``TETRA4`` -> ``TET_LINEAR``. All blocks must share one
  family (cmad ``Mesh`` is single-family).
- **Node sets**: ``node_ns{i}`` (1-based) + ``ns_prop1`` + optional
  ``ns_names``. Names default to ``f"nodeset_{id}"``.
- **Side sets**: ``elem_ss{i}`` + ``side_ss{i}`` (both 1-based) +
  ``ss_prop1`` + optional ``ss_names``. Names default to
  ``f"sideset_{id}"``. Distribution factors are read but not retained.

The reader populates :class:`Mesh.element_block_ids` /
:class:`Mesh.node_set_ids` / :class:`Mesh.side_set_ids` from each
kind's ``*_prop1`` so the original IDs survive a round-trip through
:class:`ExodusWriter`.

Out of scope (raises or ignores; revisit by extending this module):
mixed element families per mesh, higher-order elements, element /
node attributes, ID maps beyond natural ordering, QA records, info
records, global / set variables, parallel I/O.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import netCDF4
import numpy as np
from numpy.typing import NDArray

from cmad.fem.element_family import ElementFamily
from cmad.fem.mesh import Mesh
from cmad.io.results import (
    FieldSpec,
    component_names,
    to_exodus_storage,
)
from cmad.models.var_types import VarType, get_num_eqs

# SEACAS MAX_NAME_LENGTH; buffer size including the null terminator,
# so the longest writable name is 255 chars.
_LEN_STRING = 256

_ELEM_TYPE_TO_FAMILY: dict[str, ElementFamily] = {
    "HEX": ElementFamily.HEX_LINEAR,
    "HEX8": ElementFamily.HEX_LINEAR,
    "TETRA": ElementFamily.TET_LINEAR,
    "TETRA4": ElementFamily.TET_LINEAR,
}

_FAMILY_TO_ELEM_TYPE: dict[ElementFamily, str] = {
    ElementFamily.HEX_LINEAR: "HEX8",
    ElementFamily.TET_LINEAR: "TETRA4",
}


class ExodusFormatError(ValueError):
    """Raised when an Exodus file violates cmad's expected schema."""


def _decode_name_row(byte_row: NDArray[np.bytes_]) -> str:
    return byte_row.tobytes().rstrip(b"\x00").decode("utf-8")


def _decode_names(name_var) -> list[str]:
    raw = name_var[:]
    return [_decode_name_row(raw[i]) for i in range(raw.shape[0])]


def _read_coords(ds: netCDF4.Dataset) -> NDArray[np.float64]:
    """Read nodal coordinates as ``(N_nodes, 3)``.

    Accepts combined ``coord`` (shape ``(num_dim, num_nodes)``) or
    separate ``coordx`` / ``coordy`` / ``coordz``. Raises if neither.
    """
    if "coord" in ds.variables:
        coord = np.asarray(ds["coord"][:])
        if coord.ndim != 2 or coord.shape[0] != 3:
            raise ExodusFormatError(
                f"'coord' must have shape (3, num_nodes); got {coord.shape}"
            )
        return coord.T.astype(np.float64, copy=False)
    if "coordx" in ds.variables:
        cx = np.asarray(ds["coordx"][:])
        cy = np.asarray(ds["coordy"][:])
        cz = np.asarray(ds["coordz"][:])
        return np.column_stack([cx, cy, cz]).astype(np.float64, copy=False)
    raise ExodusFormatError(
        "missing nodal coordinates: expected 'coord' or "
        "'coordx'/'coordy'/'coordz'"
    )


def _read_elem_type(var) -> ElementFamily:
    raw = var.getncattr("elem_type")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    key = raw.strip().upper()
    if key not in _ELEM_TYPE_TO_FAMILY:
        raise ExodusFormatError(
            f"unsupported elem_type {raw!r}; "
            f"supported: {sorted(_ELEM_TYPE_TO_FAMILY)}"
        )
    return _ELEM_TYPE_TO_FAMILY[key]


def _read_blocks(
        ds: netCDF4.Dataset,
        n_blocks: int,
) -> tuple[
    NDArray[np.intp],
    ElementFamily,
    dict[str, NDArray[np.intp]],
    dict[str, int],
]:
    """Read all element blocks. Returns concatenated connectivity (0-based),
    the uniform element family, the per-name index partition, and the
    per-name ``eb_prop1`` IDs.
    """
    if n_blocks < 1:
        raise ExodusFormatError("file has zero element blocks")
    if "eb_prop1" not in ds.variables:
        raise ExodusFormatError("missing 'eb_prop1' (block IDs)")
    prop1 = np.asarray(ds["eb_prop1"][:]).astype(int).tolist()

    # Default names use the original ``eb_prop1`` ID. Safe because the
    # writer preserves IDs through the parallel ``element_block_ids`` dict,
    # so default names and ``eb_prop1`` always agree on round-trip.
    if "eb_names" in ds.variables:
        raw = _decode_names(ds["eb_names"])
        block_names = [
            n if n else f"block_{prop1[i]}" for i, n in enumerate(raw)
        ]
    else:
        block_names = [f"block_{p}" for p in prop1]

    families: list[ElementFamily] = []
    conns: list[NDArray[np.intp]] = []
    sizes: list[int] = []
    for i in range(n_blocks):
        var = ds[f"connect{i + 1}"]
        families.append(_read_elem_type(var))
        conn1 = np.asarray(var[:]).astype(np.intp) - 1   # 1-based -> 0-based
        conns.append(conn1)
        sizes.append(conn1.shape[0])

    family_set = set(families)
    if len(family_set) != 1:
        raise ExodusFormatError(
            "all element blocks must share one element family; got "
            f"{[f.name for f in families]}"
        )
    family = families[0]

    npe_set = {c.shape[1] for c in conns}
    if len(npe_set) != 1:
        raise ExodusFormatError(
            f"all element blocks must have same nodes-per-element; got {npe_set}"
        )

    connectivity = np.concatenate(conns, axis=0).astype(np.intp)

    element_blocks: dict[str, NDArray[np.intp]] = {}
    element_block_ids: dict[str, int] = {}
    start = 0
    for name, sz, p in zip(block_names, sizes, prop1, strict=True):
        element_blocks[name] = np.arange(start, start + sz, dtype=np.intp)
        element_block_ids[name] = p
        start += sz

    return connectivity, family, element_blocks, element_block_ids


def _read_node_sets(
        ds: netCDF4.Dataset,
        n_sets: int,
) -> tuple[dict[str, NDArray[np.intp]], dict[str, int]]:
    if n_sets == 0:
        return {}, {}
    if "ns_prop1" not in ds.variables:
        raise ExodusFormatError("missing 'ns_prop1' (node-set IDs)")
    prop1 = np.asarray(ds["ns_prop1"][:]).astype(int).tolist()

    # Default names use the original ``ns_prop1`` ID; see _read_blocks.
    if "ns_names" in ds.variables:
        raw = _decode_names(ds["ns_names"])
        names = [
            n if n else f"nodeset_{prop1[i]}" for i, n in enumerate(raw)
        ]
    else:
        names = [f"nodeset_{p}" for p in prop1]

    out: dict[str, NDArray[np.intp]] = {}
    ids: dict[str, int] = {}
    for i, (name, p) in enumerate(zip(names, prop1, strict=True)):
        nodes_1based = np.asarray(ds[f"node_ns{i + 1}"][:]).astype(np.intp)
        out[name] = nodes_1based - 1
        ids[name] = p
    return out, ids


def _read_side_sets(
        ds: netCDF4.Dataset,
        n_sets: int,
) -> tuple[dict[str, NDArray[np.intp]], dict[str, int]]:
    if n_sets == 0:
        return {}, {}
    if "ss_prop1" not in ds.variables:
        raise ExodusFormatError("missing 'ss_prop1' (side-set IDs)")
    prop1 = np.asarray(ds["ss_prop1"][:]).astype(int).tolist()

    # Default names use the original ``ss_prop1`` ID; see _read_blocks.
    if "ss_names" in ds.variables:
        raw = _decode_names(ds["ss_names"])
        names = [
            n if n else f"sideset_{prop1[i]}" for i, n in enumerate(raw)
        ]
    else:
        names = [f"sideset_{p}" for p in prop1]

    out: dict[str, NDArray[np.intp]] = {}
    ids: dict[str, int] = {}
    for i, (name, p) in enumerate(zip(names, prop1, strict=True)):
        elems = np.asarray(ds[f"elem_ss{i + 1}"][:]).astype(np.intp) - 1
        sides = np.asarray(ds[f"side_ss{i + 1}"][:]).astype(np.intp) - 1
        out[name] = np.column_stack([elems, sides]).astype(np.intp)
        ids[name] = p
    return out, ids


def read_mesh(path: str | Path) -> Mesh:
    """Read an Exodus II file's mesh skeleton into a :class:`Mesh`.

    Reads coordinates, per-block connectivity, node sets, and side sets.
    Result-variable contents (if any) are ignored. Raises
    :class:`ExodusFormatError` for unsupported variants (mixed element
    families, missing required variables, unknown ``elem_type``).
    """
    path = Path(path)
    with netCDF4.Dataset(str(path), "r") as ds:
        if "num_dim" not in ds.dimensions:
            raise ExodusFormatError("missing dimension 'num_dim'")
        num_dim = len(ds.dimensions["num_dim"])
        if num_dim != 3:
            raise ExodusFormatError(
                f"cmad supports 3D meshes only; got num_dim={num_dim}"
            )

        n_blocks = (
            len(ds.dimensions["num_el_blk"])
            if "num_el_blk" in ds.dimensions else 0
        )
        n_node_sets = (
            len(ds.dimensions["num_node_sets"])
            if "num_node_sets" in ds.dimensions else 0
        )
        n_side_sets = (
            len(ds.dimensions["num_side_sets"])
            if "num_side_sets" in ds.dimensions else 0
        )

        nodes = _read_coords(ds)
        connectivity, family, element_blocks, element_block_ids = (
            _read_blocks(ds, n_blocks)
        )
        node_sets, node_set_ids = _read_node_sets(ds, n_node_sets)
        side_sets, side_set_ids = _read_side_sets(ds, n_side_sets)

    return Mesh(
        nodes=nodes,
        connectivity=connectivity,
        element_family=family,
        element_blocks=element_blocks,
        node_sets=node_sets,
        side_sets=side_sets,
        element_block_ids=element_block_ids,
        node_set_ids=node_set_ids,
        side_set_ids=side_set_ids,
    )


def _encode_names(
        names: Sequence[str],
        len_string: int = _LEN_STRING,
) -> NDArray[np.bytes_]:
    out = np.zeros((len(names), len_string), dtype="S1")
    for i, name in enumerate(names):
        encoded = name.encode("utf-8")
        if len(encoded) >= len_string:
            raise ExodusFormatError(
                f"name {name!r} ({len(encoded)} bytes) does not fit in "
                f"len_string={len_string} (need 1 byte for null terminator)"
            )
        out[i, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
    return out


def _write_metadata(ds: netCDF4.Dataset, title: str) -> None:
    ds.title = title
    # Spec era we actually emit: blocks/sets/named entities, no truth
    # tables / attributes / info records. Bump when those features land.
    ds.version = np.float32(5.1)
    ds.api_version = np.float32(5.1)
    ds.floating_point_word_size = np.int64(8)


def _write_dimensions(ds: netCDF4.Dataset, mesh: Mesh) -> None:
    ds.createDimension("num_dim", 3)
    ds.createDimension("num_nodes", mesh.nodes.shape[0])
    ds.createDimension("num_elem", mesh.connectivity.shape[0])
    ds.createDimension("num_el_blk", len(mesh.element_blocks))
    if mesh.node_sets:
        ds.createDimension("num_node_sets", len(mesh.node_sets))
    if mesh.side_sets:
        ds.createDimension("num_side_sets", len(mesh.side_sets))
    ds.createDimension("len_string", _LEN_STRING)
    ds.createDimension("time_step", None)   # unlimited; populated by write_step


def _write_coordinates(ds: netCDF4.Dataset, mesh: Mesh) -> None:
    # Combined `coord (num_dim, num_nodes)` is required for Paraview /
    # VTK; the older split coordx/coordy/coordz path crashes their reader.
    coord = ds.createVariable("coord", "f8", ("num_dim", "num_nodes"))
    coord[:] = mesh.nodes.T
    coor_names = ds.createVariable(
        "coor_names", "S1", ("num_dim", "len_string")
    )
    coor_names[:] = _encode_names(["x", "y", "z"])
    # Empty time_whole: Paraview / VTK expect this to exist even when no
    # time steps have been written yet.
    ds.createVariable("time_whole", "f8", ("time_step",))


def _write_blocks(ds: netCDF4.Dataset, mesh: Mesh) -> None:
    elem_type = _FAMILY_TO_ELEM_TYPE[mesh.element_family]
    block_names = list(mesh.element_blocks)
    n_blocks = len(block_names)

    eb_prop1 = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
    if mesh.element_block_ids:
        eb_prop1[:] = np.array(
            [mesh.element_block_ids[name] for name in block_names],
            dtype=np.int32,
        )
    else:
        eb_prop1[:] = np.arange(1, n_blocks + 1, dtype=np.int32)

    eb_names = ds.createVariable(
        "eb_names", "S1", ("num_el_blk", "len_string")
    )
    eb_names[:] = _encode_names(block_names)

    npe = mesh.connectivity.shape[1]
    for i, name in enumerate(block_names):
        elem_indices = mesh.element_blocks[name]
        n_in_blk = int(elem_indices.shape[0])
        ds.createDimension(f"num_el_in_blk{i + 1}", n_in_blk)
        ds.createDimension(f"num_nod_per_el{i + 1}", npe)
        connect = ds.createVariable(
            f"connect{i + 1}",
            "i4",
            (f"num_el_in_blk{i + 1}", f"num_nod_per_el{i + 1}"),
        )
        connect.elem_type = elem_type
        connect[:] = (mesh.connectivity[elem_indices] + 1).astype(np.int32)


def _write_node_sets(ds: netCDF4.Dataset, mesh: Mesh) -> None:
    if not mesh.node_sets:
        return
    ns_names = list(mesh.node_sets)
    n_sets = len(ns_names)

    ns_prop1 = ds.createVariable("ns_prop1", "i4", ("num_node_sets",))
    if mesh.node_set_ids:
        ns_prop1[:] = np.array(
            [mesh.node_set_ids[name] for name in ns_names],
            dtype=np.int32,
        )
    else:
        ns_prop1[:] = np.arange(1, n_sets + 1, dtype=np.int32)

    ns_names_var = ds.createVariable(
        "ns_names", "S1", ("num_node_sets", "len_string")
    )
    ns_names_var[:] = _encode_names(ns_names)

    for i, name in enumerate(ns_names):
        nodes = mesh.node_sets[name]
        ds.createDimension(f"num_nod_ns{i + 1}", int(nodes.shape[0]))
        var = ds.createVariable(
            f"node_ns{i + 1}", "i4", (f"num_nod_ns{i + 1}",)
        )
        var[:] = (nodes + 1).astype(np.int32)


def _write_side_sets(ds: netCDF4.Dataset, mesh: Mesh) -> None:
    if not mesh.side_sets:
        return
    ss_names = list(mesh.side_sets)
    n_sets = len(ss_names)

    ss_prop1 = ds.createVariable("ss_prop1", "i4", ("num_side_sets",))
    if mesh.side_set_ids:
        ss_prop1[:] = np.array(
            [mesh.side_set_ids[name] for name in ss_names],
            dtype=np.int32,
        )
    else:
        ss_prop1[:] = np.arange(1, n_sets + 1, dtype=np.int32)

    ss_names_var = ds.createVariable(
        "ss_names", "S1", ("num_side_sets", "len_string")
    )
    ss_names_var[:] = _encode_names(ss_names)

    for i, name in enumerate(ss_names):
        pairs = mesh.side_sets[name]
        n_in_set = int(pairs.shape[0])
        ds.createDimension(f"num_side_ss{i + 1}", n_in_set)
        elem_var = ds.createVariable(
            f"elem_ss{i + 1}", "i4", (f"num_side_ss{i + 1}",)
        )
        side_var = ds.createVariable(
            f"side_ss{i + 1}", "i4", (f"num_side_ss{i + 1}",)
        )
        elem_var[:] = (pairs[:, 0] + 1).astype(np.int32)
        side_var[:] = (pairs[:, 1] + 1).astype(np.int32)


def _validate_nodal_specs(
        nodal_specs: Sequence[FieldSpec], ndims: int,
) -> list[str]:
    """Validate uniqueness + name-length for nodal specs; return the
    flat decorated-name list in disk order."""
    decorated: list[str] = []
    seen_roots: set[str] = set()
    for spec in nodal_specs:
        if spec.name in seen_roots:
            raise ExodusFormatError(
                f"duplicate nodal field name '{spec.name}'"
            )
        seen_roots.add(spec.name)
        for cname in component_names(spec, ndims):
            if len(cname.encode("utf-8")) >= _LEN_STRING:
                raise ExodusFormatError(
                    f"decorated component name {cname!r} does not fit "
                    f"in len_string={_LEN_STRING}"
                )
            decorated.append(cname)
    if len(set(decorated)) != len(decorated):
        raise ExodusFormatError(
            "nodal component names collide across specs: "
            f"{sorted(set(d for d in decorated if decorated.count(d) > 1))}"
        )
    return decorated


def _validate_element_specs(
        mesh: Mesh,
        elem_specs_by_block: dict[str, Sequence[FieldSpec]],
        ndims: int,
) -> tuple[list[str], dict[str, VarType], list[str]]:
    """Validate element-spec inputs; return the flat decorated-name
    list (disk order), the root -> VarType map, and the union root-name
    list (disk-insertion order)."""
    for block_name in elem_specs_by_block:
        if block_name not in mesh.element_blocks:
            raise ExodusFormatError(
                f"element_field_specs key '{block_name}' is not in "
                f"mesh.element_blocks ({sorted(mesh.element_blocks)})"
            )
    root_var_types: dict[str, VarType] = {}
    root_order: list[str] = []
    for block_name, specs in elem_specs_by_block.items():
        seen_in_block: set[str] = set()
        for spec in specs:
            if spec.name in seen_in_block:
                raise ExodusFormatError(
                    f"duplicate element field name '{spec.name}' in "
                    f"block '{block_name}'"
                )
            seen_in_block.add(spec.name)
            if spec.name in root_var_types:
                if root_var_types[spec.name] != spec.var_type:
                    raise ExodusFormatError(
                        f"element field '{spec.name}' has VarType "
                        f"{spec.var_type.name} on block '{block_name}' "
                        f"but {root_var_types[spec.name].name} on an "
                        f"earlier block; var_types must agree across "
                        f"blocks for the same name"
                    )
            else:
                root_var_types[spec.name] = spec.var_type
                root_order.append(spec.name)
    decorated: list[str] = []
    for root in root_order:
        synthetic = FieldSpec(name=root, var_type=root_var_types[root])
        for cname in component_names(synthetic, ndims):
            if len(cname.encode("utf-8")) >= _LEN_STRING:
                raise ExodusFormatError(
                    f"decorated component name {cname!r} does not fit "
                    f"in len_string={_LEN_STRING}"
                )
            decorated.append(cname)
    if len(set(decorated)) != len(decorated):
        raise ExodusFormatError(
            "element component names collide across specs: "
            f"{sorted(set(d for d in decorated if decorated.count(d) > 1))}"
        )
    return decorated, root_var_types, root_order


def _write_nodal_var_schema(
        ds: netCDF4.Dataset,
        nodal_specs: Sequence[FieldSpec],
        ndims: int,
) -> dict[str, list[int]]:
    """Create dimensions / variables for nodal result variables.

    Returns ``var_indices``: spec name -> list of 1-based disk-side
    component indices in disk (Exodus) order.
    """
    if not nodal_specs:
        return {}
    var_indices: dict[str, list[int]] = {}
    next_idx = 1
    for spec in nodal_specs:
        n_comp = get_num_eqs(spec.var_type, ndims)
        var_indices[spec.name] = list(range(next_idx, next_idx + n_comp))
        next_idx += n_comp
    n_components = next_idx - 1
    decorated = _validate_nodal_specs(nodal_specs, ndims)

    ds.createDimension("num_nod_var", n_components)
    name_nod_var = ds.createVariable(
        "name_nod_var", "S1", ("num_nod_var", "len_string"),
    )
    name_nod_var[:] = _encode_names(decorated)

    for n in range(1, n_components + 1):
        ds.createVariable(
            f"vals_nod_var{n}", "f8", ("time_step", "num_nodes"),
        )
    return var_indices


def _write_element_var_schema(
        ds: netCDF4.Dataset,
        mesh: Mesh,
        elem_specs_by_block: dict[str, Sequence[FieldSpec]],
        ndims: int,
) -> tuple[dict[str, dict[str, list[int]]], dict[str, int]]:
    """Create dimensions / variables for element result variables.

    Returns ``(var_indices_by_block, block_idx_by_name)`` where the
    inner ``list[int]`` is 1-based disk-side component indices in
    disk order, and ``block_idx_by_name`` is the 1-based ``eb{B}``
    index per element-block name.
    """
    block_names_in_mesh = list(mesh.element_blocks)
    block_idx_by_name = {
        name: i + 1 for i, name in enumerate(block_names_in_mesh)
    }
    if not any(specs for specs in elem_specs_by_block.values()):
        return {}, block_idx_by_name

    decorated, root_var_types, root_order = _validate_element_specs(
        mesh, elem_specs_by_block, ndims,
    )

    # Map each root name to its disk-side component-index list.
    root_to_indices: dict[str, list[int]] = {}
    next_idx = 1
    for root in root_order:
        n_comp = get_num_eqs(root_var_types[root], ndims)
        root_to_indices[root] = list(range(next_idx, next_idx + n_comp))
        next_idx += n_comp
    n_components = len(decorated)
    n_blocks = len(block_names_in_mesh)

    ds.createDimension("num_elem_var", n_components)
    name_elem_var = ds.createVariable(
        "name_elem_var", "S1", ("num_elem_var", "len_string"),
    )
    name_elem_var[:] = _encode_names(decorated)

    truth = np.zeros((n_blocks, n_components), dtype=np.int32)
    for block_name, specs in elem_specs_by_block.items():
        b = block_idx_by_name[block_name] - 1
        for spec in specs:
            for n in root_to_indices[spec.name]:
                truth[b, n - 1] = 1
    elem_var_tab = ds.createVariable(
        "elem_var_tab", "i4", ("num_el_blk", "num_elem_var"),
    )
    elem_var_tab[:] = truth

    var_indices_by_block: dict[str, dict[str, list[int]]] = {}
    for block_name, specs in elem_specs_by_block.items():
        block_var_indices: dict[str, list[int]] = {}
        b = block_idx_by_name[block_name]
        for spec in specs:
            indices = root_to_indices[spec.name]
            block_var_indices[spec.name] = indices
            for n in indices:
                ds.createVariable(
                    f"vals_elem_var{n}eb{b}", "f8",
                    ("time_step", f"num_el_in_blk{b}"),
                )
        var_indices_by_block[block_name] = block_var_indices
    return var_indices_by_block, block_idx_by_name


class ExodusWriter:
    """Writes an Exodus II file's mesh skeleton + time-stepped results.

    On construction, opens ``path`` and writes coordinates, per-block
    connectivity, node sets, and side sets in NETCDF4 format. Block /
    nodeset / sideset IDs come from ``mesh.element_block_ids`` /
    ``node_set_ids`` / ``side_set_ids`` if non-empty, otherwise are
    assigned sequentially starting at 1 in
    ``mesh.element_blocks`` / ``node_sets`` / ``side_sets`` insertion
    order. Element type strings emitted: ``HEX8`` for HEX_LINEAR,
    ``TETRA4`` for TET_LINEAR.

    ``nodal_field_specs`` and ``element_field_specs`` declare the
    result-variable schema at construction (Exodus's variable-name
    list is fixed once the file is opened). Each :class:`FieldSpec`
    carries ``(name, var_type)``; the writer derives Exodus-side
    decorated component names via
    :func:`cmad.io.results.component_names` and lays out
    ``vals_nod_var{N}`` / ``vals_elem_var{N}eb{B}`` arrays empty
    along the unlimited time axis. ``element_field_specs`` keys are
    element-block names. The same root name on different blocks must
    carry the same VarType; the truth table ``elem_var_tab`` records
    which (block, variable) pairs are populated.

    Each :meth:`write_step` call appends one time row across all
    declared variables. Internally cmad's sym-tensor vec order is
    ``[xx, xy, xz, yy, yz, zz]``; the writer permutes to the Exodus
    disk order ``[xx, yy, zz, xy, xz, yz]`` before emitting.

    Caller must call :meth:`close` (or use as a context manager).
    """

    def __init__(
            self,
            path: str | Path,
            mesh: Mesh,
            title: str = "",
            *,
            nodal_field_specs: Sequence[FieldSpec] = (),
            element_field_specs: dict[
                str, Sequence[FieldSpec]
            ] | None = None,
    ) -> None:
        self._path = Path(path)
        self._mesh = mesh
        self._ds: netCDF4.Dataset | None = None

        if element_field_specs is None:
            element_field_specs = {}
        self._nodal_specs: dict[str, FieldSpec] = {
            s.name: s for s in nodal_field_specs
        }
        self._elem_specs: dict[str, dict[str, FieldSpec]] = {
            block: {s.name: s for s in specs}
            for block, specs in element_field_specs.items()
        }
        ndims = int(mesh.nodes.shape[1])
        self._ndims = ndims
        self._step_count = 0

        ds = netCDF4.Dataset(str(self._path), "w", format="NETCDF4")
        try:
            _write_metadata(ds, title)
            _write_dimensions(ds, mesh)
            _write_coordinates(ds, mesh)
            _write_blocks(ds, mesh)
            _write_node_sets(ds, mesh)
            _write_side_sets(ds, mesh)
            self._nodal_var_indices = _write_nodal_var_schema(
                ds, list(nodal_field_specs), ndims,
            )
            (
                self._elem_var_indices,
                self._elem_block_idx,
            ) = _write_element_var_schema(
                ds, mesh, element_field_specs, ndims,
            )
        except Exception:
            ds.close()
            raise
        self._ds = ds

    def write_step(
            self,
            time: float,
            nodal_data: dict[
                str, NDArray[np.floating]
            ] | None = None,
            element_data: dict[
                str, dict[str, NDArray[np.floating]]
            ] | None = None,
    ) -> None:
        """Append one time step of result-variable data.

        ``nodal_data[name]`` shape ``(n_nodes, *components)`` matches
        the spec's VarType (SCALAR accepts both ``(n_nodes,)`` and
        ``(n_nodes, 1)``). The trailing component axis is in cmad's
        internal order; the writer permutes to the Exodus disk order
        for SYM_TENSOR. ``element_data[block][name]`` is the per-block
        analogue with shape ``(n_elems_in_block, *components)``.

        Each call writes exactly the declared specs; missing or extra
        keys raise. Constructing the writer with no specs and calling
        ``write_step`` raises — mesh-only files write zero time rows.

        The writer does not auto-emit an initial-condition step:
        ``n_steps_in_file == number_of_write_step_calls``. Caller is
        responsible for writing step 0 explicitly when wanted (e.g.
        for Paraview-consistent animations).

        Caller is responsible for aligning nodal arrays to
        ``mesh.nodes``. For a Q1-on-Q1 mesh this is trivial; for a
        non-vertex basis (subparametric / Q2) the caller projects
        from basis coefficients to mesh nodes upstream.
        """
        if self._ds is None:
            raise ValueError("ExodusWriter is closed")
        if not self._nodal_specs and not self._elem_specs:
            raise ValueError(
                "ExodusWriter constructed with no result-variable "
                "specs; no time-stepped data to write"
            )

        nodal_data = nodal_data or {}
        element_data = element_data or {}

        if set(nodal_data.keys()) != set(self._nodal_specs.keys()):
            raise ValueError(
                f"nodal_data keys {sorted(nodal_data.keys())} do not "
                f"match declared nodal specs "
                f"{sorted(self._nodal_specs.keys())}"
            )
        if set(element_data.keys()) != set(self._elem_specs.keys()):
            raise ValueError(
                f"element_data block keys {sorted(element_data.keys())}"
                f" do not match declared element-spec blocks "
                f"{sorted(self._elem_specs.keys())}"
            )
        for block_name, declared in self._elem_specs.items():
            if set(element_data[block_name].keys()) != set(declared):
                raise ValueError(
                    f"element_data[{block_name!r}] keys "
                    f"{sorted(element_data[block_name].keys())} do not "
                    f"match declared specs {sorted(declared)}"
                )

        n_nodes = self._mesh.nodes.shape[0]
        ndims = self._ndims

        def _canonicalize(
                values, n_rows: int, var_type: VarType, what: str,
        ) -> NDArray[np.floating]:
            arr = np.asarray(values, dtype=np.float64)
            n_comp = get_num_eqs(var_type, ndims)
            if var_type == VarType.SCALAR:
                if arr.shape == (n_rows,):
                    arr = arr[:, None]
                if arr.shape != (n_rows, 1):
                    raise ValueError(
                        f"{what}: SCALAR expects ({n_rows},) or "
                        f"({n_rows}, 1); got {arr.shape}"
                    )
            else:
                if arr.shape != (n_rows, n_comp):
                    raise ValueError(
                        f"{what}: {var_type.name} expects "
                        f"({n_rows}, {n_comp}); got {arr.shape}"
                    )
            return arr

        # Stage all canonicalized + permuted arrays first so any shape
        # error raises before we touch the file.
        staged_nodal: dict[str, NDArray[np.floating]] = {}
        for name, raw in nodal_data.items():
            spec = self._nodal_specs[name]
            arr = _canonicalize(raw, n_nodes, spec.var_type, f"nodal[{name!r}]")
            staged_nodal[name] = np.asarray(
                to_exodus_storage(arr, spec.var_type),
            )
        staged_elem: dict[str, dict[str, NDArray[np.floating]]] = {}
        for block_name, declared in self._elem_specs.items():
            n_elem = int(self._mesh.element_blocks[block_name].shape[0])
            block_staged: dict[str, NDArray[np.floating]] = {}
            for name in declared:
                spec = declared[name]
                arr = _canonicalize(
                    element_data[block_name][name],
                    n_elem, spec.var_type,
                    f"element[{block_name!r}][{name!r}]",
                )
                block_staged[name] = np.asarray(
                    to_exodus_storage(arr, spec.var_type),
                )
            staged_elem[block_name] = block_staged

        step_idx = self._step_count
        self._ds["time_whole"][step_idx] = float(time)
        for name, perm in staged_nodal.items():
            for c, n in enumerate(self._nodal_var_indices[name]):
                self._ds[f"vals_nod_var{n}"][step_idx, :] = perm[:, c]
        for block_name, block_staged in staged_elem.items():
            b = self._elem_block_idx[block_name]
            for name, perm in block_staged.items():
                for c, n in enumerate(
                    self._elem_var_indices[block_name][name],
                ):
                    self._ds[f"vals_elem_var{n}eb{b}"][step_idx, :] = (
                        perm[:, c]
                    )
        self._step_count += 1

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __enter__(self) -> "ExodusWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
