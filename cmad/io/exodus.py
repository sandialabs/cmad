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
]:
    """Read all element blocks. Returns concatenated connectivity (0-based),
    the uniform element family, and the per-name index partition.
    """
    if n_blocks < 1:
        raise ExodusFormatError("file has zero element blocks")
    if "eb_prop1" not in ds.variables:
        raise ExodusFormatError("missing 'eb_prop1' (block IDs)")

    block_ids = np.asarray(ds["eb_prop1"][:]).astype(int)
    if "eb_names" in ds.variables:
        raw = _decode_names(ds["eb_names"])
        block_names = [
            n if n else f"block_{block_ids[i]}" for i, n in enumerate(raw)
        ]
    else:
        block_names = [f"block_{bid}" for bid in block_ids]

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
    start = 0
    for name, sz in zip(block_names, sizes, strict=True):
        element_blocks[name] = np.arange(start, start + sz, dtype=np.intp)
        start += sz

    return connectivity, family, element_blocks


def _read_node_sets(
        ds: netCDF4.Dataset,
        n_sets: int,
) -> dict[str, NDArray[np.intp]]:
    if n_sets == 0:
        return {}
    ids = np.asarray(ds["ns_prop1"][:]).astype(int)
    if "ns_names" in ds.variables:
        raw = _decode_names(ds["ns_names"])
        names = [
            n if n else f"nodeset_{ids[i]}" for i, n in enumerate(raw)
        ]
    else:
        names = [f"nodeset_{sid}" for sid in ids]

    out: dict[str, NDArray[np.intp]] = {}
    for i, name in enumerate(names):
        nodes_1based = np.asarray(ds[f"node_ns{i + 1}"][:]).astype(np.intp)
        out[name] = nodes_1based - 1
    return out


def _read_side_sets(
        ds: netCDF4.Dataset,
        n_sets: int,
) -> dict[str, NDArray[np.intp]]:
    if n_sets == 0:
        return {}
    ids = np.asarray(ds["ss_prop1"][:]).astype(int)
    if "ss_names" in ds.variables:
        raw = _decode_names(ds["ss_names"])
        names = [
            n if n else f"sideset_{ids[i]}" for i, n in enumerate(raw)
        ]
    else:
        names = [f"sideset_{sid}" for sid in ids]

    out: dict[str, NDArray[np.intp]] = {}
    for i, name in enumerate(names):
        elems = np.asarray(ds[f"elem_ss{i + 1}"][:]).astype(np.intp) - 1
        sides = np.asarray(ds[f"side_ss{i + 1}"][:]).astype(np.intp) - 1
        out[name] = np.column_stack([elems, sides]).astype(np.intp)
    return out


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
        connectivity, family, element_blocks = _read_blocks(ds, n_blocks)
        node_sets = _read_node_sets(ds, n_node_sets)
        side_sets = _read_side_sets(ds, n_side_sets)

    return Mesh(
        nodes=nodes,
        connectivity=connectivity,
        element_family=family,
        element_blocks=element_blocks,
        node_sets=node_sets,
        side_sets=side_sets,
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


class ExodusWriter:
    """Writes an Exodus II file's mesh skeleton.

    On construction, opens ``path`` and writes coordinates, per-block
    connectivity, node sets, and side sets in NETCDF4 format. Block /
    nodeset / sideset IDs are assigned sequentially starting at 1 in
    ``mesh.element_blocks`` / ``node_sets`` / ``side_sets`` insertion
    order. Element type strings emitted: ``HEX8`` for HEX_LINEAR,
    ``TETRA4`` for TET_LINEAR.

    Caller must call :meth:`close` (or use as a context manager).
    Time-stepped result variables are not yet supported.
    """

    def __init__(
            self,
            path: str | Path,
            mesh: Mesh,
            title: str = "",
    ) -> None:
        self._path = Path(path)
        self._mesh = mesh
        self._ds: netCDF4.Dataset | None = None
        ds = netCDF4.Dataset(str(self._path), "w", format="NETCDF4")
        try:
            _write_metadata(ds, title)
            _write_dimensions(ds, mesh)
            _write_coordinates(ds, mesh)
            _write_blocks(ds, mesh)
            _write_node_sets(ds, mesh)
            _write_side_sets(ds, mesh)
        except Exception:
            ds.close()
            raise
        self._ds = ds

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __enter__(self) -> "ExodusWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
