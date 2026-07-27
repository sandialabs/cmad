"""gmsh mesh input for cmad's FE workflow.

Reads a gmsh ``.msh`` file into :class:`cmad.fem.mesh.Mesh` through the gmsh
Python API. Parallel to :func:`cmad.io.exodus.read_mesh`; a deck's
``discretization.mesh file`` accepts either format via the suffix dispatch in
:func:`cmad.io.mesh_io.read_mesh_file`.

What the reader maps:

- **Nodes**: ``gmsh.model.mesh.getNodes`` coordinates. gmsh returns three
  coordinates per node; a 2D mesh (which lies in a plane of constant z) keeps
  the x and y columns, a 3D mesh all three. gmsh node tags (any positive
  integers) remap to 0-based contiguous indices.
- **Elements**: the top element dimension present (3 if the mesh has volume
  elements, else 2), one family per mesh (raises on mixed). gmsh element type
  2 (3-node triangle) is ``ElementFamily.TRI_LINEAR``; type 3 (4-node
  quadrilateral) is ``QUAD_LINEAR``; type 4 (4-node tetrahedron) is
  ``TET_LINEAR``; type 5 (8-node hexahedron) is ``HEX_LINEAR``.
- **Element blocks**: each physical group of the mesh dimension becomes one
  block (name from the group name, id from the group tag into
  ``element_block_ids``). With no such physical groups, a single ``"all"``
  block holds every element.

Boundary conditions attach through the bounding box side sets that
:func:`cmad.fem.mesh.coordinate_side_sets` builds at deck-load (the deck's
``build coordinate sidesets`` option), so ``node_sets`` and ``side_sets`` come
back empty.
"""
from __future__ import annotations

from pathlib import Path

import gmsh
import numpy as np
from numpy.typing import NDArray

from cmad.fem.element_family import ElementFamily
from cmad.fem.mesh import Mesh

_GMSH_TYPE_TO_FAMILY: dict[int, ElementFamily] = {
    2: ElementFamily.TRI_LINEAR,
    3: ElementFamily.QUAD_LINEAR,
    4: ElementFamily.TET_LINEAR,
    5: ElementFamily.HEX_LINEAR,
}


class GmshFormatError(ValueError):
    """Raised when a gmsh mesh violates cmad's expected schema."""


def read_gmsh_mesh(path: str | Path) -> Mesh:
    """Read a gmsh ``.msh`` file into a :class:`Mesh` (module docstring for the
    node / element / block mapping).

    Raises :class:`GmshFormatError` for no nodes, no 2D or 3D elements, mixed
    or unknown element families, a 2D mesh that varies in z, or physical
    groups that do not partition the elements.
    """
    path = Path(path)
    if not path.is_file():
        raise GmshFormatError(f"gmsh mesh file not found: {path}")

    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        dim = _mesh_dimension()
        nodes, index_by_tag = _read_nodes(dim)
        connectivity, family, elem_tags = _read_elements(dim, index_by_tag)
        element_blocks, element_block_ids = _read_blocks(
            dim, connectivity.shape[0], elem_tags,
        )
    finally:
        if started:
            gmsh.finalize()

    return Mesh(
        nodes=nodes,
        connectivity=connectivity,
        element_family=family,
        element_blocks=element_blocks,
        node_sets={},
        side_sets={},
        element_block_ids=element_block_ids,
    )


def _mesh_dimension() -> int:
    """Top element dimension present, 3 (volume) preferred over 2 (surface).

    A 3D mesh may also carry 2D boundary elements, so dimension 3 is checked
    first. Raises if neither dimension has elements.
    """
    for dim in (3, 2):
        elem_types, _tags, _nodes = gmsh.model.mesh.getElements(dim)
        if len(elem_types) > 0:
            return dim
    raise GmshFormatError("gmsh mesh has no 2D or 3D elements")


def _index_by_tag(tags: NDArray[np.int64], n: int) -> NDArray[np.intp]:
    """Array indexed by tag giving each tag's 0-based position (-1 if absent),
    so a tag array remaps to positions by fancy indexing.
    """
    table = np.full(int(tags.max()) + 1, -1, dtype=np.intp)
    table[tags] = np.arange(n, dtype=np.intp)
    return table


def _read_nodes(dim: int) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    if node_tags.shape[0] == 0:
        raise GmshFormatError("gmsh mesh has no nodes")
    nodes = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if dim == 2:
        nodes = _drop_z_for_planar_mesh(nodes)
    return nodes, _index_by_tag(node_tags, node_tags.shape[0])


def _drop_z_for_planar_mesh(
        nodes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the x and y columns of a 2D mesh's ``(N, 3)`` gmsh coordinates.

    A 2D cmad mesh lies in a plane of constant z, but gmsh still stores three
    coordinates per node. Raises if z varies (a surface mesh read as 2D),
    measured relative to the xy extent.
    """
    z_extent = float(nodes[:, 2].max() - nodes[:, 2].min())
    xy = nodes[:, :2]
    xy_extent = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    if z_extent > 1e-9 * max(xy_extent, 1.0):
        raise GmshFormatError(
            f"2D gmsh mesh varies in z (z extent {z_extent:.3e} vs xy extent "
            f"{xy_extent:.3e}); cmad 2D meshes lie in a plane of constant z"
        )
    return xy


def _read_elements(
        dim: int, index_by_tag: NDArray[np.intp],
) -> tuple[NDArray[np.intp], ElementFamily, NDArray[np.int64]]:
    elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(dim)
    if len(elem_types) > 1:
        raise GmshFormatError(
            "all elements must share one element family; got gmsh element "
            f"types {[int(t) for t in elem_types]}"
        )
    gmsh_type = int(elem_types[0])
    if gmsh_type not in _GMSH_TYPE_TO_FAMILY:
        raise GmshFormatError(
            f"unsupported gmsh element type {gmsh_type}; supported types are "
            f"{sorted(_GMSH_TYPE_TO_FAMILY)} (3-node tri, 4-node quad, 4-node "
            f"tet, 8-node hex)"
        )
    family = _GMSH_TYPE_TO_FAMILY[gmsh_type]

    tags = np.asarray(elem_tags[0], dtype=np.int64)
    flat = np.asarray(node_tags[0], dtype=np.int64)
    n_elems = tags.shape[0]
    npe = flat.shape[0] // n_elems
    connectivity = index_by_tag[flat].reshape(n_elems, npe).astype(np.intp)
    return connectivity, family, tags


def _read_blocks(
        dim: int, n_elems: int, elem_tags: NDArray[np.int64],
) -> tuple[dict[str, NDArray[np.intp]], dict[str, int]]:
    groups = gmsh.model.getPhysicalGroups(dim)
    if not groups:
        return {"all": np.arange(n_elems, dtype=np.intp)}, {}

    row_of_tag = _index_by_tag(elem_tags, n_elems)
    blocks: dict[str, NDArray[np.intp]] = {}
    block_ids: dict[str, int] = {}
    assigned = np.zeros(n_elems, dtype=bool)
    for group_dim, tag in sorted(groups, key=lambda g: g[1]):
        name = gmsh.model.getPhysicalName(group_dim, tag) or f"block_{tag}"
        group_tags: list[NDArray[np.int64]] = []
        for entity in gmsh.model.getEntitiesForPhysicalGroup(group_dim, tag):
            _types, ent_tags, _nodes = gmsh.model.mesh.getElements(
                group_dim, int(entity),
            )
            group_tags.extend(np.asarray(a, dtype=np.int64) for a in ent_tags)
        rows = (
            np.unique(row_of_tag[np.concatenate(group_tags)])
            if group_tags else np.empty(0, dtype=np.intp)
        )
        if assigned[rows].any():
            raise GmshFormatError(
                f"physical group {name!r} overlaps another group on some "
                "elements; cmad blocks must partition the elements"
            )
        assigned[rows] = True
        blocks[name] = rows.astype(np.intp)
        block_ids[name] = int(tag)

    if not assigned.all():
        raise GmshFormatError(
            f"{int((~assigned).sum())} elements are in no {dim}D physical "
            "group; every element must belong to exactly one physical group"
        )
    return blocks, block_ids
