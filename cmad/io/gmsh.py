"""gmsh mesh input for cmad's FE workflow.

Reads a gmsh ``.msh`` file into :class:`cmad.fem.mesh.Mesh` through the gmsh
Python API. Parallel to :func:`cmad.io.exodus.read_mesh`; a deck's
``discretization.mesh file`` accepts either format via the suffix dispatch in
:func:`cmad.io.mesh_io.read_mesh_file`.

What the reader maps:

- **Nodes**: ``gmsh.model.mesh.getNodes`` coordinates as ``(N, 3)``. gmsh node
  tags (any positive integers) remap to 0-based contiguous indices.
- **Elements**: 3D elements only, one family per mesh (raises on mixed). gmsh
  element type 4 (4-node tetrahedron) is ``ElementFamily.TET_LINEAR``; type 5
  (8-node hexahedron) is ``HEX_LINEAR``.
- **Element blocks**: each 3D physical group becomes one block (name from the
  group name, id from the group tag into ``element_block_ids``). With no 3D
  physical groups, a single ``"all"`` block holds every element.

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
    4: ElementFamily.TET_LINEAR,
    5: ElementFamily.HEX_LINEAR,
}


class GmshFormatError(ValueError):
    """Raised when a gmsh mesh violates cmad's expected schema."""


def read_gmsh_mesh(path: str | Path) -> Mesh:
    """Read a gmsh ``.msh`` file into a :class:`Mesh` (module docstring for the
    node / element / block mapping).

    Raises :class:`GmshFormatError` for no nodes, no 3D elements, mixed or
    unknown element families, or physical groups that do not partition the
    elements.
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
        nodes, index_by_tag = _read_nodes()
        connectivity, family, elem_tags = _read_volume_elements(index_by_tag)
        element_blocks, element_block_ids = _read_blocks(
            connectivity.shape[0], elem_tags,
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


def _index_by_tag(tags: NDArray[np.int64], n: int) -> NDArray[np.intp]:
    """Array indexed by tag giving each tag's 0-based position (-1 if absent),
    so a tag array remaps to positions by fancy indexing.
    """
    table = np.full(int(tags.max()) + 1, -1, dtype=np.intp)
    table[tags] = np.arange(n, dtype=np.intp)
    return table


def _read_nodes() -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    if node_tags.shape[0] == 0:
        raise GmshFormatError("gmsh mesh has no nodes")
    nodes = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    return nodes, _index_by_tag(node_tags, node_tags.shape[0])


def _read_volume_elements(
        index_by_tag: NDArray[np.intp],
) -> tuple[NDArray[np.intp], ElementFamily, NDArray[np.int64]]:
    elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(3)
    if len(elem_types) == 0:
        raise GmshFormatError("gmsh mesh has no 3D elements")
    if len(elem_types) > 1:
        raise GmshFormatError(
            "all elements must share one element family; got gmsh element "
            f"types {[int(t) for t in elem_types]}"
        )
    gmsh_type = int(elem_types[0])
    if gmsh_type not in _GMSH_TYPE_TO_FAMILY:
        raise GmshFormatError(
            f"unsupported gmsh element type {gmsh_type}; supported types are "
            f"{sorted(_GMSH_TYPE_TO_FAMILY)} (4-node tet, 8-node hex)"
        )
    family = _GMSH_TYPE_TO_FAMILY[gmsh_type]

    tags = np.asarray(elem_tags[0], dtype=np.int64)
    flat = np.asarray(node_tags[0], dtype=np.int64)
    n_elems = tags.shape[0]
    npe = flat.shape[0] // n_elems
    connectivity = index_by_tag[flat].reshape(n_elems, npe).astype(np.intp)
    return connectivity, family, tags


def _read_blocks(
        n_elems: int, elem_tags: NDArray[np.int64],
) -> tuple[dict[str, NDArray[np.intp]], dict[str, int]]:
    groups = gmsh.model.getPhysicalGroups(3)
    if not groups:
        return {"all": np.arange(n_elems, dtype=np.intp)}, {}

    row_of_tag = _index_by_tag(elem_tags, n_elems)
    blocks: dict[str, NDArray[np.intp]] = {}
    block_ids: dict[str, int] = {}
    assigned = np.zeros(n_elems, dtype=bool)
    for dim, tag in sorted(groups, key=lambda g: g[1]):
        name = gmsh.model.getPhysicalName(dim, tag) or f"block_{tag}"
        group_tags: list[NDArray[np.int64]] = []
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            _types, ent_tags, _nodes = gmsh.model.mesh.getElements(
                dim, int(entity),
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
            f"{int((~assigned).sum())} elements are in no 3D physical group; "
            "every element must belong to exactly one physical volume"
        )
    return blocks, block_ids
