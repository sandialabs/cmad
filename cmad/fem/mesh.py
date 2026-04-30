"""3D mesh data structure with element-block / node-set / side-set support.

Field naming follows Exodus convention so Exodus IO is a thin
translation layer rather than a remap. The structured-hex builder emits a
default ``"all"`` element block plus six built-in node sets and six
built-in side sets named ``{x,y,z}{min,max}_{nodes,sides}``. Node sets
serve as Dirichlet BC attach handles; side sets serve Neumann BCs and
Exodus mesh round-trip.

Hex local-face numbering (Exodus 0-based)::

    0: -z   1: +z   2: -y   3: +x   4: +y   5: -x

Tet local-face numbering (Exodus 0-based, with tet nodes (origin, +x, +y,
+z) per :func:`cmad.fem.interpolants.tet_linear`)::

    0: -y face        (nodes 0, 1, 3)
    1: slant face     (nodes 1, 2, 3)   outward normal (+x+y+z)
    2: -x face        (nodes 0, 3, 2)
    3: -z face        (nodes 0, 2, 1)

Hex node ordering matches :func:`cmad.fem.interpolants.hex_linear`::

    0: (-,-,-)   4: (-,-,+)
    1: (+,-,-)   5: (+,-,+)
    2: (+,+,-)   6: (+,+,+)
    3: (-,+,-)   7: (-,+,+)

The ``hex_to_tet_split`` helper uses the canonical 6-tet-per-hex diagonal
split along the body diagonal joining hex nodes 0 and 6 (Howell 1992
pattern). All 6 tets have positive volume on a positively-oriented hex.
"""
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cmad.fem.element_family import ElementFamily
from cmad.fem.finite_element import (
    P1_TET,
    Q1_HEX,
    EntityType,
    FiniteElement,
)

_NODES_PER_ELEMENT: dict[ElementFamily, int] = {
    ElementFamily.HEX_LINEAR: 8,
    ElementFamily.TET_LINEAR: 4,
}

_FACES_PER_ELEMENT: dict[ElementFamily, int] = {
    ElementFamily.HEX_LINEAR: 6,
    ElementFamily.TET_LINEAR: 4,
}


# Hex face-to-node table. Each row lists the four hex-local node indices
# of one face in CCW order from outside the element.
_HEX_FACE_NODES: NDArray[np.intp] = np.array(
    [
        [0, 3, 2, 1],   # face 0: -z
        [4, 5, 6, 7],   # face 1: +z
        [0, 1, 5, 4],   # face 2: -y
        [1, 2, 6, 5],   # face 3: +x
        [2, 3, 7, 6],   # face 4: +y
        [3, 0, 4, 7],   # face 5: -x
    ],
    dtype=np.intp,
)


# Tet face-to-node table. Each row lists the three tet-local node indices
# of one face in CCW order from outside the element.
_TET_FACE_NODES: NDArray[np.intp] = np.array(
    [
        [0, 1, 3],   # face 0: -y
        [1, 2, 3],   # face 1: slant
        [0, 3, 2],   # face 2: -x
        [0, 2, 1],   # face 3: -z
    ],
    dtype=np.intp,
)


# Hex local-edge table. Each row lists the two hex-local node indices of
# one edge. Order: 4 bottom-face edges, 4 top-face edges, 4 verticals.
_HEX_LOCAL_EDGES: NDArray[np.intp] = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],   # bottom (z=-1) face, CCW
        [4, 5], [5, 6], [6, 7], [7, 4],   # top (z=+1) face, CCW
        [0, 4], [1, 5], [2, 6], [3, 7],   # verticals
    ],
    dtype=np.intp,
)


# Tet local-edge table. Each row lists the two tet-local node indices of
# one edge.
_TET_LOCAL_EDGES: NDArray[np.intp] = np.array(
    [
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3],
        [2, 3],
    ],
    dtype=np.intp,
)


_LOCAL_EDGES_PER_ELEMENT: dict[ElementFamily, NDArray[np.intp]] = {
    ElementFamily.HEX_LINEAR: _HEX_LOCAL_EDGES,
    ElementFamily.TET_LINEAR: _TET_LOCAL_EDGES,
}

_LOCAL_FACES_PER_ELEMENT: dict[ElementFamily, NDArray[np.intp]] = {
    ElementFamily.HEX_LINEAR: _HEX_FACE_NODES,
    ElementFamily.TET_LINEAR: _TET_FACE_NODES,
}

_GEOMETRIC_FINITE_ELEMENT_PER_ELEMENT: dict[ElementFamily, FiniteElement] = {
    ElementFamily.HEX_LINEAR: Q1_HEX,
    ElementFamily.TET_LINEAR: P1_TET,
}


# Hex-to-tet split: each hex's 8 corners produce 6 tetrahedra. Each row
# lists the 4 hex-local node indices that form one tet, in tet_linear
# ordering (origin, +x, +y, +z). The split shares the body diagonal 0-6.
# All 6 rows produce positive-volume tets when the hex is positively
# oriented (verified analytically on the unit cube).
_HEX_TO_TET_LOCAL: NDArray[np.intp] = np.array(
    [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ],
    dtype=np.intp,
)


# Hex-face-to-tet-faces correspondence for the body-diagonal split
# encoded in ``_HEX_TO_TET_LOCAL``. For each hex face id (0..5), the two
# (tet_local_idx, tet_face_id) pairs that the hex face splits into.
# Shape (6, 2, 2): [hex_face, which_tet, (tet_local_idx, tet_face_id)].
_HEX_FACE_TO_TET_FACES: NDArray[np.intp] = np.array(
    [
        [[0, 3], [1, 3]],   # hex face 0 (-z) -> tet 0 face 3, tet 1 face 3
        [[3, 1], [4, 1]],   # hex face 1 (+z) -> tet 3 face 1, tet 4 face 1
        [[4, 3], [5, 3]],   # hex face 2 (-y) -> tet 4 face 3, tet 5 face 3
        [[0, 1], [5, 1]],   # hex face 3 (+x) -> tet 0 face 1, tet 5 face 1
        [[1, 1], [2, 1]],   # hex face 4 (+y) -> tet 1 face 1, tet 2 face 1
        [[2, 3], [3, 3]],   # hex face 5 (-x) -> tet 2 face 3, tet 3 face 3
    ],
    dtype=np.intp,
)


def _enumerate_edges(
        connectivity: NDArray[np.intp],
        local_edges: NDArray[np.intp],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Build the deduplicated edge table + per-element edge index map.

    For each element, look up vertex pairs via the family's local-edge
    table and the element's connectivity row; sort each pair so the
    smaller vertex index comes first; deduplicate across all elements
    via :func:`numpy.unique` with ``return_inverse``.

    Returns ``(edges, element_edges)``: ``edges`` has shape
    ``(N_edges, 2)`` with sorted-vertex-pair entries; ``element_edges``
    has shape ``(N_elems, n_local_edges)`` with global edge indices in
    the local-edge order matching the family's local table.
    """
    n_elems = connectivity.shape[0]
    n_local = local_edges.shape[0]
    edge_pairs = connectivity[:, local_edges]            # (n_elems, n_local, 2)
    edge_pairs_sorted = np.sort(edge_pairs, axis=2)
    flat = edge_pairs_sorted.reshape(-1, 2)
    edges, inverse = np.unique(flat, axis=0, return_inverse=True)
    element_edges = inverse.reshape(n_elems, n_local).astype(np.intp)
    return edges.astype(np.intp), element_edges


def _enumerate_faces(
        connectivity: NDArray[np.intp],
        local_faces: NDArray[np.intp],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Build the deduplicated face table + per-element face index map.

    Same pattern as :func:`_enumerate_edges`: per-element face vertex
    tuples are sorted into a canonical form and deduplicated across the
    mesh. Variable face-vertex counts (mixed quad / tri) are not
    supported here; ``local_faces`` must be a fixed ``(n_local_faces,
    n_face_vertices)`` table.

    Returns ``(faces, element_faces)``: ``faces`` has shape
    ``(N_faces, n_face_vertices)`` with sorted-vertex-tuple entries;
    ``element_faces`` has shape ``(N_elems, n_local_faces)`` with
    global face indices in the local-face order matching the family's
    local table.
    """
    n_elems = connectivity.shape[0]
    n_local = local_faces.shape[0]
    n_face_verts = local_faces.shape[1]
    face_verts = connectivity[:, local_faces]            # (n_elems, n_local, n_face_verts)
    face_verts_sorted = np.sort(face_verts, axis=2)
    flat = face_verts_sorted.reshape(-1, n_face_verts)
    faces, inverse = np.unique(flat, axis=0, return_inverse=True)
    element_faces = inverse.reshape(n_elems, n_local).astype(np.intp)
    return faces.astype(np.intp), element_faces


@dataclass(frozen=True)
class Mesh:
    """3D mesh with element_blocks, node_sets, and side_sets.

    ``nodes`` has shape ``(N_nodes, 3)`` — geometric node coordinates.
    For linear hex / linear tet bases, every mesh node is also a
    geometric vertex.

    ``connectivity`` has shape ``(N_elems, nodes_per_elem)``;
    ``nodes_per_elem`` is fixed by ``element_family`` (8 for HEX_LINEAR,
    4 for TET_LINEAR).

    ``element_blocks`` is the multi-material dispatch handle: a name-keyed
    dict of element-index arrays. Single-material decks use the implicit
    ``"all"`` block. The collection must form a strict partition of
    ``[0, N_elems)`` — every element in exactly one block, no overlaps,
    no gaps. Validated in ``__post_init__``.

    ``node_sets`` is a name-keyed dict of node-index arrays (Exodus node
    sets). Used as Dirichlet BC attach handles.

    ``side_sets`` is a name-keyed dict of ``(n_in_set, 2)`` arrays of
    ``(elem_id, local_face_id)`` pairs (Exodus side sets). Used as
    Neumann BC attach handles and for Exodus mesh round-trip.

    ``geometric_finite_element`` is the :class:`FiniteElement` whose
    reference-frame interpolant maps reference-element coords to
    physical coords. Defaults (when ``None`` is passed) to a P1
    element matching ``element_family`` (P1_TET / Q1_HEX). Resolved
    in ``__post_init__``; never ``None`` after construction.

    ``edges``, ``element_edges``, ``faces``, ``element_faces`` are
    derived tables computed at construction (``init=False`` fields).
    ``edges`` has shape ``(N_edges, 2)`` — deduplicated edge endpoints
    as sorted-vertex pairs. ``element_edges`` has shape
    ``(N_elems, n_edges_per_elem)`` — per-element global edge indices
    in canonical local-edge order. ``faces`` and ``element_faces``
    have analogous shapes for the face enumeration; ``n_face_vertices``
    is fixed by ``element_family`` (4 for hex, 3 for tet).

    Frozen dataclass; no mutation API. Validation and derived-table
    computation run in ``__post_init__``.
    """

    nodes: NDArray[np.floating]
    connectivity: NDArray[np.intp]
    element_family: ElementFamily
    element_blocks: dict[str, NDArray[np.intp]]
    node_sets: dict[str, NDArray[np.intp]]
    side_sets: dict[str, NDArray[np.intp]]
    geometric_finite_element: FiniteElement | None = None
    edges: NDArray[np.intp] = field(
        init=False,
        default_factory=lambda: np.empty((0, 2), dtype=np.intp),
    )
    element_edges: NDArray[np.intp] = field(
        init=False,
        default_factory=lambda: np.empty((0, 0), dtype=np.intp),
    )
    faces: NDArray[np.intp] = field(
        init=False,
        default_factory=lambda: np.empty((0, 0), dtype=np.intp),
    )
    element_faces: NDArray[np.intp] = field(
        init=False,
        default_factory=lambda: np.empty((0, 0), dtype=np.intp),
    )

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(
                f"nodes must have shape (N_nodes, 3); got {self.nodes.shape}"
            )
        if self.connectivity.ndim != 2:
            raise ValueError(
                "connectivity must be 2D; got shape "
                f"{self.connectivity.shape}"
            )
        expected_npe = _NODES_PER_ELEMENT[self.element_family]
        if self.connectivity.shape[1] != expected_npe:
            raise ValueError(
                f"connectivity columns ({self.connectivity.shape[1]}) does "
                f"not match {self.element_family.name}'s {expected_npe} "
                "nodes per element"
            )

        n_nodes = self.nodes.shape[0]
        n_elems = self.connectivity.shape[0]

        if n_elems > 0:
            min_idx = int(self.connectivity.min())
            max_idx = int(self.connectivity.max())
            if min_idx < 0 or max_idx >= n_nodes:
                raise ValueError(
                    f"connectivity indices must be in [0, {n_nodes}); "
                    f"got [{min_idx}, {max_idx}]"
                )

        # element_blocks: strict partition of [0, n_elems)
        counts = np.zeros(n_elems, dtype=np.intp)
        for name, indices in self.element_blocks.items():
            if indices.ndim != 1:
                raise ValueError(
                    f"element_blocks['{name}'] must be 1D; got shape "
                    f"{indices.shape}"
                )
            if indices.size > 0:
                if int(indices.min()) < 0 or int(indices.max()) >= n_elems:
                    raise ValueError(
                        f"element_blocks['{name}'] indices out of range "
                        f"[0, {n_elems})"
                    )
                np.add.at(counts, indices, 1)
        if not np.array_equal(counts, np.ones(n_elems, dtype=np.intp)):
            n_missing = int((counts == 0).sum())
            n_doubled = int((counts > 1).sum())
            raise ValueError(
                "element_blocks must form a strict partition of "
                f"[0, {n_elems}); {n_missing} unassigned and {n_doubled} "
                "multiply-assigned elements"
            )

        # node_sets: indices in range
        for name, indices in self.node_sets.items():
            if indices.ndim != 1:
                raise ValueError(
                    f"node_sets['{name}'] must be 1D; got shape "
                    f"{indices.shape}"
                )
            if indices.size > 0 and (
                int(indices.min()) < 0 or int(indices.max()) >= n_nodes
            ):
                raise ValueError(
                    f"node_sets['{name}'] indices out of range "
                    f"[0, {n_nodes})"
                )

        # side_sets: (elem_id, local_face_id) pairs in range
        n_faces = _FACES_PER_ELEMENT[self.element_family]
        for name, pairs in self.side_sets.items():
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError(
                    f"side_sets['{name}'] must have shape (n, 2); got "
                    f"{pairs.shape}"
                )
            if pairs.shape[0] > 0:
                elem_ids = pairs[:, 0]
                face_ids = pairs[:, 1]
                if (
                    int(elem_ids.min()) < 0
                    or int(elem_ids.max()) >= n_elems
                ):
                    raise ValueError(
                        f"side_sets['{name}'] elem_ids out of range "
                        f"[0, {n_elems})"
                    )
                if (
                    int(face_ids.min()) < 0
                    or int(face_ids.max()) >= n_faces
                ):
                    raise ValueError(
                        f"side_sets['{name}'] local_face_ids out of range "
                        f"[0, {n_faces}) for {self.element_family.name}"
                    )

        edges, element_edges = _enumerate_edges(
            self.connectivity,
            _LOCAL_EDGES_PER_ELEMENT[self.element_family],
        )
        faces, element_faces = _enumerate_faces(
            self.connectivity,
            _LOCAL_FACES_PER_ELEMENT[self.element_family],
        )
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "element_edges", element_edges)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "element_faces", element_faces)

        if self.geometric_finite_element is None:
            object.__setattr__(
                self,
                "geometric_finite_element",
                _GEOMETRIC_FINITE_ELEMENT_PER_ELEMENT[self.element_family],
            )

    def entity_count(self, entity_type: EntityType) -> int:
        """Number of mesh entities of the given ``entity_type``."""
        if entity_type == EntityType.VERTEX:
            return int(self.nodes.shape[0])
        if entity_type == EntityType.EDGE:
            return int(self.edges.shape[0])
        if entity_type == EntityType.FACE:
            return int(self.faces.shape[0])
        if entity_type == EntityType.CELL:
            return int(self.connectivity.shape[0])
        raise ValueError(f"unknown entity_type: {entity_type!r}")


def StructuredHexMesh(
    lengths: tuple[float, float, float],
    divisions: tuple[int, int, int],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Mesh:
    """Build a structured hex mesh on a Cartesian box.

    Generates a regular grid of ``(nx+1) * (ny+1) * (nz+1)`` nodes and
    ``nx * ny * nz`` linear hex elements on the box
    ``[origin, origin + lengths]``. Element nodes are emitted in the
    hex_linear node ordering (bottom face CCW from (-,-,-), top face CCW
    from (-,-,+)). Element index ``e = i * ny * nz + j * nz + k`` with
    ``(i, j, k)`` the per-axis element-cell index.

    Populates a default ``"all"`` element block, six built-in node sets
    (``{x,y,z}{min,max}_nodes``) and six built-in side sets
    (``{x,y,z}{min,max}_sides``).

    Raises :class:`ValueError` if any division is < 1.
    """
    Lx, Ly, Lz = lengths
    nx, ny, nz = divisions
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError(
            f"divisions must all be >= 1; got ({nx}, {ny}, {nz})"
        )
    ox, oy, oz = origin

    xs = np.linspace(ox, ox + Lx, nx + 1)
    ys = np.linspace(oy, oy + Ly, ny + 1)
    zs = np.linspace(oz, oz + Lz, nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    n_elems = nx * ny * nz
    vid_grid = np.arange(
        (nx + 1) * (ny + 1) * (nz + 1), dtype=np.intp
    ).reshape(nx + 1, ny + 1, nz + 1)

    EI, EJ, EK = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    connectivity = np.stack(
        [
            vid_grid[EI, EJ, EK],               # 0: (-,-,-)
            vid_grid[EI + 1, EJ, EK],           # 1: (+,-,-)
            vid_grid[EI + 1, EJ + 1, EK],       # 2: (+,+,-)
            vid_grid[EI, EJ + 1, EK],           # 3: (-,+,-)
            vid_grid[EI, EJ, EK + 1],           # 4: (-,-,+)
            vid_grid[EI + 1, EJ, EK + 1],       # 5: (+,-,+)
            vid_grid[EI + 1, EJ + 1, EK + 1],   # 6: (+,+,+)
            vid_grid[EI, EJ + 1, EK + 1],       # 7: (-,+,+)
        ],
        axis=-1,
    ).reshape(-1, 8)

    element_blocks = {"all": np.arange(n_elems, dtype=np.intp)}

    node_sets = {
        "xmin_nodes": vid_grid[0, :, :].ravel(),
        "xmax_nodes": vid_grid[-1, :, :].ravel(),
        "ymin_nodes": vid_grid[:, 0, :].ravel(),
        "ymax_nodes": vid_grid[:, -1, :].ravel(),
        "zmin_nodes": vid_grid[:, :, 0].ravel(),
        "zmax_nodes": vid_grid[:, :, -1].ravel(),
    }

    elem_idx_grid = np.arange(n_elems, dtype=np.intp).reshape(nx, ny, nz)

    def _side_set(elems: NDArray[np.intp], face_id: int) -> NDArray[np.intp]:
        return np.column_stack(
            [elems, np.full(elems.shape, face_id, dtype=np.intp)]
        )

    side_sets = {
        "xmin_sides": _side_set(elem_idx_grid[0, :, :].ravel(), 5),
        "xmax_sides": _side_set(elem_idx_grid[-1, :, :].ravel(), 3),
        "ymin_sides": _side_set(elem_idx_grid[:, 0, :].ravel(), 2),
        "ymax_sides": _side_set(elem_idx_grid[:, -1, :].ravel(), 4),
        "zmin_sides": _side_set(elem_idx_grid[:, :, 0].ravel(), 0),
        "zmax_sides": _side_set(elem_idx_grid[:, :, -1].ravel(), 1),
    }

    return Mesh(
        nodes=nodes,
        connectivity=connectivity,
        element_family=ElementFamily.HEX_LINEAR,
        element_blocks=element_blocks,
        node_sets=node_sets,
        side_sets=side_sets,
    )


def hex_to_tet_split(mesh: Mesh) -> Mesh:
    """Split each hex into 6 tetrahedra along a body diagonal.

    Each hex with corners 0..7 in hex_linear ordering splits into 6 tets
    sharing the body diagonal from corner 0 (-,-,-) to corner 6 (+,+,+).
    Tet node ordering matches :func:`cmad.fem.interpolants.tet_linear`
    (origin, +x, +y, +z). All 6 tets have positive volume on a
    positively-oriented hex.

    Element-block membership maps each hex's entry to its 6 descendant
    tets at indices ``[6 * hex_id, 6 * hex_id + 5]``. Node sets carry over
    unchanged. Side sets remap through the hex-face-to-tet-face
    correspondence (each hex face splits into 2 triangular tet faces).

    Raises :class:`ValueError` if the input mesh is not HEX_LINEAR.
    """
    if mesh.element_family != ElementFamily.HEX_LINEAR:
        raise ValueError(
            "hex_to_tet_split requires HEX_LINEAR mesh; got "
            f"{mesh.element_family.name}"
        )

    # connectivity_tet[6*e + t, c] = mesh.connectivity[e, _HEX_TO_TET_LOCAL[t, c]]
    connectivity_tet = mesh.connectivity[:, _HEX_TO_TET_LOCAL].reshape(-1, 4)

    element_blocks_tet: dict[str, NDArray[np.intp]] = {}
    for name, hex_indices in mesh.element_blocks.items():
        tet_indices = (
            hex_indices[:, None] * 6
            + np.arange(6, dtype=np.intp)[None, :]
        ).ravel()
        element_blocks_tet[name] = tet_indices

    node_sets_tet = {k: v.copy() for k, v in mesh.node_sets.items()}

    side_sets_tet: dict[str, NDArray[np.intp]] = {}
    for name, hex_sides in mesh.side_sets.items():
        if hex_sides.shape[0] == 0:
            side_sets_tet[name] = np.empty((0, 2), dtype=np.intp)
            continue
        hex_ids = hex_sides[:, 0]
        hex_face_ids = hex_sides[:, 1]
        # _HEX_FACE_TO_TET_FACES[face_id] -> 2 (tet_local_idx, tet_face_id)
        tet_pairs = _HEX_FACE_TO_TET_FACES[hex_face_ids]   # (n, 2, 2)
        tet_local_idx = tet_pairs[:, :, 0]                 # (n, 2)
        tet_face_id = tet_pairs[:, :, 1]                   # (n, 2)
        tet_ids = hex_ids[:, None] * 6 + tet_local_idx     # (n, 2)
        tet_sides = np.stack(
            [tet_ids, tet_face_id], axis=-1
        ).reshape(-1, 2)
        side_sets_tet[name] = tet_sides

    return Mesh(
        nodes=mesh.nodes.copy(),
        connectivity=connectivity_tet,
        element_family=ElementFamily.TET_LINEAR,
        element_blocks=element_blocks_tet,
        node_sets=node_sets_tet,
        side_sets=side_sets_tet,
    )
