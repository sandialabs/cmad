"""Per-family reference-element topology tables.

Owns the static lookup tables that describe the geometric topology of
each :class:`~cmad.fem.element_family.ElementFamily`'s reference
element: which local vertex slots make up each face, which make up
each side (boundary entity), how to lift a face quadrature point to
a ref-volume coord, etc. Leaf module — no class definitions and no
imports from other ``cmad.fem`` modules — so consumers (``mesh``,
``finite_element``, ``neumann``) can import freely without cycles.

Three distinct dispatches share the same underlying face-vertex arrays
for our current 3D families, but have separate semantic roles:

- :data:`_LOCAL_FACES_PER_ELEMENT` — per-3D-family local-face vertex
  slots, consumed by mesh face enumeration. Genuinely 3D-only;
  2D families have no face sub-entities (their cells are the 2D
  entity).
- :data:`_LOCAL_SIDES_PER_ELEMENT` — per-family local-side vertex
  slots, consumed by sideset-keyed BC and per-side surface-integral
  resolvers. Side = (d-1)-dim boundary entity; for 3D families
  sides are faces (entries alias the face tables above); for future
  2D families sides will be edges (entries will point to edge-vertex
  tables).
- :data:`_REF_SIDE_LIFT_PER_ELEMENT` — per-(family, local_side_id)
  ref-side → ref-volume affine lift, consumed by per-side
  integral evaluators. Each entry is an ``(origin, tangents)`` pair
  such that a side quadrature point ``(s, t)`` lifts to the ref-
  volume coord ``ξ = origin + tangents @ [s, t]``. Tangent columns
  are oriented so right-hand-rule
  ``cross(tangents[:, 0], tangents[:, 1])`` points outward, inheriting the
  CCW-from-outside vertex ordering of
  :data:`_LOCAL_FACES_PER_ELEMENT`. Hex face params live in
  ``[-1, 1]^2`` (matching :func:`cmad.fem.quadrature.quad_quadrature`);
  tet face params in the unit triangle (matching
  :func:`cmad.fem.quadrature.tri_quadrature`).

Hex local-face numbering (Exodus 0-based)::

    0: -z   1: +z   2: -y   3: +x   4: +y   5: -x

Tet local-face numbering (Exodus 0-based, with tet nodes (origin, +x,
+y, +z) per :func:`cmad.fem.interpolants.tet_linear`)::

    0: -y face        (nodes 0, 1, 3)
    1: slant face     (nodes 1, 2, 3)   outward normal (+x+y+z)
    2: -x face        (nodes 0, 3, 2)
    3: -z face        (nodes 0, 2, 1)
"""
import numpy as np
from numpy.typing import NDArray

from cmad.fem.element_family import ElementFamily

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


# Per-3D-family local face → vertex slots table. Used by mesh face
# enumeration in `_enumerate_faces`. Genuinely 3D-only; 2D families
# (when added) will not have entries here.
_LOCAL_FACES_PER_ELEMENT: dict[ElementFamily, NDArray[np.intp]] = {
    ElementFamily.HEX_LINEAR: _HEX_FACE_NODES,
    ElementFamily.TET_LINEAR: _TET_FACE_NODES,
}


# Per-family local side → vertex slots table. Side = (d-1)-dim boundary
# entity. For 3D families sides are faces (entries alias the face tables
# above). For future 2D families sides will be edges; entries will point
# to edge-vertex tables.
_LOCAL_SIDES_PER_ELEMENT: dict[ElementFamily, NDArray[np.intp]] = {
    ElementFamily.HEX_LINEAR: _HEX_FACE_NODES,
    ElementFamily.TET_LINEAR: _TET_FACE_NODES,
}


# Reference-element node coordinates per family. Used to derive the
# ref-side lift tables below; a private restatement of the geometric
# anchor points encoded by the basis functions in
# :mod:`cmad.fem.interpolants`. Kept here so this module stays a
# leaf with no ``cmad.fem`` imports.
_HEX_REFERENCE_NODES: NDArray[np.floating] = np.array(
    [
        [-1.0, -1.0, -1.0],   # 0
        [+1.0, -1.0, -1.0],   # 1
        [+1.0, +1.0, -1.0],   # 2
        [-1.0, +1.0, -1.0],   # 3
        [-1.0, -1.0, +1.0],   # 4
        [+1.0, -1.0, +1.0],   # 5
        [+1.0, +1.0, +1.0],   # 6
        [-1.0, +1.0, +1.0],   # 7
    ],
    dtype=np.float64,
)


_TET_REFERENCE_NODES: NDArray[np.floating] = np.array(
    [
        [0.0, 0.0, 0.0],   # 0
        [1.0, 0.0, 0.0],   # 1
        [0.0, 1.0, 0.0],   # 2
        [0.0, 0.0, 1.0],   # 3
    ],
    dtype=np.float64,
)


def _quad_face_lift(
        face_node_ids: NDArray[np.intp],
        ref_nodes: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build ``(origin, tangents)`` for a 4-vertex face on ``[-1, 1]^2``.

    The bilinear isoparametric map of a planar quad face with
    vertices ``(v0, v1, v2, v3)`` in CCW-from-outside order reduces
    to the affine form ``ξ(s, t) = origin + tangents @ [s, t]``
    where ``origin`` is the face centroid and the tangent columns
    are the bilinear interpolant's per-axis derivatives at the
    centroid:

    - ``tangents[:, 0] = (-v0 + v1 + v2 - v3) / 4`` = ``∂ξ/∂s``;
    - ``tangents[:, 1] = (-v0 - v1 + v2 + v3) / 4`` = ``∂ξ/∂t``.

    With CCW-from-outside vertex ordering, right-hand-rule
    ``cross(tangents[:, 0], tangents[:, 1])`` points outward.
    """
    v = ref_nodes[face_node_ids]   # (4, 3)
    origin = v.mean(axis=0)
    col_s = (-v[0] + v[1] + v[2] - v[3]) / 4.0
    col_t = (-v[0] - v[1] + v[2] + v[3]) / 4.0
    tangents = np.stack([col_s, col_t], axis=1)
    return origin, tangents


def _tri_face_lift(
        face_node_ids: NDArray[np.intp],
        ref_nodes: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build ``(origin, tangents)`` for a 3-vertex face on the unit triangle.

    Linear interpolation ``ξ(s, t) = (1 - s - t) v0 + s v1 + t v2``
    over the unit triangle ``{(s, t) : s ≥ 0, t ≥ 0, s + t ≤ 1}``
    gives the affine form ``ξ = origin + tangents @ [s, t]`` with
    ``origin = v0``, ``tangents[:, 0] = v1 - v0``,
    ``tangents[:, 1] = v2 - v0``. CCW-from-outside vertex ordering
    encodes outward orientation via right-hand rule.
    """
    v = ref_nodes[face_node_ids]   # (3, 3)
    origin = v[0]
    col_s = v[1] - v[0]
    col_t = v[2] - v[0]
    tangents = np.stack([col_s, col_t], axis=1)
    return origin, tangents


# Per-family ref-side lift tables. Each entry is a list of
# ``(origin, tangents)`` pairs ordered to match
# :data:`_LOCAL_SIDES_PER_ELEMENT`. ``origin`` has shape ``(3,)``;
# ``tangents`` has shape ``(3, 2)``. The lift maps a side IP coord
# ``(s, t)`` to a ref-volume coord ``ξ = origin + tangents @ [s, t]``;
# for the 3D families the entries are face lifts with ``(s, t)`` over
# ``[-1, 1]^2`` for hex faces and over the unit triangle for tet
# faces.
_REF_SIDE_LIFT_PER_ELEMENT: dict[
    ElementFamily,
    list[tuple[NDArray[np.floating], NDArray[np.floating]]],
] = {
    ElementFamily.HEX_LINEAR: [
        _quad_face_lift(_HEX_FACE_NODES[i], _HEX_REFERENCE_NODES)
        for i in range(_HEX_FACE_NODES.shape[0])
    ],
    ElementFamily.TET_LINEAR: [
        _tri_face_lift(_TET_FACE_NODES[i], _TET_REFERENCE_NODES)
        for i in range(_TET_FACE_NODES.shape[0])
    ],
}


def ref_side_lift(
        family: ElementFamily,
        local_side_id: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return ``(origin, tangents)`` for one (family, local_side_id) lift.

    The lift maps a side quadrature point ``(s, t)`` to a ref-volume
    coord ``ξ = origin + tangents @ [s, t]``; ``origin`` has shape
    ``(3,)`` and ``tangents`` has shape ``(3, 2)``. For
    :data:`~cmad.fem.element_family.ElementFamily.HEX_LINEAR`,
    ``(s, t)`` ranges over ``[-1, 1]^2`` (matching
    :func:`cmad.fem.quadrature.quad_quadrature`); for
    :data:`~cmad.fem.element_family.ElementFamily.TET_LINEAR`,
    ``(s, t)`` ranges over the unit triangle (matching
    :func:`cmad.fem.quadrature.tri_quadrature`).

    The right-hand-rule cross product of the tangent columns points
    outward by construction; the CCW-from-outside vertex ordering of
    :data:`_LOCAL_FACES_PER_ELEMENT` is the source of the
    orientation. Side integrators consume ``|cross(t_s, t_t)|`` for the
    per-side area element (length element in 2D) and the normalized
    cross product as the outward unit normal.

    Raises ``KeyError`` if ``family`` has no lift table; ``IndexError``
    if ``local_side_id`` is out of range.
    """
    return _REF_SIDE_LIFT_PER_ELEMENT[family][local_side_id]
