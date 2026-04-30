"""Per-family reference-element topology tables.

Owns the static lookup tables that describe the geometric topology of
each :class:`~cmad.fem.element_family.ElementFamily`'s reference
element: which local vertex slots make up each face, which make up
each side (boundary entity), etc. Leaf module — no class definitions
and no imports from other ``cmad.fem`` modules — so consumers
(``mesh``, ``finite_element``) can import freely without cycles.

Two distinct dispatches share the same underlying face-vertex arrays
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
