"""Geometric element-family tag for FE meshes.

Defined as a leaf module so it can be imported from any FE module
without creating an import cycle.
"""
from enum import IntEnum


class ElementFamily(IntEnum):
    """Geometric element family tag.

    Each member fixes the reference-element topology and pairs with a
    reference interpolant in :mod:`cmad.fem.interpolants` and a
    quadrature rule in :mod:`cmad.fem.quadrature`:

    - HEX_LINEAR / TET_LINEAR -- 3D; ``hex_linear`` / ``tet_linear``
      with ``hex_quadrature`` / ``tet_quadrature``. Sides are faces.
    - QUAD_LINEAR / TRI_LINEAR -- 2D; ``quad_linear`` / ``tri_linear``
      with ``quad_quadrature`` / ``tri_quadrature``. The cell is the 2D
      entity; sides are edges (2D families have no face sub-entities).
    """

    HEX_LINEAR = 0
    TET_LINEAR = 1
    QUAD_LINEAR = 2
    TRI_LINEAR = 3
