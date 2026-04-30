"""Geometric element-family tag for FE meshes.

Defined as a leaf module so it can be imported from any FE module
without creating an import cycle.
"""
from enum import IntEnum


class ElementFamily(IntEnum):
    """Element family tag; fixes nodes_per_element and faces_per_element.

    HEX_LINEAR pairs with :func:`cmad.fem.interpolants.hex_linear` and
    :func:`cmad.fem.quadrature.hex_quadrature`. TET_LINEAR pairs with
    :func:`cmad.fem.interpolants.tet_linear` and
    :func:`cmad.fem.quadrature.tet_quadrature`.
    """

    HEX_LINEAR = 0
    TET_LINEAR = 1
