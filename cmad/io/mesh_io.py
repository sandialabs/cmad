"""Mesh-file reading dispatched by file suffix: Exodus or gmsh."""
from __future__ import annotations

from pathlib import Path

from cmad.fem.mesh import Mesh
from cmad.io.exodus import read_mesh
from cmad.io.gmsh import read_gmsh_mesh

_EXODUS_SUFFIXES = frozenset({".exo", ".g", ".e"})
_GMSH_SUFFIXES = frozenset({".msh"})


def read_mesh_file(path: str | Path) -> Mesh:
    """Read a mesh file into a :class:`Mesh`, choosing the reader by suffix:
    Exodus (``.exo`` / ``.g`` / ``.e``) or gmsh (``.msh``).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _EXODUS_SUFFIXES:
        return read_mesh(path)
    if suffix in _GMSH_SUFFIXES:
        return read_gmsh_mesh(path)
    raise ValueError(
        f"unrecognized mesh suffix {path.suffix!r}; expected one of "
        f"{sorted(_EXODUS_SUFFIXES | _GMSH_SUFFIXES)}"
    )
