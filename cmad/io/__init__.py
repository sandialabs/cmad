"""Public I/O surface for cmad.

Mesh + time-stepped result-variable I/O against Exodus II files.
:class:`FieldSpec` declares a result field's ``(name, var_type)``;
:class:`ExodusWriter` lays down the mesh skeleton plus the result-
variable schema at construction and appends time rows via
:meth:`ExodusWriter.write_step`. :func:`read_mesh` and
:func:`read_results` are the read-side counterparts.

Component-naming and storage-permutation helpers
(:func:`component_names`, internal vs Exodus sym-tensor ordering)
plus IP- and global-field reductions (:func:`ip_average_to_element`,
:func:`volume_average_global_field`) live alongside the writer +
reader so callers can build / consume Exodus result variables
without crossing into ``cmad.fem``. Submodules remain importable —
the names below are the curated public surface.
"""
from cmad.io.exodus import (
    ExodusFormatError,
    ExodusWriter,
    read_mesh,
    read_results,
)
from cmad.io.results import (
    ExodusResults,
    FieldSpec,
    component_names,
    ip_average_to_element,
    volume_average_global_field,
)

__all__ = [
    "ExodusFormatError",
    "ExodusResults",
    "ExodusWriter",
    "FieldSpec",
    "component_names",
    "ip_average_to_element",
    "read_mesh",
    "read_results",
    "volume_average_global_field",
]
