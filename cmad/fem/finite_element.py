"""Per-element-family finite-element DOF placement spec + interpolant.

A :class:`FiniteElement` pairs a geometric :class:`ElementFamily` with
a per-entity-type DOF placement count (vertex / edge / face / cell)
and a reference-frame interpolant. The DOF map allocates global DOFs
by walking the mesh's element topology and consulting each field's
``FiniteElement`` for "how many DOFs live on each entity of this kind"
— independent of how many geometric vertices a connectivity row
carries.

Equal-order Lagrange (P1 tet, Q1 hex) is the degenerate case:
``dofs_per_entity = {VERTEX: 1}``. Higher-order CG, mixed-basis, and
DG elements layer in by populating other entity-type entries (e.g.
``{VERTEX: 1, EDGE: 1}`` for P2 tet, ``{CELL: k}`` for DG of order
k-1).
"""
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from cmad.fem.element_family import ElementFamily
from cmad.fem.interpolants import hex_linear, tet_linear
from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.fem.topology import _LOCAL_SIDES_PER_ELEMENT
from cmad.typing import JaxArray


class EntityType(IntEnum):
    """Mesh entity type a FE DOF can be anchored at.

    Canonical ordering ``VERTEX < EDGE < FACE < CELL`` is the
    per-element DOF layout convention: when a FE has DOFs on multiple
    entity types, the per-element DOF vector concatenates them in
    this order, then within each type by local-entity index, then by
    multiplicity.
    """

    VERTEX = 0
    EDGE = 1
    FACE = 2
    CELL = 3


_ENTITY_COUNTS_PER_ELEMENT: dict[ElementFamily, dict[EntityType, int]] = {
    ElementFamily.HEX_LINEAR: {
        EntityType.VERTEX: 8,
        EntityType.EDGE: 12,
        EntityType.FACE: 6,
        EntityType.CELL: 1,
    },
    ElementFamily.TET_LINEAR: {
        EntityType.VERTEX: 4,
        EntityType.EDGE: 6,
        EntityType.FACE: 4,
        EntityType.CELL: 1,
    },
}


@dataclass(frozen=True, eq=False)
class FiniteElement:
    """Per-field DOF placement spec paired with a reference-frame interpolant.

    ``name`` is a human-readable label (e.g. ``"P1_TET"``,
    ``"Q1_HEX"``). Used in error messages and post-processing.

    ``element_family`` is the geometric host the FE rides on. The
    field's FE family must match the mesh's element family.

    ``dofs_per_entity`` maps each :class:`EntityType` the FE places
    DOFs on to the per-entity DOF count. Entries with value 0 may be
    omitted; missing entries imply 0. P1/Q1 examples carry only
    ``{VERTEX: 1}``; a P2 tet would carry ``{VERTEX: 1, EDGE: 1}``;
    a DG0 element would carry ``{CELL: 1}``.

    ``interpolant_fn`` evaluates the reference-frame shape functions
    at one integration point: ``xi -> ShapeFunctionsAtIP`` whose
    leading-axis size matches :attr:`num_dofs_per_element`. The shape
    functions are ordered ``VERTEX → EDGE → FACE → CELL`` then by
    within-entity local index then by multiplicity.

    Frozen with ``eq=False`` (no auto-generated ``__eq__`` /
    ``__hash__``): ``dofs_per_entity`` is a dict and is unhashable,
    and identity equality is sufficient for the consumer patterns
    (FE instances live in lists / on registries, never as dict keys).
    """

    name: str
    element_family: ElementFamily
    dofs_per_entity: dict[EntityType, int]
    interpolant_fn: Callable[[JaxArray], ShapeFunctionsAtIP]

    def __post_init__(self) -> None:
        if self.element_family not in _ENTITY_COUNTS_PER_ELEMENT:
            raise ValueError(
                f"FiniteElement '{self.name}': unknown element_family "
                f"{self.element_family!r}"
            )
        for entity_type, count in self.dofs_per_entity.items():
            if not isinstance(entity_type, EntityType):
                raise ValueError(
                    f"FiniteElement '{self.name}': dofs_per_entity keys "
                    f"must be EntityType members; got {entity_type!r}"
                )
            if not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"FiniteElement '{self.name}': "
                    f"dofs_per_entity[{entity_type.name}] must be a "
                    f"non-negative int; got {count!r}"
                )

    @property
    def num_dofs_per_element(self) -> int:
        """Total DOFs on one element of this FE.

        Sum over ``entity_type`` of ``entity_count_per_element *
        dofs_per_entity[entity_type]``, where ``entity_count_per_element``
        is the family's count of that entity type (e.g. 8 vertices on a
        hex, 6 edges on a tet).
        """
        entity_counts = _ENTITY_COUNTS_PER_ELEMENT[self.element_family]
        return sum(
            entity_counts[entity_type] * count
            for entity_type, count in self.dofs_per_entity.items()
        )

    def side_basis_fns(self, local_side_id: int) -> NDArray[np.intp]:
        """Per-element basis-fn local indices that lie on a local side.

        Side restriction of the per-element basis-fn ordering: returns
        the indices of basis functions whose support intersects the
        side. The companion to assembly's ``_element_basis_fns``,
        which gives global indices over the full element; this gives
        local indices over a single side.

        The "side" is the (d-1)-dim boundary entity of an element of
        family dimension d: in 3D this is a face; in 2D (when 2D
        families land) this would be an edge. Side indexing matches
        the family's local-side ordering convention (Exodus 0-based
        for 3D faces — see :mod:`cmad.fem.topology`).

        For VERTEX-only DOF placement (P1 / Q1 and other vertex-
        anchored fields), returns the family's local-side vertex
        indices in the family's canonical order — e.g., for
        ``HEX_LINEAR`` side 0 (face -z), returns ``[0, 3, 2, 1]``.

        Edge / face / cell DOF placements (P2 with edge-DOFs, etc.)
        require walking the side's sub-entity incidence to pick up
        per-entity DOF basis-fns; those incidence tables don't yet
        exist.

        Raises:
            NotImplementedError: if any non-VERTEX entry of
                ``dofs_per_entity`` is positive.
            ValueError: if ``local_side_id`` is out of range for the
                family.
        """
        non_vertex_terms = [
            et for et, count in self.dofs_per_entity.items()
            if et != EntityType.VERTEX and count > 0
        ]
        if non_vertex_terms:
            names = [et.name for et in non_vertex_terms]
            raise NotImplementedError(
                f"FiniteElement '{self.name}' has DOFs on {names} "
                "entities; side_basis_fns only supports VERTEX-only "
                "DOF placement. Sub-entity walks (edge/face/cell DOFs "
                "on a side) lift with P2+ assembly support."
            )
        sides = _LOCAL_SIDES_PER_ELEMENT[self.element_family]
        n_sides = sides.shape[0]
        if not (0 <= local_side_id < n_sides):
            raise ValueError(
                f"FiniteElement '{self.name}' (family "
                f"{self.element_family.name}): local_side_id "
                f"{local_side_id} out of range [0, {n_sides})"
            )
        return sides[local_side_id]


P1_TET = FiniteElement(
    name="P1_TET",
    element_family=ElementFamily.TET_LINEAR,
    dofs_per_entity={EntityType.VERTEX: 1},
    interpolant_fn=tet_linear,
)


Q1_HEX = FiniteElement(
    name="Q1_HEX",
    element_family=ElementFamily.HEX_LINEAR,
    dofs_per_entity={EntityType.VERTEX: 1},
    interpolant_fn=hex_linear,
)
