"""Global field DOF management with formula-based equation numbering.

Block-major (field-major) DOF numbering across one or more global
fields. For field f at basis function i with component k::

    eq = block_offsets[f] + i * num_dofs_per_basis_fn[f] + k

No eq-table is stored; element scatter computes the eq inline. Free vs.
prescribed status is tracked separately as a flat ``prescribed_indices``
array, decoupling DOF numbering (structural) from BC enforcement
(configurable). The DofMap commits to no particular enforcement
strategy; consumers choose UF/UP reduction, symmetric strong
enforcement, or matrix-free apply against the same metadata.

Per-field :class:`GlobalFieldLayout` carries a :class:`FiniteElement`
spec that says how many DOFs sit on each mesh entity type (vertex /
edge / face / cell). The DOF map allocates per-field DOFs by walking
the mesh's entity tables: ``mesh.entity_count(et) *
fe.dofs_per_entity[et]`` summed across entity types. P1 / Q1 fields
have ``{VERTEX: 1}`` and reduce to one basis-fn per mesh vertex.

Multi-field layouts naturally support fields without any BCs (those
fields contribute zero entries to ``prescribed_indices``).

Prescribed values are evaluated on demand via
:meth:`GlobalDofMap.evaluate_prescribed_values`, supporting time-
dependent BCs and expression-parser-backed callables. Each resolved BC
caches its node set's coordinate slice at construction so re-evaluation
is mesh-handle-free.
"""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cmad.fem.bcs import DirichletBC
from cmad.fem.finite_element import EntityType, FiniteElement
from cmad.fem.mesh import Mesh


@dataclass(frozen=True)
class GlobalFieldLayout:
    """Per-field DOF layout.

    ``name`` is the field symbol (e.g., ``"u"``, ``"p"``, ``"T"``) —
    used as the dict key in ``U.fields[name]`` / ``U.grad_fields[name]``
    and matched against ``DirichletBC.field_name``. By convention
    matches the corresponding entry in ``GlobalResidual.var_names``.
    The parallel ``GlobalResidual.resid_names`` carries the governing-
    equation label (``"displacement"``, ``"pressure"``, ``"energy"``)
    separately for output / deck schema / post-processing.

    ``finite_element`` is the :class:`FiniteElement` spec; its
    ``dofs_per_entity`` table determines how many basis-function
    coefficients live on each entity type (vertex / edge / face /
    cell), and its ``interpolant_fn`` supplies the reference-frame
    shape functions consumed by assembly. The FE's ``element_family``
    must match the mesh's ``element_family``; this is checked at
    :func:`build_dof_map` time.

    ``num_dofs_per_basis_fn`` is the components-per-basis-function
    count (3 for a vector displacement field, 1 for a scalar pressure
    / temperature field). Independent of the basis-fn topology owned
    by ``finite_element``.
    """

    name: str
    finite_element: FiniteElement
    num_dofs_per_basis_fn: int

    def __post_init__(self) -> None:
        if self.num_dofs_per_basis_fn < 1:
            raise ValueError(
                f"GlobalFieldLayout '{self.name}': num_dofs_per_basis_fn "
                f"must be >= 1; got {self.num_dofs_per_basis_fn}"
            )


@dataclass(frozen=True)
class _ResolvedBC:
    """Internal: BC paired with cached coords and pre-computed eq indices.

    ``set_coords`` is the ``(N_set, 3)`` slice of ``mesh.nodes`` for the
    BC's nodeset. Cached so re-evaluation of values is mesh-handle-free.

    ``eq_indices`` is the ``(N_set * len(dofs),)`` flat array of global
    equation numbers in (basis_fn-major, dof-minor) order to match the
    flatten convention used by
    :meth:`GlobalDofMap.evaluate_prescribed_values`.
    """

    bc: DirichletBC
    set_coords: NDArray[np.floating]
    eq_indices: NDArray[np.intp]


@dataclass(frozen=True)
class GlobalDofMap:
    """Global-field DOF map with formula-based eq numbering.

    ``field_layouts`` is the list of per-field layouts; index order is
    the field-major ordering of the global eq vector.

    ``block_offsets`` has shape ``(n_fields + 1,)``;
    ``block_offsets[i]`` is the starting eq index for field ``i``, and
    the total DOF count is ``block_offsets[-1]``.

    ``prescribed_indices`` is a flat ``(P,)`` int array of global eq
    numbers prescribed by Dirichlet BCs. ``num_free_dofs ==
    num_total_dofs - len(prescribed_indices)``.

    ``resolved_bcs`` holds the BC list with cached coords and per-BC
    eq_indices for time-dependent re-evaluation via
    :meth:`evaluate_prescribed_values`.
    """

    field_layouts: list[GlobalFieldLayout]
    block_offsets: NDArray[np.intp]
    prescribed_indices: NDArray[np.intp]
    resolved_bcs: list[_ResolvedBC]

    @property
    def num_total_dofs(self) -> int:
        return int(self.block_offsets[-1])

    @property
    def num_free_dofs(self) -> int:
        return self.num_total_dofs - len(self.prescribed_indices)

    @property
    def num_prescribed_dofs(self) -> int:
        return len(self.prescribed_indices)

    def eq_index(self, field_idx: int, basis_fn: int, dof: int) -> int:
        """Compute the global eq number for ``(field_idx, basis_fn, dof)``.

        Formulaic:
        ``eq = block_offsets[field_idx] + basis_fn * ndofs[field_idx] + dof``.
        Element scatter at the assembly layer computes the same expression
        vectorized per element.
        """
        layout = self.field_layouts[field_idx]
        return (
            int(self.block_offsets[field_idx])
            + basis_fn * layout.num_dofs_per_basis_fn
            + dof
        )

    def evaluate_prescribed_values(
        self, t: float = 0.0
    ) -> NDArray[np.floating]:
        """Materialize prescribed values at time ``t``.

        Returns a flat ``(P,)`` float array aligned with
        ``prescribed_indices``. Re-evaluates each BC's value source
        (``None`` / ``Sequence[float]`` / callable) using the cached set
        coordinates. Time-independent BCs ignore ``t``.
        """
        if len(self.prescribed_indices) == 0:
            return np.empty(0, dtype=np.float64)
        chunks: list[NDArray[np.floating]] = []
        for rbc in self.resolved_bcs:
            n_set = rbc.set_coords.shape[0]
            n_dofs = len(rbc.bc.dofs)
            values = rbc.bc.values
            if values is None:
                vals = np.zeros((n_set, n_dofs), dtype=np.float64)
            elif callable(values):
                vals_arr = np.asarray(values(rbc.set_coords, t))
                if vals_arr.shape != (n_set, n_dofs):
                    raise ValueError(
                        "DirichletBC values callable returned shape "
                        f"{vals_arr.shape}; expected {(n_set, n_dofs)}"
                    )
                vals = vals_arr.astype(np.float64)
            else:
                vals = np.broadcast_to(
                    np.asarray(values, dtype=np.float64), (n_set, n_dofs)
                ).copy()
            chunks.append(vals.ravel())
        return np.concatenate(chunks)


def _num_basis_fns_in_mesh(
        layout: GlobalFieldLayout, mesh: Mesh,
) -> int:
    """Total per-mesh basis-fn count for a field: sum over entity types
    of ``mesh.entity_count(et) * fe.dofs_per_entity[et]``.
    """
    return sum(
        mesh.entity_count(et) * count
        for et, count in layout.finite_element.dofs_per_entity.items()
    )


def build_dof_map(
    mesh: Mesh,
    field_layouts: list[GlobalFieldLayout],
    bcs: list[DirichletBC],
) -> GlobalDofMap:
    """Build a :class:`GlobalDofMap` from a mesh, field layouts, and BCs.

    Validates field-name uniqueness and per-field FE-family vs
    mesh-family agreement, computes per-field DOF block sizes by walking
    the FE's ``dofs_per_entity`` against the mesh's entity counts,
    resolves each BC's nodeset to coord-and-eq-index arrays, detects
    double-prescription, and assembles the global flat
    ``prescribed_indices`` array.

    Field-major ordering: ``block_offsets[0] = 0``, and
    ``block_offsets[i+1] = block_offsets[i] +
    n_basis_fns_for_field_i * field_layouts[i].num_dofs_per_basis_fn``,
    where ``n_basis_fns_for_field_i = sum(mesh.entity_count(et) *
    fe.dofs_per_entity[et])``.

    BCs are nodeset-keyed and resolve through the field's VERTEX DOFs:
    each mesh vertex in the nodeset contributes
    ``dofs_per_entity[VERTEX]`` basis fns to the constrained set. Fields
    whose FE has no VERTEX DOFs cannot be constrained via nodeset BCs;
    sideset-keyed BCs for non-vertex DOFs are not yet supported.

    Raises:
        ValueError: if field-layout names are not unique;
                    if a layout's FE element_family doesn't match the
                    mesh's element_family;
                    if a BC references an unknown field;
                    if a BC's dofs index is out of range for its field;
                    if a BC targets a field whose FE has no VERTEX
                    DOFs;
                    if two BCs prescribe the same
                    ``(field, basis_fn, dof)``.
        NotImplementedError: if a BC targets a field whose FE has
                    ``dofs_per_entity[VERTEX] > 1`` (multiplicity beyond
                    1 per vertex needs a richer BC API).
        KeyError:   if a BC references a nodeset name that's not in
                    ``mesh.node_sets``.
    """
    names = [fl.name for fl in field_layouts]
    if len(set(names)) != len(names):
        raise ValueError(
            f"GlobalFieldLayout names must be unique; got {names}"
        )

    for layout in field_layouts:
        fe_family = layout.finite_element.element_family
        if fe_family != mesh.element_family:
            raise ValueError(
                f"GlobalFieldLayout '{layout.name}': finite_element family "
                f"({fe_family.name}) does not match mesh element_family "
                f"({mesh.element_family.name})"
            )

    sizes = [
        _num_basis_fns_in_mesh(fl, mesh) * fl.num_dofs_per_basis_fn
        for fl in field_layouts
    ]
    block_offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.intp)

    name_to_idx = {fl.name: i for i, fl in enumerate(field_layouts)}
    resolved_bcs: list[_ResolvedBC] = []
    prescribed_set: set[tuple[int, int, int]] = set()
    eq_chunks: list[NDArray[np.intp]] = []

    for bc in bcs:
        if bc.field_name not in name_to_idx:
            raise ValueError(
                f"DirichletBC references unknown field "
                f"'{bc.field_name}'; known fields: {names}"
            )
        field_idx = name_to_idx[bc.field_name]
        layout = field_layouts[field_idx]
        fe = layout.finite_element

        for dof in bc.dofs:
            if dof < 0 or dof >= layout.num_dofs_per_basis_fn:
                raise ValueError(
                    f"DirichletBC for field '{bc.field_name}' references "
                    f"dof {dof} outside "
                    f"[0, {layout.num_dofs_per_basis_fn})"
                )

        if bc.nodeset_name not in mesh.node_sets:
            raise KeyError(
                f"DirichletBC references unknown nodeset "
                f"'{bc.nodeset_name}'; known nodesets: "
                f"{sorted(mesh.node_sets)}"
            )
        set_nodes = mesh.node_sets[bc.nodeset_name]

        dofs_per_vertex = fe.dofs_per_entity.get(EntityType.VERTEX, 0)
        if dofs_per_vertex == 0:
            raise ValueError(
                f"DirichletBC on field '{bc.field_name}' targets nodeset "
                f"'{bc.nodeset_name}', but the field's FiniteElement "
                f"'{fe.name}' has no VERTEX DOFs. Sideset-keyed BCs for "
                "non-vertex DOFs are not yet supported."
            )
        if dofs_per_vertex > 1:
            raise NotImplementedError(
                f"DirichletBC on field '{bc.field_name}': FiniteElement "
                f"'{fe.name}' has dofs_per_entity[VERTEX]={dofs_per_vertex} "
                "> 1; nodeset-keyed BCs only handle 1 DOF per vertex."
            )

        # dofs_per_vertex == 1: basis_fn[v] = v (identity).
        basis_fns_for_set = set_nodes.astype(np.intp)

        ndofs = layout.num_dofs_per_basis_fn
        block_off = int(block_offsets[field_idx])
        dofs_arr = np.asarray(list(bc.dofs), dtype=np.intp)
        eq_indices_2d = (
            block_off
            + basis_fns_for_set[:, None] * ndofs
            + dofs_arr[None, :]
        )
        eq_indices_flat = eq_indices_2d.ravel().astype(np.intp)

        for bf_idx in basis_fns_for_set:
            bf_int = int(bf_idx)
            for dof_val in dofs_arr:
                key = (field_idx, bf_int, int(dof_val))
                if key in prescribed_set:
                    raise ValueError(
                        "double-prescribed DOF "
                        f"(field='{bc.field_name}', basis_fn={bf_int}, "
                        f"dof={int(dof_val)}) appears in two BCs"
                    )
                prescribed_set.add(key)

        set_coords = mesh.nodes[set_nodes].astype(np.float64)

        resolved_bcs.append(
            _ResolvedBC(
                bc=bc,
                set_coords=set_coords,
                eq_indices=eq_indices_flat,
            )
        )
        eq_chunks.append(eq_indices_flat)

    if eq_chunks:
        prescribed_indices = np.concatenate(eq_chunks).astype(np.intp)
    else:
        prescribed_indices = np.empty(0, dtype=np.intp)

    return GlobalDofMap(
        field_layouts=list(field_layouts),
        block_offsets=block_offsets,
        prescribed_indices=prescribed_indices,
        resolved_bcs=resolved_bcs,
    )
