"""Global field DOF management with formula-based equation numbering.

Block-major (field-major) DOF numbering across one or more global
fields. For field f at basis function i with component k::

    eq = block_offsets[f] + i * num_dofs_per_basis_fn[f] + k

No eq-table is stored; element scatter computes the eq inline. Free vs.
prescribed status is tracked separately as a flat ``prescribed_indices``
array, decoupling DOF numbering (structural) from BC enforcement
(configurable). The DofMap commits to no particular enforcement
strategy; consumers choose UF/UP reduction, symmetric strong
enforcement (zero rows + cols), or matrix-free apply against the same
metadata.

Per-field :class:`GlobalFieldLayout` carries a :class:`FiniteElement`
spec that says how many DOFs sit on each mesh entity type (vertex /
edge / face / cell). The DOF map allocates per-field DOFs by walking
the mesh's entity tables: ``mesh.entity_count(et) *
fe.dofs_per_entity[et]`` summed across entity types. P1 / Q1 fields
have ``{VERTEX: 1}`` and reduce to one basis-fn per mesh vertex.

Per-field component counts (3 for a vector displacement field, 1 for a
scalar pressure / temperature field) are owned by the GlobalResidual
(``gr._num_eqs[r]`` keyed by residual block) and supplied to
:func:`build_dof_map` via the ``components_by_field`` dict keyed by
field name. The DofMap stores the resulting per-field array as
``num_dofs_per_basis_fn`` for fast eq-formula lookup at scatter time.

Multi-field layouts naturally support fields without any BCs (those
fields contribute zero entries to ``prescribed_indices``).

Sideset-keyed BCs. Each :class:`~cmad.fem.bcs.DirichletBC` lists one
or more side-set names; resolution walks every ``(elem,
local_side_id)`` pair across the named sets, calls
:meth:`~cmad.fem.finite_element.FiniteElement.side_basis_fns` for the
per-element basis-fn indices on each side, gathers global vertex
basis-fns through ``mesh.connectivity``, and deduplicates per-BC.
Multi-sideset listing is the natural way to clamp a boundary patch
covering several side sets without spurious intra-BC double-
prescription on shared edges and corners.

Cross-BC overlaps are allowed when their values agree at the queried
time. ``prescribed_indices`` is the deduplicated sorted union of the
global equation numbers prescribed by all BCs (i.e., row indices
into the global residual / solution vector). The structurally
overprescribed subset — positions written by two or more BCs — is
computed once at build via a joint walk of every BC's ``eq_indices``
and stored on the dofmap as ``overprescribed_dbc_groups``. The
value-consistency check at
:meth:`GlobalDofMap.evaluate_prescribed_values` time iterates only
this small subset (per call, so time-dependent BCs that diverge
surface at the divergent step rather than at construction).

Prescribed values are evaluated on demand via
:meth:`GlobalDofMap.evaluate_prescribed_values`, supporting time-
dependent BCs and expression-parser-backed callables. Each resolved BC
caches its deduplicated boundary-vertex coordinate slice at
construction so re-evaluation is mesh-handle-free.
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

    Component count (the per-basis-function DOF multiplicity, 3 for a
    vector displacement field, 1 for a scalar pressure / temperature
    field) is owned by the GlobalResidual (``gr._num_eqs[r]``) and
    threaded into :func:`build_dof_map` via ``components_by_field``;
    it lives on :class:`GlobalDofMap` rather than on the layout.
    """

    name: str
    finite_element: FiniteElement


@dataclass(frozen=True)
class _ResolvedBC:
    """Internal: BC paired with cached coords and per-BC global
    equation indices.

    ``set_coords`` is the ``(N_set, 3)`` slice of ``mesh.nodes`` for
    the BC's deduplicated boundary-vertex set (after walking all
    sidesets in ``bc.sideset_names`` and unique-ing the gathered
    vertices). Cached so re-evaluation of values is mesh-handle-free.

    ``eq_indices`` is the ``(N_set * len(dofs),)`` flat array of
    global equation numbers (row indices into the global residual /
    solution vector) in (vertex-major, dof-minor) order to match the
    flatten convention used by
    :meth:`GlobalDofMap.evaluate_prescribed_values`.
    """

    bc: DirichletBC
    set_coords: NDArray[np.floating]
    eq_indices: NDArray[np.intp]


@dataclass(frozen=True)
class _OverprescribedDBCGroup:
    """Internal: one prescribed global equation written by two or more
    DirichletBCs.

    Records the structural overprescription identified at build time so
    the runtime value-consistency check can iterate this small set
    instead of every prescribed position. DBC-specific by name: Neumann
    BCs accumulate at assembly and don't fix DOFs, so they can't
    conflict in the same sense.

    ``position`` is an index into ``GlobalDofMap.prescribed_indices``;
    ``prescribed_indices[position]`` is the overprescribed global
    equation number.

    ``contributors`` lists ``(bc_idx, bc_eq_idx)`` pairs for every BC
    that writes this equation. ``bc_idx`` indexes into
    ``GlobalDofMap.resolved_bcs``; ``bc_eq_idx`` indexes into that
    BC's ``eq_indices`` array (and into the parallel materialized
    values array).
    """

    position: int
    contributors: list[tuple[int, int]]


@dataclass(frozen=True)
class GlobalDofMap:
    """Global-field DOF map with formula-based eq numbering.

    ``field_layouts`` is the list of per-field layouts; index order is
    the field-major ordering of the global eq vector.

    ``num_dofs_per_basis_fn`` is a length-``n_fields`` int array of
    per-field component counts (3 for a vector displacement field, 1
    for a scalar). Sourced from the ``components_by_field`` arg to
    :func:`build_dof_map` (which in turn comes from
    ``GlobalResidual._num_eqs``); stored here so the eq-formula
    lookup and element scatter don't have to round-trip through the
    GR.

    ``block_offsets`` has shape ``(n_fields + 1,)``;
    ``block_offsets[i]`` is the starting eq index for field ``i``, and
    the total DOF count is ``block_offsets[-1]``.

    ``prescribed_indices`` is a flat ``(P,)`` int array of global eq
    numbers prescribed by Dirichlet BCs, deduplicated across all BCs
    and sorted ascending. ``num_free_dofs == num_total_dofs -
    len(prescribed_indices)``.

    ``resolved_bcs`` holds the BC list with cached coords and per-BC
    flat ``eq_indices`` (global equation numbers). Used at runtime by
    :meth:`evaluate_prescribed_values` to materialize values at the
    queried ``t``.

    ``overprescribed_dbc_groups`` lists the structurally overprescribed
    positions — entries in ``prescribed_indices`` written by two or
    more BCs — and their contributors, computed once at build via a
    joint walk of every BC's ``eq_indices``. Typically small (often
    empty); the value-consistency check at
    :meth:`evaluate_prescribed_values` iterates only this set to
    validate cross-BC agreement at the queried ``t``.
    """

    field_layouts: list[GlobalFieldLayout]
    num_dofs_per_basis_fn: NDArray[np.intp]
    block_offsets: NDArray[np.intp]
    prescribed_indices: NDArray[np.intp]
    resolved_bcs: list[_ResolvedBC]
    overprescribed_dbc_groups: list[_OverprescribedDBCGroup]

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
        return int(
            self.block_offsets[field_idx]
            + basis_fn * self.num_dofs_per_basis_fn[field_idx]
            + dof
        )

    def _decode_eq(self, eq: int) -> tuple[int, int, int]:
        """Inverse of :meth:`eq_index`: decode ``eq`` into
        ``(field_idx, basis_fn, dof)``. Used by diagnostic messages
        when the consistency check fires.
        """
        field_idx = int(
            np.searchsorted(self.block_offsets, eq, side="right")
        ) - 1
        local_eq = eq - int(self.block_offsets[field_idx])
        ndofs = int(self.num_dofs_per_basis_fn[field_idx])
        return field_idx, local_eq // ndofs, local_eq % ndofs

    def evaluate_prescribed_values(
        self, t: float = 0.0
    ) -> NDArray[np.floating]:
        """Materialize prescribed values at time ``t``.

        Three-step path:

        1. Materialize each BC's values (``None`` / ``Sequence[float]``
           / callable) once at ``t`` via its cached deduplicated
           boundary-vertex coordinates. Time-independent BCs ignore
           ``t``.
        2. Bulk-scatter every BC into the prescribed-values array,
           locating each BC's entries via
           ``np.searchsorted(prescribed_indices, rbc.eq_indices)``.
           Last-writer-wins where multiple BCs target the same
           equation; step 3 confirms agreement.
        3. Validate ``overprescribed_dbc_groups`` (small build-time
           set): for each group, every contributing BC must agree at
           this ``t`` (``np.isclose`` with ``rtol=atol=1e-12`` to
           tolerate algebraic round-off across callable BCs);
           otherwise raises ``ValueError`` with a diagnostic naming
           the conflicting BCs and decoding the offending global
           equation to its ``(field, basis_fn, dof)`` triple.

        The validation set is identified once at build (structural)
        and reused on every call; only the value comparison is
        per-call, so time-dependent BCs that disagree at one ``t``
        but not another surface at the divergent step rather than at
        construction.
        """
        n_prescribed = len(self.prescribed_indices)
        if n_prescribed == 0:
            return np.empty(0, dtype=np.float64)

        bc_vals = [
            self._materialize_bc_values(rbc, t) for rbc in self.resolved_bcs
        ]

        values = np.empty(n_prescribed, dtype=np.float64)
        for rbc, vals in zip(self.resolved_bcs, bc_vals):
            positions = np.searchsorted(
                self.prescribed_indices, rbc.eq_indices,
            )
            values[positions] = vals

        for group in self.overprescribed_dbc_groups:
            ref_bc_idx, ref_bc_eq_idx = group.contributors[0]
            ref = float(bc_vals[ref_bc_idx][ref_bc_eq_idx])
            for bc_idx, bc_eq_idx in group.contributors[1:]:
                v = float(bc_vals[bc_idx][bc_eq_idx])
                if not np.isclose(v, ref, rtol=1e-12, atol=1e-12):
                    eq = int(self.prescribed_indices[group.position])
                    field_idx, basis_fn, dof = self._decode_eq(eq)
                    field_name = self.field_layouts[field_idx].name
                    ref_bc = self.resolved_bcs[ref_bc_idx].bc
                    cur_bc = self.resolved_bcs[bc_idx].bc
                    raise ValueError(
                        "DirichletBC inconsistent prescribed values "
                        f"at eq {eq} (field='{field_name}', "
                        f"basis_fn={basis_fn}, dof={dof}, t={t}): "
                        f"BC #{ref_bc_idx} (sideset_names="
                        f"{list(ref_bc.sideset_names)}) gives {ref}; "
                        f"BC #{bc_idx} (sideset_names="
                        f"{list(cur_bc.sideset_names)}) gives {v}"
                    )
        return values

    @staticmethod
    def _materialize_bc_values(
        rbc: _ResolvedBC, t: float,
    ) -> NDArray[np.floating]:
        """Evaluate one BC's value source into a flat
        ``(N_set * len(dofs),)`` array in (vertex-major, dof-minor)
        order, matching ``rbc.eq_indices``.
        """
        n_set = rbc.set_coords.shape[0]
        n_dofs = len(rbc.bc.dofs)
        bc_values = rbc.bc.values
        if bc_values is None:
            vals = np.zeros((n_set, n_dofs), dtype=np.float64)
        elif callable(bc_values):
            vals_arr = np.asarray(bc_values(rbc.set_coords, t))
            if vals_arr.shape != (n_set, n_dofs):
                raise ValueError(
                    "DirichletBC values callable returned shape "
                    f"{vals_arr.shape}; expected {(n_set, n_dofs)}"
                )
            vals = vals_arr.astype(np.float64)
        else:
            vals = np.broadcast_to(
                np.asarray(bc_values, dtype=np.float64), (n_set, n_dofs),
            ).copy()
        return vals.ravel()


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
    components_by_field: dict[str, int],
) -> GlobalDofMap:
    """Build a :class:`GlobalDofMap` from a mesh, field layouts, BCs, and
    per-field component counts.

    Validates field-name uniqueness and per-field FE-family vs
    mesh-family agreement, validates ``components_by_field`` against
    the field names, computes per-field DOF block sizes by walking
    the FE's ``dofs_per_entity`` against the mesh's entity counts,
    resolves each BC's side sets to deduplicated boundary-vertex
    coord-and-eq-index arrays, and assembles the global flat
    ``prescribed_indices`` array (deduplicated across BCs, sorted
    ascending).

    ``components_by_field`` maps each field name to its component
    multiplicity (3 for a vector displacement field, 1 for a scalar).
    The DofMap stores it as ``num_dofs_per_basis_fn`` (intp array,
    parallel to ``field_layouts``) rather than on the layout itself,
    so the GR-owned per-residual ``_num_eqs`` is the single source of
    truth for component counts.

    Field-major ordering: ``block_offsets[0] = 0``, and
    ``block_offsets[i+1] = block_offsets[i] +
    n_basis_fns_for_field_i * components_by_field[field_layouts[i].name]``,
    where ``n_basis_fns_for_field_i = sum(mesh.entity_count(et) *
    fe.dofs_per_entity[et])``.

    BC resolution. For each BC, walk every ``(elem, local_side_id)``
    pair across all listed ``sideset_names``; call the field's FE
    :meth:`~cmad.fem.finite_element.FiniteElement.side_basis_fns`
    per pair to get the per-element vertex slot indices on the side;
    gather global basis-fns via ``mesh.connectivity[elem, slots]``;
    deduplicate via ``np.unique`` (intra-BC). The deduplicated
    boundary-vertex set drives both ``set_coords`` (for value
    callables) and ``eq_indices``. Cross-BC: a joint
    ``argsort`` + ``np.unique(..., return_counts=True)`` over every
    BC's ``eq_indices`` produces ``prescribed_indices`` (deduplicated
    sorted union) and identifies the structurally overprescribed
    subset (positions with count > 1), recorded as
    :class:`_OverprescribedDBCGroup` entries on the dofmap with
    ``(bc_idx, bc_eq_idx)`` contributors. The value-consistency
    check between contributors fires at
    :meth:`GlobalDofMap.evaluate_prescribed_values` time (per call,
    since values can be ``t``-dependent), iterating only the
    overprescribed subset.

    BCs only resolve through the field's VERTEX DOFs: each boundary
    vertex contributes ``dofs_per_entity[VERTEX]`` basis fns to the
    constrained set. Fields whose FE has no VERTEX DOFs cannot be
    constrained via sideset BCs.

    Raises:
        ValueError: if field-layout names are not unique;
                    if a layout's FE element_family doesn't match the
                    mesh's element_family;
                    if ``components_by_field`` keys don't match the
                    field-layout names, or any value is < 1;
                    if a BC references an unknown field;
                    if a BC's dofs index is out of range for its field;
                    if a BC targets a field whose FE has no VERTEX
                    DOFs.
        NotImplementedError: if a BC targets a field whose FE has
                    ``dofs_per_entity[VERTEX] > 1`` (multiplicity beyond
                    1 per vertex needs a richer BC API).
        KeyError:   if a BC references a sideset name that's not in
                    ``mesh.side_sets``.
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

    if set(components_by_field.keys()) != set(names):
        raise ValueError(
            f"components_by_field keys ({sorted(components_by_field)}) "
            f"must match field-layout names ({sorted(names)})"
        )
    for fname, count in components_by_field.items():
        if count < 1:
            raise ValueError(
                f"components_by_field['{fname}']={count}; must be >= 1"
            )
    num_dofs_per_basis_fn = np.array(
        [components_by_field[fl.name] for fl in field_layouts],
        dtype=np.intp,
    )

    sizes = [
        _num_basis_fns_in_mesh(fl, mesh) * num_dofs_per_basis_fn[i]
        for i, fl in enumerate(field_layouts)
    ]
    block_offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.intp)

    name_to_idx = {fl.name: i for i, fl in enumerate(field_layouts)}
    resolved_bcs: list[_ResolvedBC] = []
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
        ndofs = num_dofs_per_basis_fn[field_idx]

        for dof in bc.dofs:
            if dof < 0 or dof >= ndofs:
                raise ValueError(
                    f"DirichletBC for field '{bc.field_name}' references "
                    f"dof {dof} outside [0, {ndofs})"
                )

        for sideset_name in bc.sideset_names:
            if sideset_name not in mesh.side_sets:
                raise KeyError(
                    f"DirichletBC references unknown sideset "
                    f"'{sideset_name}'; known sidesets: "
                    f"{sorted(mesh.side_sets)}"
                )

        dofs_per_vertex = fe.dofs_per_entity.get(EntityType.VERTEX, 0)
        if dofs_per_vertex == 0:
            raise ValueError(
                f"DirichletBC on field '{bc.field_name}' targets sidesets "
                f"{list(bc.sideset_names)}, but the field's FiniteElement "
                f"'{fe.name}' has no VERTEX DOFs. Sideset BCs only "
                "address vertex-anchored DOFs."
            )
        if dofs_per_vertex > 1:
            raise NotImplementedError(
                f"DirichletBC on field '{bc.field_name}': FiniteElement "
                f"'{fe.name}' has dofs_per_entity[VERTEX]={dofs_per_vertex} "
                "> 1; sideset BCs only handle 1 DOF per vertex."
            )

        # Walk every (elem, local_side_id) pair across all listed
        # sidesets; gather global basis-fns via the side resolver and
        # mesh.connectivity; deduplicate intra-BC.
        per_pair_basis_fns: list[NDArray[np.intp]] = []
        for sideset_name in bc.sideset_names:
            pairs = mesh.side_sets[sideset_name]
            for elem_id, local_side_id in pairs:
                local_basis_fns = fe.side_basis_fns(int(local_side_id))
                global_basis_fns = mesh.connectivity[
                    int(elem_id), local_basis_fns,
                ].astype(np.intp)
                per_pair_basis_fns.append(global_basis_fns)
        if per_pair_basis_fns:
            bc_basis_fns = np.unique(
                np.concatenate(per_pair_basis_fns)
            ).astype(np.intp)
        else:
            bc_basis_fns = np.empty(0, dtype=np.intp)

        block_offset = block_offsets[field_idx]
        dofs_arr = np.asarray(list(bc.dofs), dtype=np.intp)
        eq_indices_2d = (
            block_offset
            + bc_basis_fns[:, None] * ndofs
            + dofs_arr[None, :]
        )
        eq_indices_flat = eq_indices_2d.ravel().astype(np.intp)

        set_coords = mesh.nodes[bc_basis_fns].astype(np.float64)

        resolved_bcs.append(
            _ResolvedBC(
                bc=bc,
                set_coords=set_coords,
                eq_indices=eq_indices_flat,
            )
        )
        eq_chunks.append(eq_indices_flat)

    if eq_chunks:
        sizes = np.asarray([c.size for c in eq_chunks], dtype=np.intp)
        all_eqs = np.concatenate(eq_chunks)
        all_bc_idx = np.repeat(
            np.arange(len(eq_chunks), dtype=np.intp), sizes,
        )
        all_bc_eq_idx = np.concatenate(
            [np.arange(s, dtype=np.intp) for s in sizes]
        )
        order = np.argsort(all_eqs, kind="stable")
        sorted_bc_idx = all_bc_idx[order]
        sorted_bc_eq_idx = all_bc_eq_idx[order]
        prescribed_indices, group_starts, group_counts = np.unique(
            all_eqs[order], return_index=True, return_counts=True,
        )
        prescribed_indices = prescribed_indices.astype(np.intp)
        over_positions = np.flatnonzero(group_counts > 1)
        overprescribed_dbc_groups: list[_OverprescribedDBCGroup] = [
            _OverprescribedDBCGroup(
                position=int(p),
                contributors=[
                    (
                        int(sorted_bc_idx[group_starts[p] + k]),
                        int(sorted_bc_eq_idx[group_starts[p] + k]),
                    )
                    for k in range(int(group_counts[p]))
                ],
            )
            for p in over_positions
        ]
    else:
        prescribed_indices = np.empty(0, dtype=np.intp)
        overprescribed_dbc_groups = []

    return GlobalDofMap(
        field_layouts=list(field_layouts),
        num_dofs_per_basis_fn=num_dofs_per_basis_fn,
        block_offsets=block_offsets,
        prescribed_indices=prescribed_indices,
        resolved_bcs=resolved_bcs,
        overprescribed_dbc_groups=overprescribed_dbc_groups,
    )
