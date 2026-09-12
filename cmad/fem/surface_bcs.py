"""Per-side surface integral evaluators for the Neumann and Robin
conditions.

A :class:`~cmad.fem.bcs.NeumannBC` declares a surface flux on the
union of named sidesets; this module turns that declaration into an
``R -= ∫_∂Ω N · t̄ dA`` contribution scattered into the global
residual at the field's basis fns on the named sides. The pipeline:

1. :func:`resolve_neumann_bcs` — build-time walk: for each NBC, look
   up the field index, validate the FE has VERTEX-only DOFs with
   exactly 1 DOF per vertex, walk listed sidesets to collect
   ``(elem, local_side_id)`` pairs, group them by
   ``(family, local_side_id)``, and materialize sequence-form values
   into ndarrays.
2. :func:`build_neumann_side_arrays` — build-time precompute: for each
   NBC and ``(family, local_side_id)`` group, lift the side quadrature
   points to ref-volume coords via
   :func:`cmad.fem.topology.ref_side_lift`, evaluate the geometric and
   field interpolants there once, and form the per-element surface
   measure ``dA`` and physical side-IP coordinates ``coords_ip``, the
   mesh-uniform side shape values ``N_side`` and weights ``side_w``,
   and the flat scatter indices ``eq_flat`` — collected per group into
   a :class:`NeumannSideGroup`, the boundary-side analogue of the
   volume geometry cache in :mod:`cmad.fem.precompute`, hoisted out of
   the per-call kernel for the reasons given there.
3. :func:`per_side_neumann_R` — per-element side residual: contract the
   precomputed per-IP arrays into ``-N · t̄ · dA · w`` accumulated over
   IPs into a per-element buffer of shape ``(num_basis_fns,
   num_components)``.
4. :func:`assemble_side_neumann` — outer driver: iterates resolved
   NBCs and ``(family, local_side_id)`` groups, vmaps
   :func:`per_side_neumann_R` over the group's per-element surface
   arrays, and scatters per-element side residuals into the global
   residual vector at the precomputed equation indices.

Sign convention. ``R -= ∫_∂Ω N · t̄ dA`` follows the existing body-
force scatter at :func:`cmad.fem.assembly.per_element_R_and_K`,
which subtracts external forcing from R so the Newton driver solves
``K · dU = -R``. K is not contributed because explicit
``(coords, t)`` flux is U-independent.

Grouping. Within one ``(family, local_side_id)`` group, the family
fixes the volume interpolant and the side quadrature rule, and the
local_side_id fixes the lift ``(origin, tangents)``. Lifted ref-
volume coords are element-invariant 2D arrays that vmap broadcasts
over the group's element ids. Built-in StructuredHexMesh sidesets
(``xmin_sides`` etc.) are uniform-(family, local_side_id) so each
NBC over a built-in sideset produces exactly one group; user-
defined heterogeneous sidesets get one group per
(family, local_side_id) partition.

Cross-NBC overlaps are silent-additive: surface tractions superpose
linearly, so two NBCs sharing a side simply sum their contributions
into R. No consistency check (in contrast to DirichletBC).
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, vmap
from jax.tree_util import register_pytree_node_class
from numpy.typing import NDArray

from cmad.fem.bcs import NeumannBC, RobinBC
from cmad.fem.dof import GlobalDofMap
from cmad.fem.element_family import ElementFamily
from cmad.fem.finite_element import EntityType, FiniteElement
from cmad.fem.mesh import Mesh
from cmad.fem.quadrature import QuadratureRule
from cmad.fem.topology import ref_side_lift
from cmad.typing import JaxArray, Scalar


@register_pytree_node_class
@dataclass(frozen=True)
class NeumannSideShared:
    """Mesh-uniform per-side-IP arrays for one side group (``in_axes=None``).

    The boundary-side analogue of
    :class:`cmad.fem.precompute.BlockIPGeometryShared`. Shapes (``n_ip``
    side quadrature points, ``n_side_basis_fns`` field basis fns
    resident on the side):

    - ``N_side``: ``(n_ip, n_side_basis_fns)`` — field shape values
      restricted to the side's basis fns.
    - ``side_w``: ``(n_ip,)`` — side quadrature weights.
    """
    N_side: JaxArray
    side_w: JaxArray

    def tree_flatten(self) -> tuple[tuple[JaxArray, JaxArray], None]:
        return (self.N_side, self.side_w), None

    @classmethod
    def tree_unflatten(
            cls, aux_data: None, children: tuple[JaxArray, JaxArray],
    ) -> "NeumannSideShared":
        N_side, side_w = children
        return cls(N_side=N_side, side_w=side_w)


@register_pytree_node_class
@dataclass(frozen=True)
class NeumannSidePerElem:
    """Per-(side-element, IP) arrays for one side group (``in_axes=0``).

    The boundary-side analogue of
    :class:`cmad.fem.precompute.BlockIPGeometryPerElem`, plus the flat
    scatter indices ``eq_flat`` fused in so the surface kernel reads one
    structure per side group. Shapes (``ndims`` spatial dims,
    ``num_basis_fns`` field basis fns per element, ``num_components``
    field components):

    - ``dA``: ``(n_side_elems, n_ip)`` — per-IP side measure: the area
      element ``norm(cross(t0, t1))`` for a 3D face (two tangents), the
      length element ``norm(t0)`` for a 2D edge (one tangent).
    - ``coords_ip``: ``(n_side_elems, n_ip, ndims)`` — physical-frame
      side-IP coordinates.
    - ``eq_flat``: ``(n_side_elems, num_basis_fns * num_components)`` —
      flat global equation indices the per-element side residual
      scatters into.
    - ``k_scatter``: ``(n_side_elems, (num_basis_fns * num_components)^2)``
      positions in the deduped COO pattern of every pair of the element's
      equation numbers, where a Robin condition's side tangent is added;
      ``None`` for a Neumann condition, which has no tangent.
    """
    dA: JaxArray
    coords_ip: JaxArray
    eq_flat: JaxArray
    k_scatter: JaxArray | None = None

    def tree_flatten(
            self,
    ) -> tuple[tuple[JaxArray, JaxArray, JaxArray, JaxArray | None], None]:
        return (self.dA, self.coords_ip, self.eq_flat, self.k_scatter), None

    @classmethod
    def tree_unflatten(
            cls, aux_data: None,
            children: tuple[JaxArray, JaxArray, JaxArray, JaxArray | None],
    ) -> "NeumannSidePerElem":
        dA, coords_ip, eq_flat, k_scatter = children
        return cls(
            dA=dA, coords_ip=coords_ip, eq_flat=eq_flat, k_scatter=k_scatter,
        )


@register_pytree_node_class
@dataclass(frozen=True)
class NeumannSideGroup:
    """Per-side-group container: per-element + shared cached arrays.

    Registered as a JAX pytree; the children are the already-pytree
    :class:`NeumannSidePerElem` and :class:`NeumannSideShared`
    sub-structures. Mirrors
    :class:`cmad.fem.precompute.BlockIPGeometryCache`.
    """
    per_elem: NeumannSidePerElem
    shared: NeumannSideShared

    def tree_flatten(
            self,
    ) -> tuple[tuple[NeumannSidePerElem, NeumannSideShared], None]:
        return (self.per_elem, self.shared), None

    @classmethod
    def tree_unflatten(
            cls, aux_data: None,
            children: tuple[NeumannSidePerElem, NeumannSideShared],
    ) -> "NeumannSideGroup":
        per_elem, shared = children
        return cls(per_elem=per_elem, shared=shared)


NeumannSideArrays: TypeAlias = tuple[
    dict[tuple[ElementFamily, int], NeumannSideGroup], ...
]
"""Per-NBC cached side-assembly data, threaded as traced data.

One dict per resolved Neumann BC, in :func:`resolve_neumann_bcs`
order, keyed by ``(ElementFamily, local_side_id)`` to match
:attr:`ResolvedNeumannBC.elem_ids_by_side`. Each value is the
:class:`NeumannSideGroup` for that side group — the precomputed
surface geometry (``dA``, ``coords_ip``), the field shape values on
the side (``N_side``), the side quadrature weights (``side_w``), and
the flat scatter indices (``eq_flat``).
:func:`build_neumann_side_arrays` builds it;
:func:`assemble_side_neumann` consumes it.
"""


@dataclass(frozen=True)
class ResolvedNeumannBC:
    """Build-time-resolved data for one :class:`NeumannBC`.

    ``field_idx`` is the index into ``dof_map.field_layouts`` of the
    field this NBC contributes to. ``num_components`` matches
    ``dof_map.num_dofs_per_basis_fn[field_idx]``. ``finite_element``
    is the field's FE. ``elem_ids_by_side`` maps
    ``(ElementFamily, local_side_id)`` to a sorted-unique array of
    element ids that participate in the NBC under that side
    geometry. ``values`` is either the original callable (for
    expression-driven flux) or a constant ndarray of shape
    ``(num_components,)`` (for sequence-form flux).
    """

    field_idx: int
    num_components: int
    finite_element: FiniteElement
    elem_ids_by_side: dict[
        tuple[ElementFamily, int], NDArray[np.intp]
    ]
    values: (
        NDArray[np.floating]
        | Callable[
            [NDArray[np.floating] | JaxArray, Scalar],
            NDArray[np.floating] | JaxArray,
        ]
    )


@dataclass(frozen=True)
class ResolvedRobinBC:
    """Build-time-resolved data for one :class:`RobinBC`: the side
    geometry fields of :class:`ResolvedNeumannBC` and the flux callable."""

    field_idx: int
    num_components: int
    finite_element: FiniteElement
    elem_ids_by_side: dict[
        tuple[ElementFamily, int], NDArray[np.intp]
    ]
    flux: Callable[[JaxArray, JaxArray, Scalar], JaxArray]


def _resolve_side_groups(
        mesh: Mesh,
        dof_map: GlobalDofMap,
        field_name: str,
        sideset_names: Sequence[str],
        label: str,
) -> tuple[
    int, int, FiniteElement,
    dict[tuple[ElementFamily, int], NDArray[np.intp]],
]:
    """The side geometry of one surface BC: the field index and component
    count, the field's FE (VERTEX only, one DOF per vertex), and the
    element ids of every listed sideset grouped by ``(family,
    local_side_id)``."""
    name_to_field_idx = {
        fl.name: i for i, fl in enumerate(dof_map.field_layouts)
    }
    if field_name not in name_to_field_idx:
        raise ValueError(
            f"{label}.field_name='{field_name}' "
            f"has no matching GlobalFieldLayout (known: "
            f"{sorted(name_to_field_idx)})"
        )
    field_idx = name_to_field_idx[field_name]
    layout = dof_map.field_layouts[field_idx]
    fe = layout.finite_element

    non_vertex = sorted(
        et.name
        for et, count in fe.dofs_per_entity.items()
        if et != EntityType.VERTEX and count > 0
    )
    if non_vertex:
        raise NotImplementedError(
            f"{label} on field '{field_name}' "
            f"with FE '{fe.name}' has DOFs on {non_vertex} "
            "entities; side resolution requires VERTEX-only "
            "placement."
        )
    vertex_count = fe.dofs_per_entity.get(EntityType.VERTEX, 0)
    if vertex_count != 1:
        raise NotImplementedError(
            f"{label} on field '{field_name}' "
            f"with FE '{fe.name}' has dofs_per_entity[VERTEX]"
            f"={vertex_count}; side resolution requires exactly "
            "1 DOF per vertex."
        )

    num_components = int(dof_map.num_dofs_per_basis_fn[field_idx])

    known_sidesets = sorted(mesh.side_sets)
    elem_ids_by_side_lists: dict[
        tuple[ElementFamily, int], list[int]
    ] = {}
    for sideset_name in sideset_names:
        if sideset_name not in mesh.side_sets:
            raise ValueError(
                f"{label} sideset_name="
                f"'{sideset_name}' not in mesh.side_sets "
                f"(known: {known_sidesets})"
            )
        pairs = mesh.side_sets[sideset_name]
        for elem_id, local_side_id in pairs:
            key = (mesh.element_family, int(local_side_id))
            elem_ids_by_side_lists.setdefault(key, []).append(
                int(elem_id),
            )
    elem_ids_by_side = {
        k: np.unique(np.asarray(v, dtype=np.intp))
        for k, v in elem_ids_by_side_lists.items()
    }
    return field_idx, num_components, fe, elem_ids_by_side


def resolve_robin_bcs(
        mesh: Mesh,
        dof_map: GlobalDofMap,
        robin_bcs: Sequence[RobinBC],
) -> list[ResolvedRobinBC]:
    """Resolve a list of RobinBCs against a mesh + dof_map, the side
    geometry as for :func:`resolve_neumann_bcs`."""
    resolved: list[ResolvedRobinBC] = []
    for bc_idx, bc in enumerate(robin_bcs):
        field_idx, num_components, fe, elem_ids_by_side = (
            _resolve_side_groups(
                mesh, dof_map, bc.field_name, bc.sideset_names,
                f"RobinBC[{bc_idx}]",
            )
        )
        resolved.append(ResolvedRobinBC(
            field_idx=field_idx,
            num_components=num_components,
            finite_element=fe,
            elem_ids_by_side=elem_ids_by_side,
            flux=bc.flux,
        ))
    return resolved


def resolve_neumann_bcs(
        mesh: Mesh,
        dof_map: GlobalDofMap,
        neumann_bcs: Sequence[NeumannBC],
) -> list[ResolvedNeumannBC]:
    """Resolve a list of NeumannBCs against a mesh + dof_map.

    Validates each BC's ``field_name`` against
    ``dof_map.field_layouts``, requires the field's FE to have no
    edge / face / cell DOFs and exactly 1 DOF per vertex, walks
    listed sidesets to collect ``(elem, local_side_id)`` pairs
    grouped by ``(family, local_side_id)``, and materializes
    sequence-form values to ndarrays. Sequence values must have
    length equal to the resolved field's component count.
    """
    resolved: list[ResolvedNeumannBC] = []
    for nbc_idx, bc in enumerate(neumann_bcs):
        field_idx, num_components, fe, elem_ids_by_side = (
            _resolve_side_groups(
                mesh, dof_map, bc.field_name, bc.sideset_names,
                f"NeumannBC[{nbc_idx}]",
            )
        )

        values: (
            NDArray[np.floating]
            | Callable[
                [NDArray[np.floating] | JaxArray, Scalar],
                NDArray[np.floating] | JaxArray,
            ]
        )
        if callable(bc.values):
            values = bc.values
        else:
            arr = np.asarray(bc.values, dtype=np.float64)
            if arr.shape != (num_components,):
                raise ValueError(
                    f"NeumannBC[{nbc_idx}] values shape "
                    f"{tuple(arr.shape)} does not match field "
                    f"'{bc.field_name}' component count "
                    f"({num_components},)"
                )
            values = arr

        resolved.append(ResolvedNeumannBC(
            field_idx=field_idx,
            num_components=num_components,
            finite_element=fe,
            elem_ids_by_side=elem_ids_by_side,
            values=values,
        ))
    return resolved


def build_neumann_side_arrays(
        mesh: Mesh,
        dof_map: GlobalDofMap,
        resolved_neumann_bcs: Sequence[ResolvedNeumannBC | ResolvedRobinBC],
        side_quadrature: dict[ElementFamily, QuadratureRule],
        thickness: float | None = None,
        coo_pattern: tuple[NDArray[np.intp], NDArray[np.intp]] | None = None,
) -> NeumannSideArrays:
    """Precompute the per-NBC :class:`NeumannSideGroup` cache.

    Builds the cached surface geometry, side shape values, and scatter
    indices each side group needs; see the module docstring for the
    pipeline and the per-group contents. Returns an empty tuple when
    there are no Neumann BCs.

    ``dA`` is the unsigned side measure (area element from the tangent
    cross product for a 3D face, tangent length times the out of plane
    ``thickness`` for a 2D edge); the lift's orientation is preserved in
    ``(origin, tangents)`` for follower-load extensions that consume the
    signed normal.

    With ``coo_pattern``, the deduped ``(rows, cols)`` of the assembled
    tangent, each group also carries ``k_scatter``, the positions of
    every pair of the element's equation numbers in that pattern, for a
    Robin condition's side tangent.
    """
    if not resolved_neumann_bcs:
        return ()
    if coo_pattern is not None:
        coo_rows, coo_cols = coo_pattern
        n_dofs = dof_map.num_total_dofs
        pattern_keys = (
            coo_rows.astype(np.int64) * n_dofs + coo_cols.astype(np.int64)
        )
    if mesh.geometric_finite_element is None:
        raise ValueError(
            "Mesh.geometric_finite_element is required for "
            "Neumann surface assembly; mesh is malformed."
        )
    geom_interpolant_fn = mesh.geometric_finite_element.interpolant_fn

    per_nbc: list[
        dict[tuple[ElementFamily, int], NeumannSideGroup]
    ] = []
    for nbc in resolved_neumann_bcs:
        fe = nbc.finite_element
        field_interpolant_fn = fe.interpolant_fn
        num_components = nbc.num_components
        block_offset = int(dof_map.block_offsets[nbc.field_idx])
        k_arr = np.arange(num_components)
        group_arrays: dict[
            tuple[ElementFamily, int], NeumannSideGroup
        ] = {}
        for (family, local_side_id), elem_ids in (
                nbc.elem_ids_by_side.items()
        ):
            if family not in side_quadrature:
                raise ValueError(
                    f"side_quadrature has no rule for family "
                    f"{family.name}; required by NeumannBC on "
                    f"field '{fe.name}'"
                )
            sq = side_quadrature[family]
            side_xi = jnp.asarray(sq.xi)
            side_w = jnp.asarray(sq.w)

            origin_np, tangents_np = ref_side_lift(family, local_side_id)
            origin = jnp.asarray(origin_np)
            tangents = jnp.asarray(tangents_np)
            side_basis_fns = fe.side_basis_fns(local_side_id)

            connectivity_block = mesh.connectivity[elem_ids].astype(np.intp)
            X_block = jnp.asarray(mesh.nodes[connectivity_block])
            n_elems = connectivity_block.shape[0]

            xi_vol = origin[None, :] + side_xi @ tangents.T
            geom_shapes = vmap(geom_interpolant_fn)(xi_vol)
            field_shapes = vmap(field_interpolant_fn)(xi_vol)
            N_side = field_shapes.N[:, side_basis_fns]

            iso_jac = jnp.einsum(
                "eai,paj->epij", X_block, geom_shapes.grad_N,
            )
            surface_jac = jnp.einsum("epij,jm->epim", iso_jac, tangents)
            if tangents.shape[1] == 2:
                # 3D face: area element from the two tangent columns
                dA = jnp.linalg.norm(
                    jnp.cross(surface_jac[..., 0], surface_jac[..., 1]),
                    axis=-1,
                )
            else:
                # 2D edge: length of the single physical tangent column,
                # over the out of plane thickness
                dA = jnp.linalg.norm(surface_jac[..., 0], axis=-1) * (
                    1.0 if thickness is None else thickness
                )
            coords_ip = jnp.einsum("pa,eai->epi", geom_shapes.N, X_block)

            eq_3d = (
                block_offset
                + connectivity_block[:, :, None] * num_components
                + k_arr[None, None, :]
            )
            eq_per_elem = eq_3d.reshape(n_elems, -1).astype(np.int64)
            eq_flat = jnp.asarray(eq_per_elem)

            k_scatter = None
            if coo_pattern is not None:
                pair_keys = (
                    eq_per_elem[:, :, None] * n_dofs
                    + eq_per_elem[:, None, :]
                ).reshape(n_elems, -1)
                positions = np.searchsorted(pattern_keys, pair_keys)
                if not np.array_equal(pattern_keys[positions], pair_keys):
                    raise ValueError(
                        "a side's equation pair is missing from the "
                        "assembled tangent's pattern"
                    )
                k_scatter = jnp.asarray(positions)

            group_arrays[(family, local_side_id)] = NeumannSideGroup(
                per_elem=NeumannSidePerElem(
                    dA=dA, coords_ip=coords_ip, eq_flat=eq_flat,
                    k_scatter=k_scatter,
                ),
                shared=NeumannSideShared(N_side=N_side, side_w=side_w),
            )
        per_nbc.append(group_arrays)
    return tuple(per_nbc)


def per_side_neumann_R(
        dA_elem: JaxArray,
        coords_ip_elem: JaxArray,
        N_side: JaxArray,
        side_w: JaxArray,
        side_basis_fns: JaxArray,
        num_basis_fns: int,
        num_components: int,
        values_fn: Callable[
            [NDArray[np.floating] | JaxArray, Scalar],
            NDArray[np.floating] | JaxArray,
        ],
        t: Scalar,
) -> JaxArray:
    """Per-element side surface-flux contribution to R.

    Contracts the precomputed per-IP arrays for one side element — the
    surface measure ``dA_elem`` and physical coords ``coords_ip_elem``
    (per-element), the field shape values ``N_side`` and weights
    ``side_w`` (shared) — into an accumulator of shape
    ``(num_basis_fns, num_components)``; non-side basis fns stay zero.
    Vmap-over-elements compatible: only ``dA_elem`` / ``coords_ip_elem``
    carry the leading element axis. See the module docstring for the
    sign convention.
    """
    def per_ip(N_side_ip, w_ip, dA_ip, coords_ip_ip):
        t_bar = jnp.asarray(values_fn(coords_ip_ip[None, :], t))[0]
        return jnp.einsum("a,c->ac", N_side_ip, t_bar) * dA_ip * w_ip

    contrib_per_ip = vmap(per_ip)(
        N_side, side_w, dA_elem, coords_ip_elem,
    )
    contrib_total = contrib_per_ip.sum(axis=0)
    R_elem = jnp.zeros((num_basis_fns, num_components))
    return R_elem.at[side_basis_fns].add(-contrib_total)


def assemble_side_neumann(
        dof_map: GlobalDofMap,
        neumann_side_arrays: NeumannSideArrays,
        resolved_neumann_bcs: Sequence[ResolvedNeumannBC],
        t: Scalar,
) -> JaxArray:
    """Build the Neumann surface contribution to the global residual.

    Iterates each resolved NBC and its ``(family, local_side_id)``
    groups, vmaps :func:`per_side_neumann_R` over the group's
    precomputed per-element surface arrays, and scatters the per-element
    side residuals into a flat JAX vector of length
    ``dof_map.num_total_dofs`` at the cached ``eq_flat`` indices. K gets
    no contribution. Returns a zero vector when ``resolved_neumann_bcs``
    is empty.
    """
    n_dofs = dof_map.num_total_dofs
    R_neumann = jnp.zeros(n_dofs)
    if not resolved_neumann_bcs:
        return R_neumann

    for nbc, nbc_arrays in zip(
            resolved_neumann_bcs, neumann_side_arrays, strict=True,
    ):
        fe = nbc.finite_element
        num_basis_fns = fe.num_dofs_per_element
        num_components = nbc.num_components
        values_fn = _values_fn_for(nbc.values)

        for (_family, local_side_id), group in nbc_arrays.items():
            side_basis_fns = jnp.asarray(
                fe.side_basis_fns(local_side_id), dtype=jnp.int32,
            )
            shared = group.shared
            per_elem = group.per_elem

            side_kernel = partial(
                per_side_neumann_R,
                N_side=shared.N_side,
                side_w=shared.side_w,
                side_basis_fns=side_basis_fns,
                num_basis_fns=num_basis_fns,
                num_components=num_components,
                values_fn=values_fn,
                t=t,
            )
            R_per_elem = vmap(side_kernel)(
                per_elem.dA, per_elem.coords_ip,
            )

            n_elems = per_elem.dA.shape[0]
            R_flat = R_per_elem.reshape(n_elems, -1)
            R_neumann = R_neumann.at[per_elem.eq_flat.ravel()].add(
                R_flat.ravel(),
            )

    return R_neumann


def per_side_robin_R_and_K(
        U_elem_flat: JaxArray,
        dA_elem: JaxArray,
        coords_ip_elem: JaxArray,
        N_side: JaxArray,
        side_w: JaxArray,
        side_basis_fns: JaxArray,
        num_basis_fns: int,
        num_components: int,
        flux_fn: Callable[[JaxArray, JaxArray, Scalar], JaxArray],
        t: Scalar,
) -> tuple[JaxArray, JaxArray]:
    """Per-element side residual and tangent of a Robin condition.

    ``U_elem_flat`` holds the element's field values in ``eq_flat``
    order. The residual adds ``N flux(value, x, t) dA w`` over the side
    points to the side basis functions (an outward flux, so it is added
    where a prescribed inward flux is subtracted); the tangent is its
    derivative in ``U_elem_flat`` by ``jacfwd``, shaped
    ``(num_basis_fns * num_components,) * 2`` in the same order.
    """
    def residual(U_flat: JaxArray) -> JaxArray:
        U_side = U_flat.reshape(num_basis_fns, num_components)[side_basis_fns]

        def per_ip(N_side_ip, w_ip, dA_ip, coords_ip_ip):
            value = N_side_ip @ U_side
            q = jnp.asarray(flux_fn(value, coords_ip_ip, t)).reshape(
                num_components,
            )
            return jnp.einsum("a,c->ac", N_side_ip, q) * dA_ip * w_ip

        contrib_total = vmap(per_ip)(
            N_side, side_w, dA_elem, coords_ip_elem,
        ).sum(axis=0)
        R_elem = jnp.zeros((num_basis_fns, num_components))
        return R_elem.at[side_basis_fns].add(contrib_total)

    R_elem = residual(U_elem_flat)
    K_elem = jacfwd(residual)(U_elem_flat).reshape(
        num_basis_fns * num_components, -1,
    )
    return R_elem, K_elem


def assemble_side_robin(
        dof_map: GlobalDofMap,
        robin_side_arrays: NeumannSideArrays,
        resolved_robin_bcs: Sequence[ResolvedRobinBC],
        U_global: NDArray[np.floating] | JaxArray,
        t: Scalar,
) -> tuple[JaxArray, list[tuple[JaxArray, JaxArray]]]:
    """The Robin conditions' residual and tangent data.

    Returns the residual contribution as a flat vector of length
    ``dof_map.num_total_dofs`` and, per side group, the pair
    ``(k_scatter, K_per_elem)``: the pattern positions
    ``(n_side_elems, m * m)`` and the per element side tangents
    ``(n_side_elems, m, m)`` with ``m = num_basis_fns * num_components``,
    for the caller to add into the assembled tangent data or into the
    element tangent blocks. Both are empty (a zero vector, an empty
    list) without Robin conditions.
    """
    n_dofs = dof_map.num_total_dofs
    R_robin = jnp.zeros(n_dofs)
    tangents: list[tuple[JaxArray, JaxArray]] = []
    if not resolved_robin_bcs:
        return R_robin, tangents
    U_jax = jnp.asarray(U_global)

    for bc, bc_arrays in zip(
            resolved_robin_bcs, robin_side_arrays, strict=True,
    ):
        fe = bc.finite_element
        num_basis_fns = fe.num_dofs_per_element
        num_components = bc.num_components

        for (_family, local_side_id), group in bc_arrays.items():
            side_basis_fns = jnp.asarray(
                fe.side_basis_fns(local_side_id), dtype=jnp.int32,
            )
            shared = group.shared
            per_elem = group.per_elem
            assert per_elem.k_scatter is not None

            side_kernel = partial(
                per_side_robin_R_and_K,
                N_side=shared.N_side,
                side_w=shared.side_w,
                side_basis_fns=side_basis_fns,
                num_basis_fns=num_basis_fns,
                num_components=num_components,
                flux_fn=bc.flux,
                t=t,
            )
            R_per_elem, K_per_elem = vmap(side_kernel)(
                U_jax[per_elem.eq_flat], per_elem.dA, per_elem.coords_ip,
            )

            n_elems = per_elem.dA.shape[0]
            R_robin = R_robin.at[per_elem.eq_flat.ravel()].add(
                R_per_elem.reshape(n_elems, -1).ravel(),
            )
            tangents.append((per_elem.k_scatter, K_per_elem))

    return R_robin, tangents


def _values_fn_for(
        values: (
            NDArray[np.floating]
            | Callable[
                [NDArray[np.floating] | JaxArray, Scalar],
                NDArray[np.floating] | JaxArray,
            ]
        ),
) -> Callable[
    [NDArray[np.floating] | JaxArray, Scalar],
    NDArray[np.floating] | JaxArray,
]:
    """Wrap a ResolvedNeumannBC.values into a unified callable.

    Constant-form values broadcast across the leading point axis;
    callable-form values pass through unchanged. The unified shape
    lets :func:`per_side_neumann_R` consume both with one code path.
    """
    if callable(values):
        return values
    const_arr = jnp.asarray(values)

    def constant_values(
            coords: NDArray[np.floating] | JaxArray, t_arg: Scalar,
    ) -> NDArray[np.floating] | JaxArray:
        return jnp.broadcast_to(
            const_arr, (coords.shape[0], *const_arr.shape),
        )

    return constant_values
