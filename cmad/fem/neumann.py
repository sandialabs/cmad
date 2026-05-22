"""Per-side surface-integral evaluator for Neumann BCs.

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
from jax import vmap
from jax.tree_util import register_pytree_node_class
from numpy.typing import NDArray

from cmad.fem.bcs import NeumannBC
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

    - ``dA``: ``(n_side_elems, n_ip)`` — surface measure
      ``norm(cross(surface_jac[:, 0], surface_jac[:, 1]))`` per IP.
    - ``coords_ip``: ``(n_side_elems, n_ip, ndims)`` — physical-frame
      side-IP coordinates.
    - ``eq_flat``: ``(n_side_elems, num_basis_fns * num_components)`` —
      flat global equation indices the per-element side residual
      scatters into.
    """
    dA: JaxArray
    coords_ip: JaxArray
    eq_flat: JaxArray

    def tree_flatten(
            self,
    ) -> tuple[tuple[JaxArray, JaxArray, JaxArray], None]:
        return (self.dA, self.coords_ip, self.eq_flat), None

    @classmethod
    def tree_unflatten(
            cls, aux_data: None,
            children: tuple[JaxArray, JaxArray, JaxArray],
    ) -> "NeumannSidePerElem":
        dA, coords_ip, eq_flat = children
        return cls(dA=dA, coords_ip=coords_ip, eq_flat=eq_flat)


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
    name_to_field_idx = {
        fl.name: i for i, fl in enumerate(dof_map.field_layouts)
    }
    resolved: list[ResolvedNeumannBC] = []
    for nbc_idx, bc in enumerate(neumann_bcs):
        if bc.field_name not in name_to_field_idx:
            raise ValueError(
                f"NeumannBC[{nbc_idx}].field_name='{bc.field_name}' "
                f"has no matching GlobalFieldLayout (known: "
                f"{sorted(name_to_field_idx)})"
            )
        field_idx = name_to_field_idx[bc.field_name]
        layout = dof_map.field_layouts[field_idx]
        fe = layout.finite_element

        non_vertex = sorted(
            et.name
            for et, count in fe.dofs_per_entity.items()
            if et != EntityType.VERTEX and count > 0
        )
        if non_vertex:
            raise NotImplementedError(
                f"NeumannBC[{nbc_idx}] on field '{bc.field_name}' "
                f"with FE '{fe.name}' has DOFs on {non_vertex} "
                "entities; side resolution requires VERTEX-only "
                "placement."
            )
        vertex_count = fe.dofs_per_entity.get(EntityType.VERTEX, 0)
        if vertex_count != 1:
            raise NotImplementedError(
                f"NeumannBC[{nbc_idx}] on field '{bc.field_name}' "
                f"with FE '{fe.name}' has dofs_per_entity[VERTEX]"
                f"={vertex_count}; side resolution requires exactly "
                "1 DOF per vertex."
            )

        num_components = int(dof_map.num_dofs_per_basis_fn[field_idx])

        known_sidesets = sorted(mesh.side_sets)
        elem_ids_by_side_lists: dict[
            tuple[ElementFamily, int], list[int]
        ] = {}
        for sideset_name in bc.sideset_names:
            if sideset_name not in mesh.side_sets:
                raise ValueError(
                    f"NeumannBC[{nbc_idx}] sideset_name="
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
        resolved_neumann_bcs: Sequence[ResolvedNeumannBC],
        side_quadrature: dict[ElementFamily, QuadratureRule],
) -> NeumannSideArrays:
    """Precompute the per-NBC :class:`NeumannSideGroup` cache.

    Builds the cached surface geometry, side shape values, and scatter
    indices each side group needs; see the module docstring for the
    pipeline and the per-group contents. Returns an empty tuple when
    there are no Neumann BCs.

    ``dA`` is the unsigned cross-product norm (the surface area
    element); the lift's outward orientation is preserved in
    ``(origin, tangents)`` for follower-load extensions that consume the
    signed normal.
    """
    if not resolved_neumann_bcs:
        return ()
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
            dA = jnp.linalg.norm(
                jnp.cross(surface_jac[..., 0], surface_jac[..., 1]),
                axis=-1,
            )
            coords_ip = jnp.einsum("pa,eai->epi", geom_shapes.N, X_block)

            eq_3d = (
                block_offset
                + connectivity_block[:, :, None] * num_components
                + k_arr[None, None, :]
            )
            eq_flat = jnp.asarray(eq_3d.reshape(n_elems, -1))

            group_arrays[(family, local_side_id)] = NeumannSideGroup(
                per_elem=NeumannSidePerElem(
                    dA=dA, coords_ip=coords_ip, eq_flat=eq_flat,
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
