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
2. :func:`per_side_neumann_R` — per-element side residual: for one
   element + one side IP loop, lift the side IP to a ref-volume
   coord via :func:`cmad.fem.topology.ref_side_lift`, evaluate the
   volume interpolant for ``N`` and ``grad_N``, build the surface
   measure ``dA = |cross(J_surface[:, 0], J_surface[:, 1])|`` from
   the volume Jacobian and the lift's tangent matrix, and
   accumulate ``-N · t̄ · dA · w`` over IPs into a per-element
   buffer of shape ``(num_basis_fns, num_components)``.
3. :func:`assemble_side_neumann` — outer driver: iterates resolved
   NBCs and ``(family, local_side_id)`` groups, vmaps
   :func:`per_side_neumann_R` over the group's element ids, and
   scatters per-element side residuals into the caller's ``R_global``
   in place via the field's eq formula.

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

import jax.numpy as jnp
import numpy as np
from jax import vmap
from numpy.typing import NDArray

from cmad.fem.bcs import NeumannBC
from cmad.fem.dof import GlobalDofMap
from cmad.fem.element_family import ElementFamily
from cmad.fem.finite_element import EntityType, FiniteElement
from cmad.fem.mesh import Mesh
from cmad.fem.quadrature import QuadratureRule
from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.fem.topology import ref_side_lift
from cmad.typing import JaxArray


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
            [NDArray[np.floating] | JaxArray, float],
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
                [NDArray[np.floating] | JaxArray, float],
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


def per_side_neumann_R(
        X_elem: JaxArray,
        side_xi: JaxArray,
        side_w: JaxArray,
        origin: JaxArray,
        tangents: JaxArray,
        geom_interpolant_fn: Callable[
            [JaxArray], ShapeFunctionsAtIP,
        ],
        field_interpolant_fn: Callable[
            [JaxArray], ShapeFunctionsAtIP,
        ],
        side_basis_fns: JaxArray,
        num_basis_fns: int,
        num_components: int,
        values_fn: Callable[
            [NDArray[np.floating] | JaxArray, float],
            NDArray[np.floating] | JaxArray,
        ],
        t: float,
) -> JaxArray:
    """Per-element side surface-flux contribution to R.

    Returns an accumulator of shape ``(num_basis_fns,
    num_components)`` holding ``-∫_side N · t̄ dA`` distributed
    over side-resident basis fns; non-side basis fns stay zero.
    Vmap-over-elements compatible: only ``X_elem`` carries the
    leading element axis. The Python-level IP loop unrolls under
    jit.

    The lift ``(origin, tangents)`` carries an outward-oriented side
    parameterization; the surface measure
    ``dA = |cross(J_surface[:, 0], J_surface[:, 1])|`` is sign-
    irrelevant for explicit ``(coords, t)`` flux. For follower-load
    extensions consuming the unit normal, the sign is correct by
    construction of the lift table.

    Sign convention matches the body-force scatter in
    :func:`cmad.fem.assembly.per_element_R_and_K` — external loads
    subtract from R so the Newton driver solves ``K · dU = -R``.
    """
    R_elem = jnp.zeros((num_basis_fns, num_components))
    nips = side_xi.shape[0]
    for ip in range(nips):
        st = side_xi[ip]
        w_ip = side_w[ip]
        xi_volume = origin + tangents @ st
        geom_shapes = geom_interpolant_fn(xi_volume)
        field_shapes = field_interpolant_fn(xi_volume)
        iso_jac = X_elem.T @ geom_shapes.grad_N
        J_surface = iso_jac @ tangents
        cross = jnp.cross(J_surface[:, 0], J_surface[:, 1])
        dA = jnp.linalg.norm(cross)
        coords_ip = geom_shapes.N @ X_elem
        t_bar = jnp.asarray(values_fn(coords_ip[None, :], t))[0]
        N_side = field_shapes.N[side_basis_fns]
        contrib = jnp.einsum("a,c->ac", N_side, t_bar) * dA * w_ip
        R_elem = R_elem.at[side_basis_fns].add(-contrib)
    return R_elem


def assemble_side_neumann(
        R_global: NDArray[np.floating],
        mesh: Mesh,
        dof_map: GlobalDofMap,
        resolved_neumann_bcs: Sequence[ResolvedNeumannBC],
        side_quadrature: dict[ElementFamily, QuadratureRule],
        t: float,
) -> None:
    """Scatter Neumann surface contributions into ``R_global`` in place.

    Iterates each resolved NBC and its ``(family, local_side_id)``
    groups, vmaps :func:`per_side_neumann_R` over the group's
    element ids, and scatters per-element side residuals into
    ``R_global`` via the field's eq formula
    ``eq = block_offset + basis_fn * num_components + component``.
    K gets no contribution. ``R_global`` is mutated in place via
    ``np.add.at``; no-op when ``resolved_neumann_bcs`` is empty.
    """
    if not resolved_neumann_bcs:
        return
    if mesh.geometric_finite_element is None:
        raise ValueError(
            "Mesh.geometric_finite_element is required for "
            "Neumann surface assembly; mesh is malformed."
        )
    geom_interpolant_fn = mesh.geometric_finite_element.interpolant_fn

    for nbc in resolved_neumann_bcs:
        fe = nbc.finite_element
        field_interpolant_fn = fe.interpolant_fn
        num_basis_fns = fe.num_dofs_per_element
        num_components = nbc.num_components
        block_offset = int(dof_map.block_offsets[nbc.field_idx])

        values_fn = _values_fn_for(nbc.values)

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

            origin_np, tangents_np = ref_side_lift(
                family, local_side_id,
            )
            origin = jnp.asarray(origin_np)
            tangents = jnp.asarray(tangents_np)

            side_basis_fns_np = fe.side_basis_fns(local_side_id)
            side_basis_fns = jnp.asarray(
                side_basis_fns_np, dtype=jnp.int32,
            )

            connectivity_block = mesh.connectivity[elem_ids]
            X_block = jnp.asarray(mesh.nodes[connectivity_block])

            side_kernel = partial(
                per_side_neumann_R,
                side_xi=side_xi,
                side_w=side_w,
                origin=origin,
                tangents=tangents,
                geom_interpolant_fn=geom_interpolant_fn,
                field_interpolant_fn=field_interpolant_fn,
                side_basis_fns=side_basis_fns,
                num_basis_fns=num_basis_fns,
                num_components=num_components,
                values_fn=values_fn,
                t=t,
            )
            R_per_elem = vmap(side_kernel)(X_block)

            n_elems = elem_ids.shape[0]
            k_arr = np.arange(num_components)
            bf_per_elem = connectivity_block.astype(np.intp)
            eq_3d = (
                block_offset
                + bf_per_elem[:, :, None] * num_components
                + k_arr[None, None, :]
            )
            eq_flat = eq_3d.reshape(n_elems, -1)

            R_arr = np.asarray(R_per_elem)
            R_flat = R_arr.reshape(n_elems, -1)
            np.add.at(R_global, eq_flat.ravel(), R_flat.ravel())


def _values_fn_for(
        values: (
            NDArray[np.floating]
            | Callable[
                [NDArray[np.floating] | JaxArray, float],
                NDArray[np.floating] | JaxArray,
            ]
        ),
) -> Callable[
    [NDArray[np.floating] | JaxArray, float],
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
            coords: NDArray[np.floating] | JaxArray, t_arg: float,
    ) -> NDArray[np.floating] | JaxArray:
        return jnp.broadcast_to(
            const_arr, (coords.shape[0], *const_arr.shape),
        )

    return constant_values
