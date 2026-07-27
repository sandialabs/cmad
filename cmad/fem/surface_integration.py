"""Cached per-facet geometry for integrating a field over a sideset.

For each facet on a sideset, holds the field shape values at the side
quadrature points, the surface area element, the side quadrature weights,
and the global equation numbers that gather the field's coefficients
there. Partitioned by ``(element family, local side id)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import vmap

from cmad.fem.dof import GlobalDofMap
from cmad.fem.element_family import ElementFamily
from cmad.fem.finite_element import EntityType
from cmad.fem.mesh import Mesh
from cmad.fem.quadrature import QuadratureRule
from cmad.fem.topology import ref_side_lift
from cmad.typing import JaxArray


@dataclass(frozen=True)
class SurfaceIntegrationGroup:
    """Cached arrays to integrate a field over one ``(family, side)`` facet set.

    Shapes use ``n_facets`` facets, ``n_ip`` side quadrature points,
    ``n_side_basis_fns`` field basis fns on the side, ``n_comp``
    components:

    - ``N_side``: ``(n_ip, n_side_basis_fns)`` field shape values at the
      side quadrature points.
    - ``side_w``: ``(n_ip,)`` side quadrature weights.
    - ``dA``: ``(n_facets, n_ip)`` surface area element.
    - ``eq``: ``(n_facets, n_side_basis_fns, n_comp)`` global equation
      numbers gathering the field's coefficients on each facet.
    """

    N_side: JaxArray
    side_w: JaxArray
    dA: JaxArray
    eq: JaxArray


def build_surface_integration_groups(
        mesh: Mesh,
        dof_map: GlobalDofMap,
        field_name: str,
        sideset_name: str,
        side_quadrature: dict[ElementFamily, QuadratureRule],
) -> list[SurfaceIntegrationGroup]:
    """Precompute per-facet surface geometry for ``field_name`` on a sideset.

    Partitions the sideset's ``(elem, local_side_id)`` pairs by
    ``(element family, local side id)``; per partition, lifts the side
    quadrature points to reference volume coordinates, evaluates the
    geometric and field interpolants, and forms the surface area element
    ``dA``, the side shape values ``N_side``, and the gather indices
    ``eq``.

    The field must place its DOFs on vertices only, one per vertex
    (P1 / Q1).
    """
    name_to_idx = {fl.name: i for i, fl in enumerate(dof_map.field_layouts)}
    if field_name not in name_to_idx:
        raise ValueError(
            f"field '{field_name}' has no GlobalFieldLayout (known: "
            f"{sorted(name_to_idx)})"
        )
    field_idx = name_to_idx[field_name]
    fe = dof_map.field_layouts[field_idx].finite_element
    num_components = int(dof_map.num_dofs_per_basis_fn[field_idx])
    block_offset = int(dof_map.block_offsets[field_idx])

    non_vertex = sorted(
        et.name
        for et, count in fe.dofs_per_entity.items()
        if et != EntityType.VERTEX and count > 0
    )
    if non_vertex:
        raise NotImplementedError(
            f"surface integration of field '{field_name}' with FE "
            f"'{fe.name}' needs VERTEX-only DOFs; found DOFs on {non_vertex}"
        )
    if fe.dofs_per_entity.get(EntityType.VERTEX, 0) != 1:
        raise NotImplementedError(
            f"surface integration of field '{field_name}' with FE "
            f"'{fe.name}' needs exactly 1 DOF per vertex"
        )

    if sideset_name not in mesh.side_sets:
        raise KeyError(
            f"sideset '{sideset_name}' not in mesh.side_sets (known: "
            f"{sorted(mesh.side_sets)})"
        )
    if mesh.geometric_finite_element is None:
        raise ValueError(
            "mesh.geometric_finite_element is required for surface "
            "integration; mesh is malformed"
        )
    geom_interpolant_fn = mesh.geometric_finite_element.interpolant_fn
    field_interpolant_fn = fe.interpolant_fn

    elem_ids_by_side: dict[tuple[ElementFamily, int], list[int]] = {}
    for elem_id, local_side_id in mesh.side_sets[sideset_name]:
        key = (mesh.element_family, int(local_side_id))
        elem_ids_by_side.setdefault(key, []).append(int(elem_id))

    k_arr = np.arange(num_components)
    groups: list[SurfaceIntegrationGroup] = []
    for (family, local_side_id), elem_list in elem_ids_by_side.items():
        if family not in side_quadrature:
            raise ValueError(
                f"side_quadrature has no rule for family {family.name}; "
                f"required by sideset '{sideset_name}'"
            )
        sq = side_quadrature[family]
        side_xi = jnp.asarray(sq.xi)
        side_w = jnp.asarray(sq.w)

        origin_np, tangents_np = ref_side_lift(family, local_side_id)
        origin = jnp.asarray(origin_np)
        tangents = jnp.asarray(tangents_np)
        side_basis_fns = fe.side_basis_fns(local_side_id)

        elem_ids = np.unique(np.asarray(elem_list, dtype=np.intp))
        connectivity_block = mesh.connectivity[elem_ids].astype(np.intp)
        X_block = jnp.asarray(mesh.nodes[connectivity_block])

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

        side_nodes = connectivity_block[:, side_basis_fns]
        eq = (
            block_offset
            + side_nodes[:, :, None] * num_components
            + k_arr[None, None, :]
        )

        groups.append(
            SurfaceIntegrationGroup(
                N_side=N_side,
                side_w=side_w,
                dA=dA,
                eq=jnp.asarray(eq),
            )
        )
    return groups
