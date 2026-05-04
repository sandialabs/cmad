"""Per-element-block reference-frame geometry cache for FE assembly.

For total-Lagrangian formulations the per-(element, IP) geometry —
isoparametric Jacobian determinant, physical-frame field-shape
gradients, and IP coordinates — depends only on mesh geometry, the
geometric finite element, the per-residual-block field finite elements,
and the quadrature rule. Recomputing this every assembly call is
wasted work; this module hoists the computation up to ``FEProblem``
build time.

The cache splits into two parts to compose cleanly with ``jax.vmap``:

- :class:`BlockIPGeometryPerElem` — per-(element, IP) arrays. Vmaps
  with ``in_axes=0`` over the leading element axis.
- :class:`BlockIPGeometryShared` — mesh-uniform per-IP arrays (shape
  values that depend only on reference coordinates, plus quadrature
  weights). Identical for every element in the block; passed with
  ``in_axes=None``.

Both sub-structures are JAX pytrees (registered) so they traverse
correctly through ``vmap`` and other transforms.

Updated-Lagrangian / current-configuration kernels would have
solution-dependent geometry and bypass this cache; the current FE
pipeline only has total-Lagrangian residuals.

Memory: for a Q1 hex block with one u-residual block and 8 DOFs/elem
at 8 quadrature points, each element costs roughly
``n_ip * (1 + 3 + 8 * 3) * 8 B`` ≈ 1.8 KB. A 100k-element mesh fits
in ~180 MB; multi-million-element meshes may need a lighter cache
variant that drops the per-block lifted gradients.
"""
from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import vmap
from jax.tree_util import register_pytree_node_class

from cmad.fem.dof import GlobalFieldLayout
from cmad.fem.element_family import ElementFamily
from cmad.fem.mesh import Mesh
from cmad.fem.quadrature import QuadratureRule
from cmad.typing import JaxArray


@register_pytree_node_class
@dataclass(frozen=True)
class BlockIPGeometryPerElem:
    """Per-(element, IP) geometry for one mesh element block.

    All arrays carry a leading element axis of length ``n_b``. Designed
    to be passed to ``vmap`` with ``in_axes=0``; JAX recurses into the
    inner tuple ``field_grad_N_phys_per_block`` automatically.

    Shapes (``ndims = 3``, ``n_ip`` quadrature points, ``n_dofs_b_r``
    field DOFs for residual block ``r``):

    - ``iso_jac_det``: ``(n_b, n_ip)``
    - ``coords_ip``: ``(n_b, n_ip, ndims)``
    - ``field_grad_N_phys_per_block[r]``:
      ``(n_b, n_ip, n_dofs_b_r, ndims)``
    """
    iso_jac_det: JaxArray
    coords_ip: JaxArray
    field_grad_N_phys_per_block: tuple[JaxArray, ...]

    def tree_flatten(
            self,
    ) -> tuple[tuple[JaxArray, JaxArray, tuple[JaxArray, ...]], None]:
        children = (
            self.iso_jac_det,
            self.coords_ip,
            self.field_grad_N_phys_per_block,
        )
        return children, None

    @classmethod
    def tree_unflatten(
            cls,
            aux_data: None,
            children: tuple[JaxArray, JaxArray, tuple[JaxArray, ...]],
    ) -> "BlockIPGeometryPerElem":
        iso_jac_det, coords_ip, field_grad_N_phys_per_block = children
        return cls(
            iso_jac_det=iso_jac_det,
            coords_ip=coords_ip,
            field_grad_N_phys_per_block=tuple(field_grad_N_phys_per_block),
        )


@register_pytree_node_class
@dataclass(frozen=True)
class BlockIPGeometryShared:
    """Mesh-uniform per-IP shape values and quadrature weights.

    Identical for every element of the block (depends only on the
    reference quadrature points and the field interpolants), so passed
    to ``vmap`` with ``in_axes=None``. Field-shape *gradients* are
    physical-frame (depend on per-element ``inv(iso_jac)``), so they
    live on :class:`BlockIPGeometryPerElem` instead.

    Shapes:

    - ``quad_w``: ``(n_ip,)``
    - ``field_N_per_block[r]``: ``(n_ip, n_dofs_b_r)``
    """
    quad_w: JaxArray
    field_N_per_block: tuple[JaxArray, ...]

    def tree_flatten(
            self,
    ) -> tuple[tuple[JaxArray, tuple[JaxArray, ...]], None]:
        children = (self.quad_w, self.field_N_per_block)
        return children, None

    @classmethod
    def tree_unflatten(
            cls,
            aux_data: None,
            children: tuple[JaxArray, tuple[JaxArray, ...]],
    ) -> "BlockIPGeometryShared":
        quad_w, field_N_per_block = children
        return cls(
            quad_w=quad_w,
            field_N_per_block=tuple(field_N_per_block),
        )


@dataclass(frozen=True)
class BlockIPGeometryCache:
    """Per-element-block container: per-element + shared geometry."""
    per_elem: BlockIPGeometryPerElem
    shared: BlockIPGeometryShared


def precompute_block_geometry(
        mesh: Mesh,
        quadrature_by_family: dict[ElementFamily, QuadratureRule],
        field_layouts_per_block: Sequence[GlobalFieldLayout],
) -> dict[str, BlockIPGeometryCache]:
    """Build the per-element-block reference-frame geometry cache.

    For each element block in ``mesh.element_blocks``:

    1. Evaluate the geometric finite element's shape functions and
       reference-frame gradients at every quadrature point (mesh-
       uniform; one evaluation per IP).
    2. For each residual block, evaluate the corresponding field
       finite element's reference-frame shape functions and gradients
       at every quadrature point (also mesh-uniform).
    3. For each element in the block, form the isoparametric Jacobian
       ``iso_jac = X_elem.T @ grad_N_geom_ref`` and its determinant
       and inverse. Lift each residual block's field-shape gradients
       to the physical frame via ``inv(iso_jac)``. Compute physical-
       frame IP coordinates ``coords_ip = N_geom @ X_elem``.

    The geometric basis (``mesh.geometric_finite_element``) is used
    only for the Jacobian and ``coords_ip``; the per-block field bases
    (``field_layouts_per_block[r].finite_element``) provide the field
    shape values and reference gradients independently. This separation
    matches subparametric and other general formulations where the
    geometric basis differs from the field basis.

    ``iso_jac_det`` is propagated signed — negative values surface
    inverted elements as Newton divergence rather than silently
    absorbing them via ``abs(...)``. Mesh-orientation correctness is
    the mesh builder's responsibility.
    """
    assert mesh.geometric_finite_element is not None, (
        "Mesh.geometric_finite_element must be resolved before "
        "precomputing geometry; this is set in Mesh.__post_init__."
    )

    quad_rule = quadrature_by_family[mesh.element_family]
    quad_xi = jnp.asarray(quad_rule.xi)
    quad_w = jnp.asarray(quad_rule.w)

    geom_fn = mesh.geometric_finite_element.interpolant_fn
    geom_shapes_per_ip = vmap(geom_fn)(quad_xi)
    # geom_shapes_per_ip.N: (n_ip, n_geom_dofs)
    # geom_shapes_per_ip.grad_N: (n_ip, n_geom_dofs, ndims_ref)

    field_N_per_block: list[JaxArray] = []
    field_grad_N_ref_per_block: list[JaxArray] = []
    for layout in field_layouts_per_block:
        field_shapes = vmap(layout.finite_element.interpolant_fn)(quad_xi)
        field_N_per_block.append(field_shapes.N)
        field_grad_N_ref_per_block.append(field_shapes.grad_N)

    shared = BlockIPGeometryShared(
        quad_w=quad_w,
        field_N_per_block=tuple(field_N_per_block),
    )

    cache: dict[str, BlockIPGeometryCache] = {}
    for block_name, elem_indices in mesh.element_blocks.items():
        connectivity_block = mesh.connectivity[elem_indices]
        X_block = jnp.asarray(mesh.nodes[connectivity_block])
        # X_block: (n_b, n_geom_dofs, ndims)

        # iso_jac[e, p, i, j] = sum_a X_block[e, a, i] * grad_N_geom[p, a, j]
        iso_jac = jnp.einsum(
            "eai,paj->epij", X_block, geom_shapes_per_ip.grad_N,
        )
        iso_jac_det = jnp.linalg.det(iso_jac)
        iso_jac_inv = jnp.linalg.inv(iso_jac)

        # coords_ip[e, p, i] = sum_a N_geom[p, a] * X_block[e, a, i]
        coords_ip = jnp.einsum(
            "pa,eai->epi", geom_shapes_per_ip.N, X_block,
        )

        # field_grad_N_phys[r][e, p, n, i]
        #   = sum_j grad_N_field_ref[p, n, j] * iso_jac_inv[e, p, j, i]
        field_grad_N_phys_per_block = tuple(
            jnp.einsum(
                "pnj,epji->epni",
                grad_N_ref,
                iso_jac_inv,
            )
            for grad_N_ref in field_grad_N_ref_per_block
        )

        per_elem = BlockIPGeometryPerElem(
            iso_jac_det=iso_jac_det,
            coords_ip=coords_ip,
            field_grad_N_phys_per_block=field_grad_N_phys_per_block,
        )
        cache[block_name] = BlockIPGeometryCache(
            per_elem=per_elem, shared=shared,
        )

    return cache
