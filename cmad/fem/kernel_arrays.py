"""Traced-argument carrier for the static FE assembly arrays.

Compiling the FE objective / gradient / Hessian closes the traced
assembly and solve kernels over the mesh-derived index arrays, the
reference-frame geometry cache, and the embedded-BC sparsity pattern.
Arrays closed over by traced code bake into the compiled XLA module as
constant literals, so the module size and XLA constant-folding cost
grow with the mesh even though the kernel op count does not.

:class:`FEKernelArrays` collects those arrays into one pytree-registered
carrier. Threaded through a ``jit`` boundary as a traced argument, the
compiled module holds only their shapes, so compile cost tracks the
(mesh-independent) op count rather than the literal bytes. The per-step
solution state — displacement and internal variables — is the
trajectory carry and is threaded separately; the carrier holds only
what is constant across the solve.

:func:`build_fe_kernel_arrays` builds the carrier once during
:class:`cmad.fem.fe_problem.FEProblem` construction; the result is
stored as ``fe_problem.kernel_arrays``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from cmad.fem.assembly import _element_eq_indices, assembled_coo_indices
from cmad.fem.neumann import NeumannSideArrays, build_neumann_side_arrays
from cmad.fem.precompute import BlockIPGeometryCache
from cmad.fem.sparse_solve import EmbeddedSparsity
from cmad.typing import JaxArray

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem


_FEKernelArraysChildren = tuple[
    dict[str, tuple[JaxArray, ...]],
    dict[str, tuple[JaxArray, ...]],
    JaxArray,
    JaxArray,
    dict[str, BlockIPGeometryCache],
    EmbeddedSparsity,
    JaxArray,
    NeumannSideArrays,
]


@register_pytree_node_class
@dataclass(frozen=True)
class FEKernelArrays:
    """Static mesh-derived arrays the traced FE kernels read, in one pytree.

    Every leaf is a JAX array — the fields are arrays, dicts of
    arrays, or pytree-registered dataclasses of arrays. The structure
    (dict keys, tuple lengths, the registered sub-dataclasses) is
    mesh-independent; only the leaf shapes vary with the mesh.

    Fields:

    - ``u_gather_eq_by_block``: per-element-block tuple of per-field
      U-gather index arrays. ``u_gather_eq_by_block[block][f]`` has
      shape ``(n_elems_block, num_dofs_per_element_f,
      num_dofs_per_basis_fn_f)`` and indexes the flat global ``U`` to
      gather field ``f``'s per-element basis coefficients.
    - ``r_scatter_eq_by_block``: per-element-block tuple of
      per-residual-block R-scatter index arrays.
      ``r_scatter_eq_by_block[block][r]`` has shape
      ``(n_elems_block, num_dofs_per_element * num_dofs_per_basis_fn)``
      and gives the flat global eq indices residual block ``r``
      scatters its element residual / tangent into.
    - ``coo_rows`` / ``coo_cols``: the assembled global tangent's COO
      ``(rows, cols)`` — mesh-derived constants shared by every Newton
      iteration (only the COO values vary).
    - ``geometry_cache``: the per-element-block reference-frame
      geometry cache; the same object as ``fe_problem.geometry_cache``.
    - ``embedded_sparsity``: the embedded-BC CSR sparsity cache; the
      same object as ``fe_problem.embedded_sparsity``.
    - ``prescribed_indices``: the flat global indices of the
      Dirichlet-prescribed dofs.
    - ``neumann_side_arrays``: the per-NBC Neumann surface-assembly
      arrays — element node coordinates and scatter equation indices,
      keyed per side group; see
      :data:`cmad.fem.neumann.NeumannSideArrays`.
    """
    u_gather_eq_by_block: dict[str, tuple[JaxArray, ...]]
    r_scatter_eq_by_block: dict[str, tuple[JaxArray, ...]]
    coo_rows: JaxArray
    coo_cols: JaxArray
    geometry_cache: dict[str, BlockIPGeometryCache]
    embedded_sparsity: EmbeddedSparsity
    prescribed_indices: JaxArray
    neumann_side_arrays: NeumannSideArrays

    def tree_flatten(self) -> tuple[_FEKernelArraysChildren, None]:
        children: _FEKernelArraysChildren = (
            self.u_gather_eq_by_block,
            self.r_scatter_eq_by_block,
            self.coo_rows,
            self.coo_cols,
            self.geometry_cache,
            self.embedded_sparsity,
            self.prescribed_indices,
            self.neumann_side_arrays,
        )
        return children, None

    @classmethod
    def tree_unflatten(
            cls, aux_data: None, children: _FEKernelArraysChildren,
    ) -> FEKernelArrays:
        (u_gather_eq_by_block, r_scatter_eq_by_block, coo_rows, coo_cols,
         geometry_cache, embedded_sparsity, prescribed_indices,
         neumann_side_arrays) = children
        return cls(
            u_gather_eq_by_block=u_gather_eq_by_block,
            r_scatter_eq_by_block=r_scatter_eq_by_block,
            coo_rows=coo_rows,
            coo_cols=coo_cols,
            geometry_cache=geometry_cache,
            embedded_sparsity=embedded_sparsity,
            prescribed_indices=prescribed_indices,
            neumann_side_arrays=neumann_side_arrays,
        )


def build_fe_kernel_arrays(fe_problem: FEProblem) -> FEKernelArrays:
    """Build the :class:`FEKernelArrays` carrier for ``fe_problem``.

    The per-block index arrays are derived through the assembly
    layer's own helper (:func:`cmad.fem.assembly._element_eq_indices`)
    and the COO ``(rows, cols)`` through
    :func:`cmad.fem.assembly.assembled_coo_indices`, so the carrier's
    arrays match the in-trace assembly path bit-for-bit; the Neumann
    side arrays come from
    :func:`cmad.fem.neumann.build_neumann_side_arrays`. The geometry
    cache and embedded-BC sparsity are referenced directly off
    ``fe_problem`` — the same objects, not copies.

    A pure construction-time builder: it touches only the static
    mesh-derived assembly data, never the trajectory's initial state.

    Called once from
    :meth:`cmad.fem.fe_problem.FEProblem.__post_init__` after the
    geometry cache and embedded sparsity have been built.
    """
    mesh = fe_problem.mesh
    dof_map = fe_problem.dof_map
    num_residuals = fe_problem.gr.num_residuals
    field_idx_per_block = fe_problem.field_idx_per_block
    num_fields = len(dof_map.field_layouts)

    u_gather_eq_by_block: dict[str, tuple[JaxArray, ...]] = {}
    r_scatter_eq_by_block: dict[str, tuple[JaxArray, ...]] = {}
    for block_name in fe_problem.evaluators_by_block:
        connectivity_block = mesh.connectivity[
            mesh.element_blocks[block_name]
        ]
        n_elems = connectivity_block.shape[0]

        u_gather_eqs: list[JaxArray] = []
        for field_idx in range(num_fields):
            ndofs = int(dof_map.num_dofs_per_basis_fn[field_idx])
            eq = _element_eq_indices(
                connectivity_block, dof_map, field_idx=field_idx,
            )
            u_gather_eqs.append(
                jnp.asarray(eq.reshape(n_elems, -1, ndofs)),
            )
        u_gather_eq_by_block[block_name] = tuple(u_gather_eqs)

        r_scatter_eq_by_block[block_name] = tuple(
            jnp.asarray(_element_eq_indices(
                connectivity_block, dof_map,
                field_idx=field_idx_per_block[r],
            ))
            for r in range(num_residuals)
        )

    rows, cols = assembled_coo_indices(fe_problem)
    neumann_side_arrays = build_neumann_side_arrays(
        mesh, dof_map, fe_problem.resolved_neumann_bcs,
    )

    return FEKernelArrays(
        u_gather_eq_by_block=u_gather_eq_by_block,
        r_scatter_eq_by_block=r_scatter_eq_by_block,
        coo_rows=jnp.asarray(rows),
        coo_cols=jnp.asarray(cols),
        geometry_cache=fe_problem.geometry_cache,
        embedded_sparsity=fe_problem.embedded_sparsity,
        prescribed_indices=jnp.asarray(dof_map.prescribed_indices),
        neumann_side_arrays=neumann_side_arrays,
    )
