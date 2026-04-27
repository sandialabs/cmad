"""Element + global FE assembly machinery."""
from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
import scipy.sparse
from jax import vmap
from numpy.typing import NDArray

from cmad.fem.dof import GlobalDofMap
from cmad.fem.fe_problem import FEProblem
from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.typing import JaxArray, Params, RAndDRDUEvaluator, StateList


def iso_jac_at_ip(
        grad_N_ref: JaxArray, X_elem: JaxArray,
) -> tuple[JaxArray, JaxArray, JaxArray]:
    """Per-IP physical-frame shape gradients via the isoparametric Jacobian.

    Returns ``(grad_N_phys, iso_jac_det, iso_jac)`` where
    ``iso_jac[i, j] = ∂x_i/∂ξ_j``. With this convention
    ``inv(iso_jac)[j, i] = ∂ξ_j/∂x_i`` and the chain rule
    ``∂N_a/∂x_i = (∂N_a/∂ξ_j)(∂ξ_j/∂x_i)`` collapses to
    ``grad_N_phys = grad_N_ref @ inv(iso_jac)``.

    ``iso_jac_det`` is signed — a negative value indicates element
    inversion and is propagated, not absorbed via ``abs(...)``, so
    inverted elements surface as Newton divergence rather than silent
    garbage. Mesh-correctness is the mesh builder's responsibility.

    Naming ``iso_jac`` / ``iso_jac_det`` avoids collision with
    ``J = det(F)`` from continuum-mechanics kinematics.
    """
    iso_jac = X_elem.T @ grad_N_ref
    iso_jac_det = jnp.linalg.det(iso_jac)
    grad_N_phys = grad_N_ref @ jnp.linalg.inv(iso_jac)
    return grad_N_phys, iso_jac_det, iso_jac


def _gather_element_U(
        U_global: NDArray[np.floating] | JaxArray,
        dof_map: GlobalDofMap,
        connectivity_block: NDArray[np.intp],
) -> list[JaxArray]:
    """Gather per-element basis-coefficient arrays from the flat global U.

    Returns one array per field in ``dof_map.field_layouts``; entry
    ``f`` has shape ``(n_elems, num_basis_fns_per_elem,
    num_dofs_per_basis_fn[f])`` ready for ``vmap`` over the
    leading element axis. Each gather honors the layout's block
    offset, per-basis-function dof count, and any explicit
    ``basis_fn_to_vertex`` (identity by default).
    """
    U_jax = jnp.asarray(U_global)
    out: list[JaxArray] = []
    for field_idx, layout in enumerate(dof_map.field_layouts):
        ndofs = layout.num_dofs_per_basis_fn
        block_off = int(dof_map.block_offsets[field_idx])

        bf_per_elem = connectivity_block.astype(np.intp)
        if layout.basis_fn_to_vertex is not None:
            vertex_to_bf = {
                int(v): bf
                for bf, v in enumerate(layout.basis_fn_to_vertex)
            }
            bf_per_elem = np.vectorize(vertex_to_bf.__getitem__)(
                connectivity_block,
            ).astype(np.intp)

        k_arr = np.arange(ndofs)
        eq_3d = (
            block_off
            + bf_per_elem[:, :, None] * ndofs
            + k_arr[None, None, :]
        )
        out.append(U_jax[eq_3d])
    return out


def _element_eq_indices(
        connectivity_block: NDArray[np.intp],
        dof_map: GlobalDofMap,
        field_idx: int = 0,
) -> NDArray[np.intp]:
    """Per-element flat global eq indices for one field block.

    Returns shape ``(n_elems, num_basis_fns_per_elem *
    num_dofs_per_basis_fn)`` in ``(a, k)`` major-minor order — basis-
    function-outer, equation-inner — keeping element-stiffness sub-
    blocks contiguous in the COO scatter.
    """
    layout = dof_map.field_layouts[field_idx]
    ndofs = layout.num_dofs_per_basis_fn
    block_off = int(dof_map.block_offsets[field_idx])

    bf_per_elem = connectivity_block.astype(np.intp)
    if layout.basis_fn_to_vertex is not None:
        vertex_to_bf = {
            int(v): bf for bf, v in enumerate(layout.basis_fn_to_vertex)
        }
        bf_per_elem = np.vectorize(vertex_to_bf.__getitem__)(
            connectivity_block,
        ).astype(np.intp)

    n_elems = connectivity_block.shape[0]
    k_arr = np.arange(ndofs)
    eq = (
        block_off
        + bf_per_elem[:, :, None] * ndofs
        + k_arr[None, None, :]
    )
    return eq.reshape(n_elems, -1).astype(np.intp)


def per_element_R_and_K(
        X_elem: JaxArray,
        U_elem: Sequence[JaxArray],
        U_prev_elem: Sequence[JaxArray],
        params: Params,
        quad_xi: NDArray[np.floating] | JaxArray,
        quad_w: NDArray[np.floating] | JaxArray,
        interpolant_fn: Callable[[JaxArray], ShapeFunctionsAtIP],
        R_and_dR_dU_evaluator: RAndDRDUEvaluator,
        forcing_fns_by_block_idx: dict[int, Callable[
            [JaxArray | NDArray[np.floating], float],
            JaxArray | NDArray[np.floating],
        ]],
        residual_block_shapes: Sequence[tuple[int, int]],
        t: float,
        xi_dummy: StateList,
        xi_prev_dummy: StateList,
) -> tuple[list[JaxArray], list[list[JaxArray]]]:
    """Per-element ``(R_blocks, dR_dU_blocks)`` at all IPs of one element.

    For each IP: lift reference-frame shape gradients to physical-frame
    via :func:`iso_jac_at_ip`, call ``R_and_dR_dU_evaluator`` for the
    fused internal-force + tangent contribution, and accumulate.
    Subtract the forcing contribution ``f_ext_r = N · f_r · w · dv``
    from each residual block ``r`` listed in
    ``forcing_fns_by_block_idx`` (sparse — absent block_idx means no
    forcing on that block, e.g. the pressure block of a mixed u-p
    method).

    Returns ``(R_blocks, dR_dU_blocks)`` where ``R_blocks[r]`` has
    shape ``(num_basis_fns_r, num_eqs_r)`` and ``dR_dU_blocks[r][s]``
    has shape ``(num_basis_fns_r, num_eqs_r, num_basis_fns_s,
    num_eqs_s)``, summed across IPs. ``residual_block_shapes`` is the
    list of ``(num_basis_fns, num_eqs)`` tuples per residual block,
    used to allocate the accumulator buffers (typically sourced from
    ``gr.block_shapes``).

    Vmap-over-elements compatible: only ``X_elem``, ``U_elem``,
    ``U_prev_elem`` carry the leading element axis; the rest are
    element-invariant. The Python-level IP loop unrolls under jit.
    ``ip_set=0`` is always passed to the evaluator; GRs with multiple
    ip_sets dispatch on the trailing arg inside their residual_fn
    body (see :data:`cmad.typing.ResidualFnGR`).

    Single-element-family multi-block assumption: every residual block
    shares the same physical-frame shape functions
    ``ShapeFunctionsAtIP``. Multi-basis cases (Taylor-Hood Q2/Q1) will
    require per-block interpolant lookups in a future change.
    """
    num_blocks = len(residual_block_shapes)
    R_blocks = [
        jnp.zeros((nb, neq)) for nb, neq in residual_block_shapes
    ]
    dR_dU_blocks = [
        [
            jnp.zeros((nb_r, neq_r, nb_s, neq_s))
            for nb_s, neq_s in residual_block_shapes
        ]
        for nb_r, neq_r in residual_block_shapes
    ]

    nips = quad_xi.shape[0]
    for ip_idx in range(nips):
        shapes_ref = interpolant_fn(quad_xi[ip_idx])
        grad_N_phys, iso_jac_det, _ = iso_jac_at_ip(
            shapes_ref.grad_N, X_elem,
        )
        shapes_phys = ShapeFunctionsAtIP(
            N=shapes_ref.N, grad_N=grad_N_phys,
        )
        w_ref = quad_w[ip_idx]
        dv = iso_jac_det

        shapes_phys_per_block = [shapes_phys] * num_blocks
        R_ip, dR_dU_ip = R_and_dR_dU_evaluator(
            xi_dummy, xi_prev_dummy, params, U_elem, U_prev_elem,
            shapes_phys_per_block, w_ref, dv, 0,
        )

        for r in range(num_blocks):
            R_blocks[r] = R_blocks[r] + R_ip[r]
            for s in range(num_blocks):
                dR_dU_blocks[r][s] = dR_dU_blocks[r][s] + dR_dU_ip[r][s]

        if forcing_fns_by_block_idx:
            coords_ip = shapes_ref.N @ X_elem
            for block_idx, forcing_fn in forcing_fns_by_block_idx.items():
                f_ip = jnp.asarray(forcing_fn(coords_ip, t))
                f_ext = jnp.einsum(
                    "a,k->ak", shapes_ref.N, f_ip,
                ) * w_ref * dv
                R_blocks[block_idx] = R_blocks[block_idx] - f_ext

    return R_blocks, dR_dU_blocks


def assemble_element_block(
        fe_problem: FEProblem,
        block_name: str,
        U_global: NDArray[np.floating] | JaxArray,
        U_prev_global: NDArray[np.floating] | JaxArray,
        t: float,
) -> tuple[
    NDArray[np.intp],
    NDArray[np.intp],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Assemble one element block's COO triplets + R contribution.

    Vmaps :func:`per_element_R_and_K` over the block's elements,
    brings the per-element results back to numpy, and emits
    ``(rows, cols, vals)`` ready for the global COO construction
    plus a flat ``R_global_block`` of shape
    ``(dof_map.num_total_dofs,)`` already scattered via ``np.add.at``.

    Multi-residual-block GRs scatter via nested loops over
    ``(r, s)`` residual-block / U-block pairs; each pair scatters the
    appropriate sub-tangent into the global rows/cols using its
    block-specific eq indices, looked up from the GR's ``var_names[r]``
    against ``dof_map.field_layouts``. Single-block GRs are the
    degenerate ``r=s=0`` case.
    """
    mesh = fe_problem.mesh
    dof_map = fe_problem.dof_map
    elem_indices = mesh.element_blocks[block_name]
    connectivity_block = mesh.connectivity[elem_indices]
    X_block = jnp.asarray(mesh.nodes[connectivity_block])

    U_elem_block = _gather_element_U(U_global, dof_map, connectivity_block)
    U_prev_elem_block = _gather_element_U(
        U_prev_global, dof_map, connectivity_block,
    )

    model = fe_problem.models_by_block[block_name]
    params = model.parameters.values
    evaluators = fe_problem.evaluators_by_block[block_name]
    gr = fe_problem.gr
    block_shapes = gr.block_shapes
    num_blocks = len(block_shapes)

    quad_rule = fe_problem.assembly_quadrature[mesh.element_family]
    interpolant_fn = fe_problem.interpolant_fn[mesh.element_family]
    forcing_fns_by_block_idx = fe_problem.forcing_fns_by_block_idx or {}

    xi_dummy: StateList = [jnp.zeros_like(b) for b in model._init_xi]
    xi_prev_dummy: StateList = [jnp.zeros_like(b) for b in model._init_xi]

    quad_xi = jnp.asarray(quad_rule.xi)
    quad_w = jnp.asarray(quad_rule.w)

    R_per_elem_blocks, K_per_elem_blocks = vmap(
        lambda X, U, Up: per_element_R_and_K(
            X, U, Up, params,
            quad_xi, quad_w,
            interpolant_fn,
            evaluators["R_and_dR_dU"],
            forcing_fns_by_block_idx, block_shapes, t,
            xi_dummy, xi_prev_dummy,
        ),
    )(X_block, U_elem_block, U_prev_elem_block)

    name_to_field_idx = {
        layout.name: i for i, layout in enumerate(dof_map.field_layouts)
    }
    eq_indices_per_block: list[NDArray[np.intp]] = []
    for r in range(num_blocks):
        var_name = gr.var_names[r]
        assert var_name is not None, (
            f"GR var_names[{r}] is None; fully-initialized GRs must "
            f"populate every var_names entry"
        )
        eq_indices_per_block.append(
            _element_eq_indices(
                connectivity_block, dof_map,
                field_idx=name_to_field_idx[var_name],
            )
        )

    n_elems = connectivity_block.shape[0]

    R_global_block = np.zeros(dof_map.num_total_dofs, dtype=np.float64)
    for r in range(num_blocks):
        R_arr = np.asarray(R_per_elem_blocks[r])
        R_flat = R_arr.reshape(n_elems, -1)
        eq_r = eq_indices_per_block[r]
        np.add.at(R_global_block, eq_r.ravel(), R_flat.ravel())

    rows_all: list[NDArray[np.intp]] = []
    cols_all: list[NDArray[np.intp]] = []
    vals_all: list[NDArray[np.floating]] = []
    for r in range(num_blocks):
        eq_r = eq_indices_per_block[r]
        n_dofs_r = eq_r.shape[1]
        for s in range(num_blocks):
            eq_s = eq_indices_per_block[s]
            n_dofs_s = eq_s.shape[1]
            K_arr = np.asarray(K_per_elem_blocks[r][s])
            K_flat = K_arr.reshape(n_elems, n_dofs_r, n_dofs_s)
            rows = np.broadcast_to(
                eq_r[:, :, None],
                (n_elems, n_dofs_r, n_dofs_s),
            ).ravel()
            cols = np.broadcast_to(
                eq_s[:, None, :],
                (n_elems, n_dofs_r, n_dofs_s),
            ).ravel()
            rows_all.append(rows)
            cols_all.append(cols)
            vals_all.append(K_flat.ravel())

    return (
        np.concatenate(rows_all),
        np.concatenate(cols_all),
        np.concatenate(vals_all),
        R_global_block,
    )


def assemble_global(
        fe_problem: FEProblem,
        U_global: NDArray[np.floating] | JaxArray,
        U_prev_global: NDArray[np.floating] | JaxArray,
        t: float,
) -> tuple[scipy.sparse.coo_matrix, NDArray[np.floating]]:
    """Walk all element blocks and emit the global ``(K_coo, R)`` pair.

    Nonlinear-FE convention: ``K = dR/dU`` is the tangent stiffness and
    ``R(U) = R_int(U) - F_ext`` is the residual, with the body-force
    contribution folded into ``R`` at the per-element level (no separate
    ``F`` vector). The Newton driver in
    :func:`cmad.fem.nonlinear_solver.fe_newton_solve` solves
    ``K · dU = -R``; the linear ``K U = F`` form is the degenerate
    one-iter case for a U-linear residual.

    Each block's per-element residual and tangent are computed by
    :func:`assemble_element_block`; the global COO triplets are
    concatenated and summed into a single sparse matrix on
    ``coo_matrix`` construction (duplicates accumulate naturally on
    ``.tocsr()``). ``R`` is the flat residual sum across blocks.
    """
    n_dofs = fe_problem.dof_map.num_total_dofs
    rows_all: list[NDArray[np.intp]] = []
    cols_all: list[NDArray[np.intp]] = []
    vals_all: list[NDArray[np.floating]] = []
    R_global = np.zeros(n_dofs, dtype=np.float64)

    for block_name in fe_problem.evaluators_by_block:
        rows, cols, vals, R_block = assemble_element_block(
            fe_problem, block_name, U_global, U_prev_global, t,
        )
        rows_all.append(rows)
        cols_all.append(cols)
        vals_all.append(vals)
        R_global += R_block

    K_coo = scipy.sparse.coo_matrix(
        (np.concatenate(vals_all),
         (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(n_dofs, n_dofs),
    )
    return K_coo, R_global
