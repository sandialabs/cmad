"""Element + global FE assembly machinery."""
from collections.abc import Callable, Sequence
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
from numpy.typing import NDArray

from cmad.fem.dof import GlobalDofMap
from cmad.fem.fe_problem import FEProblem
from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.typing import JaxArray, Params, StateList


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


def _zero_body_force(
        ndims: int,
) -> Callable[[JaxArray | NDArray[np.floating], float], JaxArray]:
    """Body-force callable emitting zeros at every IP.

    Threaded through :func:`per_element_residual` when no body force
    is configured; keeps the per-element call signature stable instead
    of branching on ``body_force_fn is None`` at trace time.
    """
    def _zero(_coords: JaxArray | NDArray[np.floating], _t: float) -> JaxArray:
        return jnp.zeros(ndims)
    return _zero


def _gather_element_U(
        U_global: NDArray[np.floating] | JaxArray,
        dof_map: GlobalDofMap,
        connectivity_block: NDArray[np.intp],
) -> list[JaxArray]:
    """Gather per-element basis-coefficient arrays from the flat global U.

    Returns one array per field in ``dof_map.field_layouts``; entry
    ``f`` has shape ``(n_elems, num_basis_fns_per_elem,
    num_dofs_per_basis_fn[f])`` ready for ``jax.vmap`` over the
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


def per_element_residual(
        X_elem: JaxArray,
        U_elem: Sequence[JaxArray],
        U_prev_elem: Sequence[JaxArray],
        params: Params,
        quad_xi: NDArray[np.floating] | JaxArray,
        quad_w: NDArray[np.floating] | JaxArray,
        interpolant_fn: Callable[[JaxArray], ShapeFunctionsAtIP],
        R_evaluator: Callable[..., JaxArray],
        body_force_fn: Callable[
            [JaxArray | NDArray[np.floating], float],
            JaxArray | NDArray[np.floating],
        ],
        t: float,
        xi_dummy: StateList,
        xi_prev_dummy: StateList,
) -> JaxArray:
    """Per-element residual contribution at all IPs of one element.

    For each IP: lift reference-frame shape gradients to physical-frame
    via :func:`iso_jac_at_ip`, call ``R_evaluator`` for the internal-
    force contribution, and subtract the body-force contribution
    ``f_ext = N · b · w · dv`` from residual block 0 (the displacement
    / momentum block). Returns shape ``(num_residuals, nnodes,
    num_eqs)`` summed across IPs.

    Vmap-over-elements compatible: only ``X_elem``, ``U_elem``,
    ``U_prev_elem`` carry the leading element axis; the rest are
    element-invariant. The Python-level IP loop unrolls under jit.
    ``ip_set=0`` is always passed to ``R_evaluator``; GRs with multiple
    ip_sets dispatch on the trailing arg inside their residual_fn
    body (see :data:`cmad.typing.ResidualFnGR`).
    """
    nnodes, ndims = X_elem.shape
    R_elem = jnp.zeros((1, nnodes, ndims))

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

        R_int = R_evaluator(
            xi_dummy, xi_prev_dummy, params, U_elem, U_prev_elem,
            [shapes_phys], w_ref, dv, 0,
        )
        R_elem = R_elem + R_int

        coords_ip = shapes_ref.N @ X_elem
        b_ip = jnp.asarray(body_force_fn(coords_ip, t))
        f_ext = jnp.einsum(
            "a,k->ak", shapes_ref.N, b_ip,
        ) * w_ref * dv
        R_elem = R_elem.at[0].add(-f_ext)

    return R_elem


def per_element_tangent(
        X_elem: JaxArray,
        U_elem: Sequence[JaxArray],
        U_prev_elem: Sequence[JaxArray],
        params: Params,
        quad_xi: NDArray[np.floating] | JaxArray,
        quad_w: NDArray[np.floating] | JaxArray,
        interpolant_fn: Callable[[JaxArray], ShapeFunctionsAtIP],
        dR_dU_evaluator: Callable[..., Sequence[JaxArray]],
        xi_dummy: StateList,
        xi_prev_dummy: StateList,
) -> JaxArray:
    """Per-element tangent ``K_elem = Σ_IPs dR_int/dU`` for a single U block.

    ``dR_dU_evaluator`` returns a list whose entry ``i`` is the per-
    U-block jacobian ``dR/dU[i]``; this function sums entry ``[0]``,
    sized ``(num_residuals, nnodes, num_eqs, nnodes_block_0,
    num_eqs_block_0)``. Multi-block assembly walks ``(residual_block,
    U_block)`` pairs and scatters them externally.

    Body force is independent of U and contributes nothing to the
    tangent. Vmap-over-elements convention matches
    :func:`per_element_residual`.
    """
    nnodes, ndims = X_elem.shape
    K_elem = jnp.zeros((1, nnodes, ndims, nnodes, ndims))

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

        K_ip_blocks = dR_dU_evaluator(
            xi_dummy, xi_prev_dummy, params, U_elem, U_prev_elem,
            [shapes_phys], w_ref, dv, 0,
        )
        K_elem = K_elem + K_ip_blocks[0]

    return K_elem


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

    Vmaps :func:`per_element_residual` and :func:`per_element_tangent`
    over the block's elements, brings the per-element results back to
    numpy, and emits ``(rows, cols, vals)`` ready for the global COO
    construction plus a flat ``R_block`` of shape
    ``(dof_map.num_total_dofs,)`` already scattered via ``np.add.at``.
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

    quad_rule = fe_problem.assembly_quadrature[mesh.element_family]
    interpolant_fn = fe_problem.interpolant_fn[mesh.element_family]
    body_force_fn = fe_problem.body_force_fn or _zero_body_force(
        fe_problem.ndims,
    )

    xi_dummy: StateList = [jnp.zeros_like(b) for b in model._init_xi]
    xi_prev_dummy: StateList = [jnp.zeros_like(b) for b in model._init_xi]

    quad_xi = jnp.asarray(quad_rule.xi)
    quad_w = jnp.asarray(quad_rule.w)

    R_eval = cast(Callable[..., JaxArray], evaluators["R"])
    dR_dU_eval = cast(
        Callable[..., Sequence[JaxArray]], evaluators["dR_dU"],
    )

    R_per_elem = jax.vmap(
        lambda X, U, Up: per_element_residual(
            X, U, Up, params,
            quad_xi, quad_w,
            interpolant_fn,
            R_eval,
            body_force_fn, t,
            xi_dummy, xi_prev_dummy,
        ),
    )(X_block, U_elem_block, U_prev_elem_block)

    K_per_elem = jax.vmap(
        lambda X, U, Up: per_element_tangent(
            X, U, Up, params,
            quad_xi, quad_w,
            interpolant_fn,
            dR_dU_eval,
            xi_dummy, xi_prev_dummy,
        ),
    )(X_block, U_elem_block, U_prev_elem_block)

    R_per_elem_np = np.asarray(R_per_elem)
    K_per_elem_np = np.asarray(K_per_elem)

    eq_indices_per_elem = _element_eq_indices(connectivity_block, dof_map)
    n_elems = connectivity_block.shape[0]
    n_dofs_elem = eq_indices_per_elem.shape[1]

    R_elem_flat = R_per_elem_np.reshape(n_elems, -1)

    rows = np.broadcast_to(
        eq_indices_per_elem[:, :, None],
        (n_elems, n_dofs_elem, n_dofs_elem),
    ).ravel()
    cols = np.broadcast_to(
        eq_indices_per_elem[:, None, :],
        (n_elems, n_dofs_elem, n_dofs_elem),
    ).ravel()
    vals = K_per_elem_np.reshape(
        n_elems, n_dofs_elem, n_dofs_elem,
    ).ravel()

    R_block = np.zeros(dof_map.num_total_dofs, dtype=np.float64)
    np.add.at(R_block, eq_indices_per_elem.ravel(), R_elem_flat.ravel())

    return rows, cols, vals, R_block


def assemble_global(
        fe_problem: FEProblem,
        U_global: NDArray[np.floating] | JaxArray,
        U_prev_global: NDArray[np.floating] | JaxArray,
        t: float,
) -> tuple[scipy.sparse.coo_matrix, NDArray[np.floating]]:
    """Walk all element blocks and emit the global ``(K_coo, R)`` pair.

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
