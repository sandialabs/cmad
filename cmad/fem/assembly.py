"""Element + global FE assembly machinery."""
from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.dof import GlobalDofMap
from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.typing import JaxArray, Params, StateList


def iso_jac_at_ip(
        grad_N_ref: JaxArray, X_elem: JaxArray,
) -> tuple[JaxArray, JaxArray, JaxArray]:
    """Per-IP physical-frame shape gradients via the isoparametric Jacobian.

    Returns ``(grad_N_phys, iso_jac_det, iso_jac)`` where
    ``iso_jac[j, i] = ∂x_i/∂ξ_j`` (transpose-of-classical convention,
    chosen so ``grad_N_phys = grad_N_ref @ inv(iso_jac)``).
    ``iso_jac_det`` is signed — a negative value indicates element
    inversion and is propagated, not absorbed via ``abs(...)``, so
    inverted elements surface as Newton divergence rather than silent
    garbage. Mesh-correctness is the mesh builder's responsibility.

    Naming ``iso_jac`` / ``iso_jac_det`` avoids collision with
    ``J = det(F)`` from continuum-mechanics kinematics.
    """
    iso_jac = grad_N_ref.T @ X_elem
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
