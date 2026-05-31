"""Element + global FE assembly machinery."""
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import checkpoint, lax, vmap
from jax.experimental.sparse import BCOO
from jax.flatten_util import ravel_pytree
from numpy.typing import NDArray

from cmad.fem.dof import GlobalDofMap, GlobalFieldLayout
from cmad.fem.fe_problem import FEProblem
from cmad.fem.finite_element import EntityType
from cmad.fem.neumann import assemble_side_neumann
from cmad.fem.precompute import (
    BlockIPGeometryPerElem,
    BlockIPGeometryShared,
)
from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.typing import (
    JaxArray,
    Params,
    RAndDRDUAndXiEvaluator,
    RAndDRDUEvaluator,
    REvaluator,
    Scalar,
    StateList,
)

if TYPE_CHECKING:
    from cmad.fem.kernel_arrays import FEKernelArrays


def params_by_block_from_models(
        fe_problem: FEProblem,
) -> Mapping[str, Params]:
    """Source per-block params from each block's stored model.

    Imperative call sites (driver, MMS helpers, regression tests) build
    ``params_by_block`` from the model objects via this helper. AD
    callers construct ``params_by_block`` directly from the tracer
    input — see :class:`cmad.objectives.mp_jvp_objective.MPJVPObjective`
    for the reshape-from-flat pattern that produces a tracer-leaved
    PyTree to thread through the assembly call chain.
    """
    return {
        block_name: model.parameters.values
        for block_name, model in fe_problem.models_by_block.items()
    }


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


def _element_basis_fns(
        layout: GlobalFieldLayout,
        connectivity_block: NDArray[np.intp],
) -> NDArray[np.intp]:
    """Per-element basis-function global indices for a field.

    For VERTEX-only DOF placement (P1 / Q1 and other vertex-anchored
    fields), returns connectivity-derived indices:
    ``basis_fn[elem, slot] = connectivity[elem, slot //
    dofs_per_vertex] * dofs_per_vertex + (slot % dofs_per_vertex)``.

    Edge / face / cell DOF placement is not yet supported in this
    assembly path; fields with non-VERTEX DOFs raise
    ``NotImplementedError``. Mixed-basis support layers in alongside
    sideset-keyed BCs.
    """
    fe = layout.finite_element
    non_vertex = [
        et for et, count in fe.dofs_per_entity.items()
        if et != EntityType.VERTEX and count > 0
    ]
    if non_vertex:
        names = [et.name for et in non_vertex]
        raise NotImplementedError(
            f"GlobalFieldLayout '{layout.name}' has DOFs on {names} "
            "entities; assembly currently supports VERTEX DOFs only."
        )
    dofs_per_vertex = fe.dofs_per_entity.get(EntityType.VERTEX, 0)
    if dofs_per_vertex == 0:
        raise NotImplementedError(
            f"GlobalFieldLayout '{layout.name}' has no VERTEX DOFs; "
            "all-non-vertex DOF gather is not supported."
        )
    n_elems, n_vertices_per_elem = connectivity_block.shape
    m_arr = np.arange(dofs_per_vertex)
    bf_per_elem = (
        connectivity_block.astype(np.intp)[:, :, None] * dofs_per_vertex
        + m_arr[None, None, :]
    ).reshape(n_elems, n_vertices_per_elem * dofs_per_vertex)
    return bf_per_elem


def _gather_element_U(
        U_global: NDArray[np.floating] | JaxArray,
        fe_arrays: "FEKernelArrays",
        block_name: str,
) -> list[JaxArray]:
    """Gather per-element basis-coefficient arrays from the flat global U.

    Returns one array per field; entry ``f`` has shape
    ``(n_elems, num_dofs_per_element_f, num_dofs_per_basis_fn_f)``,
    ready for ``vmap`` over the leading element axis. The per-field
    U-gather index arrays come from
    ``fe_arrays.u_gather_eq_by_block[block_name]`` rather than being
    derived from ``mesh.connectivity`` in-trace, so they enter the
    compiled program as traced array shapes, not baked constants.
    """
    U_jax = jnp.asarray(U_global)
    return [
        U_jax[eq] for eq in fe_arrays.u_gather_eq_by_block[block_name]
    ]


def _element_eq_indices(
        connectivity_block: NDArray[np.intp],
        dof_map: GlobalDofMap,
        field_idx: int,
) -> NDArray[np.intp]:
    """Per-element flat global eq indices for one field block.

    Returns shape ``(n_elems, num_dofs_per_element *
    num_dofs_per_basis_fn)`` in ``(basis_fn, component)`` major-minor
    order — basis-function-outer, component-inner — keeping element-
    stiffness sub-blocks contiguous in the COO scatter.
    """
    layout = dof_map.field_layouts[field_idx]
    bf_per_elem = _element_basis_fns(layout, connectivity_block)
    ndofs = dof_map.num_dofs_per_basis_fn[field_idx]
    block_offset = dof_map.block_offsets[field_idx]
    n_elems = connectivity_block.shape[0]
    k_arr = np.arange(ndofs)
    eq = (
        block_offset
        + bf_per_elem[:, :, None] * ndofs
        + k_arr[None, None, :]
    )
    return eq.reshape(n_elems, -1).astype(np.intp)


def _zero_R_and_dR_dU_accumulators(
        residual_block_shapes: Sequence[tuple[int, int]],
) -> tuple[list[JaxArray], list[list[JaxArray]]]:
    """Zero ``(R_blocks, dR_dU_blocks)`` accumulators for the per-IP scan.

    ``R_blocks[r]`` is shaped ``residual_block_shapes[r]``;
    ``dR_dU_blocks[r][s]`` is shaped ``residual_block_shapes[r] +
    residual_block_shapes[s]``. Seeds the :func:`jax.lax.scan` carry
    that sums per-IP contributions in :func:`per_element_R_and_K` and
    :func:`per_element_R_and_K_coupled`.
    """
    num_blocks = len(residual_block_shapes)
    R_acc = [
        jnp.zeros(residual_block_shapes[r]) for r in range(num_blocks)
    ]
    dR_dU_acc = [
        [
            jnp.zeros(residual_block_shapes[r] + residual_block_shapes[s])
            for s in range(num_blocks)
        ]
        for r in range(num_blocks)
    ]
    return R_acc, dR_dU_acc


def _accumulate_ip(
        R_acc: list[JaxArray],
        dR_dU_acc: list[list[JaxArray]],
        R_ip: Sequence[JaxArray],
        dR_dU_ip: Sequence[Sequence[JaxArray]],
        body_force_ip_per_block: Mapping[int, JaxArray],
) -> tuple[list[JaxArray], list[list[JaxArray]]]:
    """Sum one IP's ``(R, dR_dU)`` into the per-IP scan accumulators.

    Adds ``R_ip`` / ``dR_dU_ip`` to the accumulators block by block
    and subtracts the per-IP body force from the residual blocks it
    targets. Folding the body-force subtraction in here keeps the
    per-IP scan carry free of a separate forcing buffer.
    """
    num_blocks = len(R_acc)
    R_acc = [R_acc[r] + R_ip[r] for r in range(num_blocks)]
    dR_dU_acc = [
        [dR_dU_acc[r][s] + dR_dU_ip[r][s] for s in range(num_blocks)]
        for r in range(num_blocks)
    ]
    for block_idx, body_force_ip in body_force_ip_per_block.items():
        R_acc[block_idx] = R_acc[block_idx] - body_force_ip
    return R_acc, dR_dU_acc


def per_element_R_and_K(
        U_elem: Sequence[JaxArray],
        U_prev_elem: Sequence[JaxArray],
        params: Params,
        geom_per_elem: BlockIPGeometryPerElem,
        geom_shared: BlockIPGeometryShared,
        R_and_dR_dU_evaluator: RAndDRDUEvaluator,
        forcing_fns_by_block_idx: dict[int, Callable[
            [JaxArray | NDArray[np.floating], Scalar],
            JaxArray | NDArray[np.floating],
        ]],
        residual_block_shapes: Sequence[tuple[int, int]],
        t: Scalar,
) -> tuple[list[JaxArray], list[list[JaxArray]]]:
    """Per-element ``(R_blocks, dR_dU_blocks)`` at all IPs of one element.

    For each IP: read the cached per-block physical-frame field
    shapes and integration measure from ``geom_per_elem`` /
    ``geom_shared``, call ``R_and_dR_dU_evaluator`` with the per-block
    ``field_shapes_phys_per_block`` for the fused internal-force +
    tangent contribution, and accumulate. Subtract the forcing
    contribution ``f_ext_r = N_r · f_r · w · dv`` (with ``N_r`` from
    block ``r``'s field interpolant — Galerkin test function) from
    each residual block ``r`` listed in ``forcing_fns_by_block_idx``
    (sparse — absent block_idx means no forcing on that block, e.g.
    the pressure block of a mixed u-p method).

    Returns ``(R_blocks, dR_dU_blocks)`` where ``R_blocks[r]`` has
    shape ``(num_basis_fns_r, num_eqs_r)`` and ``dR_dU_blocks[r][s]``
    has shape ``(num_basis_fns_r, num_eqs_r, num_basis_fns_s,
    num_eqs_s)``, summed across IPs. ``residual_block_shapes`` is the
    list of ``(num_basis_fns, num_eqs)`` tuples per residual block,
    used to allocate the accumulator buffers (typically sourced from
    ``fe_problem.block_shapes``).

    Vmap-over-elements compatible: ``U_elem``, ``U_prev_elem``, and
    ``geom_per_elem`` carry the leading element axis; ``geom_shared``
    is element-invariant (passed with ``in_axes=None``). The
    integration-point sum is a :func:`jax.lax.scan` over IPs with a
    running ``(R_blocks, dR_dU_blocks)`` accumulator, so no per-IP
    axis is materialized; the scan body is wrapped in
    :func:`jax.checkpoint`, so the reverse-mode (gradient / Hessian)
    pass rematerializes the per-IP intermediates rather than storing
    them stacked over the IP axis. ``ip_set=0`` is always passed to
    the evaluator; GRs with multiple ip_sets dispatch on the
    trailing arg inside their residual_fn body (see
    :data:`cmad.typing.ResidualFnGR`).

    ``geom_per_elem`` packs the per-element-IP arrays — signed
    ``iso_jac_det`` (the integration measure ``dv = iso_jac_det·w``),
    physical-frame field-shape gradients (one entry per residual
    block, lifted via ``inv(iso_jac)`` from the geometric basis) and
    physical IP coordinates (``coords_ip = N_geom · X_elem``) consumed
    by forcing callables. ``geom_shared`` packs the mesh-uniform
    arrays — quadrature weights and per-block reference-frame field-
    shape values. Both are populated by
    :func:`cmad.fem.precompute.precompute_block_geometry` at
    ``FEProblem`` build time and consumed unchanged by both the
    CLOSED_FORM and COUPLED kernels; the geometric and field
    interpolants live entirely inside the precompute step, so the
    kernel sees only the lifted physical-frame arrays.

    The kernel is U-only at the API boundary because the
    CLOSED_FORM evaluator it consumes is U-only: stress comes from
    ``model.cauchy_closed_form(params, U_ip, U_ip_prev)`` and the
    per-IP local Newton is bypassed. A coupled-mode kernel that
    threads xi gather/scatter through the per-element evaluation is
    a separate function (parallel to this one), not an extension of
    this kernel — the dispatch happens in
    :func:`assemble_element_block` rather than per-IP.
    """
    num_blocks = len(residual_block_shapes)

    def per_ip(quad_w_ip, iso_jac_det_ip, coords_ip,
               field_N_at_ip_per_block, field_grad_N_phys_at_ip_per_block):
        field_shapes_phys_per_block = [
            ShapeFunctionsAtIP(
                N=field_N_at_ip_per_block[r],
                grad_N=field_grad_N_phys_at_ip_per_block[r],
            )
            for r in range(num_blocks)
        ]
        R_ip, dR_dU_ip = R_and_dR_dU_evaluator(
            params, U_elem, U_prev_elem,
            field_shapes_phys_per_block, quad_w_ip, iso_jac_det_ip, 0,
        )
        body_force_ip_per_block = {
            block_idx: jnp.einsum(
                "a,k->ak",
                field_shapes_phys_per_block[block_idx].N,
                jnp.asarray(forcing_fn(coords_ip, t)),
            ) * quad_w_ip * iso_jac_det_ip
            for block_idx, forcing_fn in forcing_fns_by_block_idx.items()
        }
        return R_ip, dR_dU_ip, body_force_ip_per_block

    def ip_step(carry, ip_slice):
        R_acc, dR_dU_acc = carry
        R_ip, dR_dU_ip, body_force_ip_per_block = per_ip(*ip_slice)
        return _accumulate_ip(
            R_acc, dR_dU_acc, R_ip, dR_dU_ip, body_force_ip_per_block,
        ), None

    (R_blocks, dR_dU_blocks), _ = lax.scan(
        checkpoint(ip_step),
        _zero_R_and_dR_dU_accumulators(residual_block_shapes),
        (
            geom_shared.quad_w,
            geom_per_elem.iso_jac_det,
            geom_per_elem.coords_ip,
            [geom_shared.field_N_per_block[r] for r in range(num_blocks)],
            [
                geom_per_elem.field_grad_N_phys_per_block[r]
                for r in range(num_blocks)
            ],
        ),
    )

    return R_blocks, dR_dU_blocks


def per_element_R(
        U_elem: Sequence[JaxArray],
        U_prev_elem: Sequence[JaxArray],
        params: Params,
        geom_per_elem: BlockIPGeometryPerElem,
        geom_shared: BlockIPGeometryShared,
        R_evaluator: REvaluator,
        forcing_fns_by_block_idx: dict[int, Callable[
            [JaxArray | NDArray[np.floating], Scalar],
            JaxArray | NDArray[np.floating],
        ]],
        residual_block_shapes: Sequence[tuple[int, int]],
        t: Scalar,
) -> list[JaxArray]:
    """Per-element ``R_blocks`` for a CLOSED_FORM block (residual only).

    Residual-only counterpart of :func:`per_element_R_and_K`, used by
    :func:`assemble_global_residual`: it calls the ``"R"`` evaluator and
    accumulates only the residual blocks (no tangent), so a reaction-reading
    QoI evaluates ``R`` without building ``K``. CLOSED_FORM blocks carry no
    local state variables -- stress comes from
    ``model.cauchy_closed_form``, with no per-IP local Newton -- so there is
    no ``xi`` here; the COUPLED counterpart that threads ``xi_prev`` is
    :func:`per_element_R_coupled`, and :func:`assemble_element_block_residual`
    dispatches between the two on the block's mode. Each IP contributes the
    net residual ``R_int - f_ext`` (body force subtracted in place); the
    per-IP :func:`jax.lax.scan`, :func:`jax.checkpoint`-wrapped for
    reverse-mode rematerialization, sums them. See :func:`per_element_R_and_K`
    for the geometry-cache and vmap contract.
    """
    num_blocks = len(residual_block_shapes)

    def per_ip(quad_w_ip, iso_jac_det_ip, coords_ip,
               field_N_at_ip_per_block, field_grad_N_phys_at_ip_per_block):
        field_shapes_phys_per_block = [
            ShapeFunctionsAtIP(
                N=field_N_at_ip_per_block[r],
                grad_N=field_grad_N_phys_at_ip_per_block[r],
            )
            for r in range(num_blocks)
        ]
        R_ip = list(R_evaluator(
            params, U_elem, U_prev_elem,
            field_shapes_phys_per_block, quad_w_ip, iso_jac_det_ip, 0,
        ))
        for block_idx, forcing_fn in forcing_fns_by_block_idx.items():
            f_ext = jnp.einsum(
                "a,k->ak",
                field_shapes_phys_per_block[block_idx].N,
                jnp.asarray(forcing_fn(coords_ip, t)),
            ) * quad_w_ip * iso_jac_det_ip
            R_ip[block_idx] = R_ip[block_idx] - f_ext
        return R_ip

    def ip_step(R_acc, ip_slice):
        R_ip = per_ip(*ip_slice)
        return [R_acc[r] + R_ip[r] for r in range(num_blocks)], None

    R_blocks, _ = lax.scan(
        checkpoint(ip_step),
        [jnp.zeros(shape) for shape in residual_block_shapes],
        (
            geom_shared.quad_w,
            geom_per_elem.iso_jac_det,
            geom_per_elem.coords_ip,
            [geom_shared.field_N_per_block[r] for r in range(num_blocks)],
            [
                geom_per_elem.field_grad_N_phys_per_block[r]
                for r in range(num_blocks)
            ],
        ),
    )
    return R_blocks


def per_element_R_and_K_coupled(
        U_elem: Sequence[JaxArray],
        U_prev_elem: Sequence[JaxArray],
        params: Params,
        xi_prev_per_ip: JaxArray,
        geom_per_elem: BlockIPGeometryPerElem,
        geom_shared: BlockIPGeometryShared,
        R_and_dR_dU_and_xi_evaluator: RAndDRDUAndXiEvaluator,
        unravel_xi: Callable[[JaxArray], StateList],
        forcing_fns_by_block_idx: dict[int, Callable[
            [JaxArray | NDArray[np.floating], Scalar],
            JaxArray | NDArray[np.floating],
        ]],
        residual_block_shapes: Sequence[tuple[int, int]],
        t: Scalar,
) -> tuple[list[JaxArray], list[list[JaxArray]], JaxArray]:
    """Per-element ``(R_blocks, dR_dU_blocks, xi_solved_per_ip)`` for COUPLED.

    Mirror of :func:`per_element_R_and_K` for the COUPLED branch of
    :meth:`GlobalResidual.for_model`. Each IP runs the per-IP local
    Newton inside ``R_and_dR_dU_and_xi_evaluator`` (the
    Newton-running 8-arg evaluator from the COUPLED dict) and gets
    back ``(R, dR_dU, xi_solved)`` with ``dR_dU`` already
    IFT-corrected via ``make_newton_solve``'s ``custom_jvp`` rule.
    Body forces accumulate identically to CLOSED_FORM — they're
    U-dependent only, no xi coupling, so the forcing block is
    mode-independent.

    ``xi_prev_per_ip`` is flat-trailing: shape
    ``(n_ips, total_xi_dofs)``, matching the FE-state per-element
    xi-history layout. Output ``xi_solved_per_ip`` is the same
    flat-trailing layout — direct reshape into the
    ``(n_elems, n_ips, total_xi_dofs)`` block-level history when
    the caller vmaps. Pytree-vs-flat conversion happens inside this
    kernel: the per-IP local Newton inside the evaluator is
    pytree-keyed on xi (matching ``model._residual``'s
    ``StateList`` block-by-block contract), and ``unravel_xi`` /
    ``ravel_pytree`` bridge between that pytree boundary and the
    flat storage layout.

    ``unravel_xi`` is a Python-side closure capturing the pytree
    treedef from ``ravel_pytree(model._init_xi)[1]``. Capture once
    at the call site; do not pass through ``static_argnums``. The
    kernel invokes it only inside the per-IP scan body, traced once
    under jit, so the closure is seen as ordinary Python by JAX.

    Vmap-over-elements compatible: ``U_elem``, ``U_prev_elem``,
    ``xi_prev_per_ip``, and ``geom_per_elem`` carry the leading
    element axis when the caller vmaps; ``geom_shared`` and the rest
    are element-invariant. See :func:`per_element_R_and_K` for the
    cache contract; identical here. The per-IP scan body is
    likewise :func:`jax.checkpoint`-wrapped; on the reverse pass
    that rematerializes each IP's local Newton return-map rather
    than storing its per-IP intermediates. ``ip_set=0`` and the
    per-IP index ``ip_idx`` (used only to label the optional local-
    convergence print) are passed to the evaluator; multi-ip_set
    GRs dispatch on ``ip_set`` inside their residual_fn body (see
    :data:`cmad.typing.ResidualFnGR`).
    """
    num_blocks = len(residual_block_shapes)

    def per_ip(quad_w_ip, iso_jac_det_ip, coords_ip,
               xi_prev_at_ip,
               field_N_at_ip_per_block, field_grad_N_phys_at_ip_per_block,
               ip_idx):
        field_shapes_phys_per_block = [
            ShapeFunctionsAtIP(
                N=field_N_at_ip_per_block[r],
                grad_N=field_grad_N_phys_at_ip_per_block[r],
            )
            for r in range(num_blocks)
        ]
        xi_prev_blocks = unravel_xi(xi_prev_at_ip)
        R_ip, dR_dU_ip, xi_blocks = R_and_dR_dU_and_xi_evaluator(
            params, U_elem, U_prev_elem, xi_prev_blocks,
            field_shapes_phys_per_block, quad_w_ip, iso_jac_det_ip, 0,
            ip_idx,
        )
        # Discard the unravel callable; xi treedef is fixed by
        # the ``unravel_xi`` closure already.
        xi_flat, _ = ravel_pytree(xi_blocks)
        body_force_ip_per_block = {
            block_idx: jnp.einsum(
                "a,k->ak",
                field_shapes_phys_per_block[block_idx].N,
                jnp.asarray(forcing_fn(coords_ip, t)),
            ) * quad_w_ip * iso_jac_det_ip
            for block_idx, forcing_fn in forcing_fns_by_block_idx.items()
        }
        return R_ip, dR_dU_ip, xi_flat, body_force_ip_per_block

    def ip_step(carry, ip_slice):
        R_acc, dR_dU_acc = carry
        R_ip, dR_dU_ip, xi_flat, body_force_ip_per_block = per_ip(
            *ip_slice,
        )
        carry = _accumulate_ip(
            R_acc, dR_dU_acc, R_ip, dR_dU_ip, body_force_ip_per_block,
        )
        return carry, xi_flat

    (R_blocks, dR_dU_blocks), xi_solved_per_ip = lax.scan(
        checkpoint(ip_step),
        _zero_R_and_dR_dU_accumulators(residual_block_shapes),
        (
            geom_shared.quad_w,
            geom_per_elem.iso_jac_det,
            geom_per_elem.coords_ip,
            xi_prev_per_ip,
            [geom_shared.field_N_per_block[r] for r in range(num_blocks)],
            [
                geom_per_elem.field_grad_N_phys_per_block[r]
                for r in range(num_blocks)
            ],
            jnp.arange(geom_shared.quad_w.shape[0]),
        ),
    )

    return R_blocks, dR_dU_blocks, xi_solved_per_ip


def per_element_R_coupled(
        U_elem: Sequence[JaxArray],
        U_prev_elem: Sequence[JaxArray],
        params: Params,
        xi_prev_per_ip: JaxArray,
        geom_per_elem: BlockIPGeometryPerElem,
        geom_shared: BlockIPGeometryShared,
        R_coupled_evaluator: Callable[..., Sequence[JaxArray]],
        unravel_xi: Callable[[JaxArray], StateList],
        forcing_fns_by_block_idx: dict[int, Callable[
            [JaxArray | NDArray[np.floating], Scalar],
            JaxArray | NDArray[np.floating],
        ]],
        residual_block_shapes: Sequence[tuple[int, int]],
        t: Scalar,
) -> list[JaxArray]:
    """Per-element ``R_blocks`` for a COUPLED block (residual only).

    Residual-only counterpart of :func:`per_element_R_and_K_coupled`, used by
    :func:`assemble_global_residual`. ``R_coupled_evaluator`` is the COUPLED
    ``"R"`` evaluator (``coupled_r_total``), which runs the per-IP local
    Newton from ``xi_prev`` and returns ``R`` only -- no tangent, and no
    converged-xi side-product, since a reaction read needs neither. Its 8-arg
    call shape ``(params, U, U_prev, xi_prev, shapes_ip, w, dv, ip_set)``
    differs from the CLOSED_FORM :class:`cmad.typing.REvaluator`, hence the
    loose callable type. ``xi_prev_per_ip`` is flat-trailing ``(n_ips,
    total_xi_dofs)``, unraveled per IP as in
    :func:`per_element_R_and_K_coupled`. Each IP contributes the net residual
    ``R_int - f_ext``; the checkpointed per-IP scan sums them.
    """
    num_blocks = len(residual_block_shapes)

    def per_ip(quad_w_ip, iso_jac_det_ip, coords_ip, xi_prev_at_ip,
               field_N_at_ip_per_block, field_grad_N_phys_at_ip_per_block):
        field_shapes_phys_per_block = [
            ShapeFunctionsAtIP(
                N=field_N_at_ip_per_block[r],
                grad_N=field_grad_N_phys_at_ip_per_block[r],
            )
            for r in range(num_blocks)
        ]
        xi_prev_blocks = unravel_xi(xi_prev_at_ip)
        R_ip = list(R_coupled_evaluator(
            params, U_elem, U_prev_elem, xi_prev_blocks,
            field_shapes_phys_per_block, quad_w_ip, iso_jac_det_ip, 0,
        ))
        for block_idx, forcing_fn in forcing_fns_by_block_idx.items():
            f_ext = jnp.einsum(
                "a,k->ak",
                field_shapes_phys_per_block[block_idx].N,
                jnp.asarray(forcing_fn(coords_ip, t)),
            ) * quad_w_ip * iso_jac_det_ip
            R_ip[block_idx] = R_ip[block_idx] - f_ext
        return R_ip

    def ip_step(R_acc, ip_slice):
        R_ip = per_ip(*ip_slice)
        return [R_acc[r] + R_ip[r] for r in range(num_blocks)], None

    R_blocks, _ = lax.scan(
        checkpoint(ip_step),
        [jnp.zeros(shape) for shape in residual_block_shapes],
        (
            geom_shared.quad_w,
            geom_per_elem.iso_jac_det,
            geom_per_elem.coords_ip,
            xi_prev_per_ip,
            [geom_shared.field_N_per_block[r] for r in range(num_blocks)],
            [
                geom_per_elem.field_grad_N_phys_per_block[r]
                for r in range(num_blocks)
            ],
        ),
    )
    return R_blocks


def assemble_element_block(
        fe_problem: FEProblem,
        fe_arrays: "FEKernelArrays",
        params_by_block: Mapping[str, Params],
        block_name: str,
        U_global: NDArray[np.floating] | JaxArray,
        U_prev_global: NDArray[np.floating] | JaxArray,
        t: Scalar,
        xi_prev_per_block: NDArray[np.floating] | JaxArray | None = None,
) -> tuple[JaxArray, JaxArray, JaxArray | None]:
    """Assemble one element block's R contribution + COO data.

    Dispatches on ``fe_problem.modes_by_block[block_name]``. The
    CLOSED_FORM branch vmaps :func:`per_element_R_and_K`; the COUPLED
    branch vmaps :func:`per_element_R_and_K_coupled` and threads
    ``xi_prev_per_block`` (shape ``(n_elems_block, n_ips,
    total_xi_dofs)``) through the per-IP local Newton, returning the
    converged xi at the same shape.

    Returns ``(R_block, vals, xi_solved_per_block)``. ``R_block`` is a
    length-``dof_map.num_total_dofs`` JAX vector: this block's
    per-element residual contributions scattered to their global dof
    positions, and left zero at every dof the block's elements do not
    touch, so :func:`assemble_global` sums the per-block vectors
    directly. ``vals`` is the JAX-traced COO data stream, emitted in
    ``(r, s)`` residual-block / U-block order; its static
    ``(rows, cols)`` are ``fe_arrays.coo_rows`` / ``coo_cols``, which
    :func:`assembled_coo_indices` builds in the same emit order.
    ``xi_solved_per_block`` is the converged xi for COUPLED, ``None``
    for CLOSED_FORM.

    The per-element U-gather index arrays, the per-residual-block
    R-scatter eq arrays, and the reference-frame geometry cache are
    read from ``fe_arrays`` rather than derived from ``fe_problem`` /
    ``mesh.connectivity`` in-trace.

    ``xi_prev_per_block`` is required when the block is COUPLED and
    must match the cached layout ``(n_elems_block, n_ips,
    total_xi_dofs)``; for CLOSED_FORM blocks the kwarg is ignored and
    may be ``None``.

    Multi-residual-block GRs scatter via nested loops over ``(r, s)``
    residual-block / U-block pairs; single-block GRs are the
    degenerate ``r=s=0`` case.
    """
    U_elem_block = _gather_element_U(U_global, fe_arrays, block_name)
    U_prev_elem_block = _gather_element_U(
        U_prev_global, fe_arrays, block_name,
    )

    params = params_by_block[block_name]
    evaluators = fe_problem.evaluators_by_block[block_name]
    mode = fe_problem.modes_by_block[block_name]
    block_shapes = fe_problem.block_shapes
    num_blocks = len(block_shapes)
    forcing_fns_by_block_idx = fe_problem.forcing_fns_by_block_idx or {}

    geom_cache = fe_arrays.geometry_cache[block_name]

    xi_solved_per_block: JaxArray | None
    if mode == GlobalResidualMode.COUPLED:
        if xi_prev_per_block is None:
            raise ValueError(
                f"COUPLED block '{block_name}' requires "
                f"xi_prev_per_block; got None"
            )
        unravel_xi = fe_problem.unravel_xi_by_block[block_name]
        xi_prev_jax = jnp.asarray(xi_prev_per_block)
        R_per_elem_blocks, K_per_elem_blocks, xi_solved_per_block = vmap(
            lambda U, Up, geom, xi_prev: per_element_R_and_K_coupled(
                U, Up, params, xi_prev,
                geom, geom_cache.shared,
                evaluators["R_and_dR_dU_and_xi"],
                unravel_xi,
                forcing_fns_by_block_idx, block_shapes, t,
            ),
            in_axes=(0, 0, 0, 0),
            axis_name="elem",
        )(
            U_elem_block, U_prev_elem_block,
            geom_cache.per_elem, xi_prev_jax,
        )
    else:
        R_per_elem_blocks, K_per_elem_blocks = vmap(
            lambda U, Up, geom: per_element_R_and_K(
                U, Up, params,
                geom, geom_cache.shared,
                evaluators["R_and_dR_dU"],
                forcing_fns_by_block_idx, block_shapes, t,
            ),
            in_axes=(0, 0, 0),
            axis_name="elem",
        )(U_elem_block, U_prev_elem_block, geom_cache.per_elem)
        xi_solved_per_block = None

    eq_indices_per_block = fe_arrays.r_scatter_eq_by_block[block_name]
    n_elems = eq_indices_per_block[0].shape[0]
    n_dofs = fe_problem.dof_map.num_total_dofs

    R_block = jnp.zeros(n_dofs)
    for r in range(num_blocks):
        R_flat = R_per_elem_blocks[r].reshape(n_elems, -1)
        R_block = R_block.at[eq_indices_per_block[r].ravel()].add(
            R_flat.ravel(),
        )

    vals_all: list[JaxArray] = []
    for r in range(num_blocks):
        n_dofs_r = eq_indices_per_block[r].shape[1]
        for s in range(num_blocks):
            n_dofs_s = eq_indices_per_block[s].shape[1]
            K_flat = K_per_elem_blocks[r][s].reshape(
                n_elems, n_dofs_r, n_dofs_s,
            )
            vals_all.append(K_flat.ravel())

    return R_block, jnp.concatenate(vals_all), xi_solved_per_block


def assemble_element_block_residual(
        fe_problem: FEProblem,
        fe_arrays: "FEKernelArrays",
        params_by_block: Mapping[str, Params],
        block_name: str,
        U_global: NDArray[np.floating] | JaxArray,
        U_prev_global: NDArray[np.floating] | JaxArray,
        t: Scalar,
        xi_prev_per_block: NDArray[np.floating] | JaxArray | None = None,
) -> JaxArray:
    """Assemble one element block's global ``R`` contribution (no tangent).

    Residual-only counterpart of :func:`assemble_element_block`: dispatches
    on ``fe_problem.modes_by_block[block_name]``, vmaps
    :func:`per_element_R` (CLOSED_FORM) or :func:`per_element_R_coupled`
    (COUPLED) over the block's elements, and scatters the per-element
    residual blocks to their global dof positions exactly as
    :func:`assemble_element_block` does -- but emits no COO tangent data.
    ``xi_prev_per_block`` (shape ``(n_elems_block, n_ips, total_xi_dofs)``)
    is required for COUPLED blocks and ignored for CLOSED_FORM.
    """
    U_elem_block = _gather_element_U(U_global, fe_arrays, block_name)
    U_prev_elem_block = _gather_element_U(
        U_prev_global, fe_arrays, block_name,
    )

    params = params_by_block[block_name]
    evaluators = fe_problem.evaluators_by_block[block_name]
    mode = fe_problem.modes_by_block[block_name]
    block_shapes = fe_problem.block_shapes
    num_blocks = len(block_shapes)
    forcing_fns_by_block_idx = fe_problem.forcing_fns_by_block_idx or {}
    geom_cache = fe_arrays.geometry_cache[block_name]

    if mode == GlobalResidualMode.COUPLED:
        if xi_prev_per_block is None:
            raise ValueError(
                f"COUPLED block '{block_name}' requires "
                f"xi_prev_per_block; got None"
            )
        unravel_xi = fe_problem.unravel_xi_by_block[block_name]
        xi_prev_jax = jnp.asarray(xi_prev_per_block)
        R_per_elem_blocks = vmap(
            lambda U, Up, geom, xi_prev: per_element_R_coupled(
                U, Up, params, xi_prev,
                geom, geom_cache.shared,
                evaluators["R"],
                unravel_xi,
                forcing_fns_by_block_idx, block_shapes, t,
            ),
            in_axes=(0, 0, 0, 0),
            axis_name="elem",
        )(
            U_elem_block, U_prev_elem_block,
            geom_cache.per_elem, xi_prev_jax,
        )
    else:
        R_per_elem_blocks = vmap(
            lambda U, Up, geom: per_element_R(
                U, Up, params,
                geom, geom_cache.shared,
                evaluators["R"],
                forcing_fns_by_block_idx, block_shapes, t,
            ),
            in_axes=(0, 0, 0),
            axis_name="elem",
        )(U_elem_block, U_prev_elem_block, geom_cache.per_elem)

    eq_indices_per_block = fe_arrays.r_scatter_eq_by_block[block_name]
    n_elems = eq_indices_per_block[0].shape[0]
    n_dofs = fe_problem.dof_map.num_total_dofs

    R_block = jnp.zeros(n_dofs)
    for r in range(num_blocks):
        R_flat = R_per_elem_blocks[r].reshape(n_elems, -1)
        R_block = R_block.at[eq_indices_per_block[r].ravel()].add(
            R_flat.ravel(),
        )
    return R_block


def assemble_global(
        fe_problem: FEProblem,
        fe_arrays: "FEKernelArrays",
        params_by_block: Mapping[str, Params],
        U_global: NDArray[np.floating] | JaxArray,
        U_prev_global: NDArray[np.floating] | JaxArray,
        t: Scalar,
        xi_prev_by_block: Mapping[str, NDArray[np.floating] | JaxArray]
        | None = None,
) -> tuple[BCOO, JaxArray, dict[str, JaxArray]]:
    """Walk all element blocks and emit the global ``(K, R, xi_solved)``.

    ``K`` is the global tangent ``dR/dU`` as a
    :class:`jax.experimental.sparse.BCOO` of shape
    ``(n_dofs, n_dofs)``, deduplicated at the assembly boundary:
    duplicate ``(row, col)`` entries from shared-DOF assembly are
    segment-summed into the unique pattern, so ``K`` carries one
    entry per structural nonzero (``indices_sorted`` and
    ``unique_indices`` both hold).
    ``R`` is the flat JAX residual vector of length ``n_dofs``.
    ``xi_solved_by_block`` is the per-block converged-xi dict — keys
    are exactly the set of blocks whose mode is COUPLED, each entry
    shaped ``(n_elems_block, n_ips, total_xi_dofs)``. CLOSED_FORM-
    only problems get an empty dict.

    Nonlinear-FE convention: ``R(U) = R_int(U) - F_ext`` (body-force
    and surface-flux contributions folded into ``R``, no separate
    ``F`` vector). Body forces accumulate at the per-element level
    inside :func:`per_element_R_and_K` /
    :func:`per_element_R_and_K_coupled`; surface fluxes accumulate
    via :func:`cmad.fem.neumann.assemble_side_neumann` after the
    volume walk. The Newton driver in
    :func:`cmad.fem.nonlinear_solver.fe_newton_solve` solves
    ``K · dU = -R``; the linear ``K U = F`` form is the degenerate
    one-iter case for a U-linear residual. Explicit ``(coords, t)``
    Neumann fluxes are U-independent, so ``K`` gets no surface
    contribution.

    ``xi_prev_by_block`` carries the previous-step converged xi for
    each COUPLED block; the per-IP local Newton uses ``xi_prev`` as
    both initial guess and held-in-residual ``x_prev`` (path
    continuity for plasticity). ``None`` is valid only when the
    problem has no COUPLED blocks. When a COUPLED block's entry is
    missing, :func:`assemble_element_block` raises ``ValueError``
    with a clear message; shape mismatches surface as JAX vmap
    leading-axis errors when the kernel runs.

    Implementation note: each block's per-element residual is
    accumulated into a flat JAX vector via
    :func:`assemble_element_block`, which also returns the block's
    per-element-block ``vals`` (the with-duplicates COO data).
    Per-block residual contributions sum into ``R``; the per-block
    JAX ``vals`` concatenate into the duplicate-laden COO data
    buffer, which is then segment-summed into the unique pattern via
    ``fe_arrays.coo_dedup_scatter`` and paired with the static
    deduped ``fe_arrays.coo_rows`` / ``coo_cols`` (all built once by
    :func:`assembled_coo_dedup` in the same ``(block, r, s)`` emit
    order). The concatenated with-duplicates buffer is a transient,
    freed before return — the embedded-BC enforcement, the linear
    solve, and their AD shadows see only the deduped data. Non-None
    ``xi_solved_per_block`` returns populate the
    ``xi_solved_by_block`` dict. Surface fluxes add into ``R`` via
    :func:`cmad.fem.neumann.assemble_side_neumann` after the volume
    walk.
    """
    xi_prev = xi_prev_by_block or {}

    n_dofs = fe_problem.dof_map.num_total_dofs
    vals_all: list[JaxArray] = []
    R_global = jnp.zeros(n_dofs)
    xi_solved_by_block: dict[str, JaxArray] = {}

    for block_name in fe_problem.evaluators_by_block:
        R_block, vals, xi_solved = assemble_element_block(
            fe_problem, fe_arrays, params_by_block, block_name,
            U_global, U_prev_global, t,
            xi_prev_per_block=xi_prev.get(block_name),
        )
        R_global = R_global + R_block
        vals_all.append(vals)
        if xi_solved is not None:
            xi_solved_by_block[block_name] = xi_solved

    R_global = R_global + assemble_side_neumann(
        fe_problem.dof_map,
        fe_arrays.neumann_side_arrays,
        fe_problem.resolved_neumann_bcs,
        t,
    )

    vals = jnp.concatenate(vals_all)
    unique_data = jnp.zeros(
        fe_arrays.coo_rows.shape[0], dtype=vals.dtype,
    ).at[fe_arrays.coo_dedup_scatter].add(vals)
    K = BCOO(
        (unique_data,
         jnp.stack([fe_arrays.coo_rows, fe_arrays.coo_cols], axis=-1)),
        shape=(n_dofs, n_dofs),
        indices_sorted=True,
        unique_indices=True,
    )
    return K, R_global, xi_solved_by_block


def assemble_global_residual(
        fe_problem: FEProblem,
        fe_arrays: "FEKernelArrays",
        params_by_block: Mapping[str, Params],
        U_global: NDArray[np.floating] | JaxArray,
        U_prev_global: NDArray[np.floating] | JaxArray,
        t: Scalar,
        xi_prev_by_block: Mapping[str, NDArray[np.floating] | JaxArray]
        | None = None,
) -> JaxArray:
    """Global residual ``R(U) = R_int - F_ext`` without the tangent ``K``.

    Residual-only counterpart of :func:`assemble_global`: the same ``R`` as
    its second return (body force and surface flux folded in), just without
    building the tangent it would discard. For consumers that need only
    ``R``: a reaction-reading QoI
    (:class:`cmad.qois.fe_load_match.FELoadMatch`), or line-search
    trial-point evaluation -- matching ``assemble_global``'s ``R`` value for
    value is what keeps a trial residual consistent with the Newton solve it
    backtracks.

    Such a QoI reads this ``R`` at the Dirichlet dofs, where it is the
    consistent-nodal reaction -- the internal force minus the volumetric
    body force lumped to those nodes. The surface flux never enters a
    reaction: a dof is either Dirichlet-constrained or Neumann-loaded, never
    both, so :func:`cmad.fem.neumann.assemble_side_neumann` writes only the
    Neumann dofs and leaves the Dirichlet dofs untouched. It is added here
    solely so the returned vector matches ``assemble_global``'s ``R`` value
    for value. Walks each block via :func:`assemble_element_block_residual`
    (CLOSED_FORM / COUPLED dispatch).
    """
    xi_prev = xi_prev_by_block or {}
    n_dofs = fe_problem.dof_map.num_total_dofs
    R_global = jnp.zeros(n_dofs)

    for block_name in fe_problem.evaluators_by_block:
        R_global = R_global + assemble_element_block_residual(
            fe_problem, fe_arrays, params_by_block, block_name,
            U_global, U_prev_global, t,
            xi_prev_per_block=xi_prev.get(block_name),
        )

    R_global = R_global + assemble_side_neumann(
        fe_problem.dof_map,
        fe_arrays.neumann_side_arrays,
        fe_problem.resolved_neumann_bcs,
        t,
    )
    return R_global


def assembled_coo_indices(
        fe_problem: FEProblem,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Static with-duplicates ``(rows, cols)`` of the assembled COO stream.

    The per-element COO triplet stream from
    :func:`assemble_element_block` carries duplicates — a basis
    function shared by ``k`` elements yields ``k`` triplets at one
    ``(row, col)``. Its ``(rows, cols)`` are mesh-derived constants;
    this helper rebuilds them without running an assembly, matching
    the ``(block, r, s)`` emit order of :func:`assemble_element_block`.
    :func:`assembled_coo_dedup` lex-sorts this stream into the unique
    pattern the deduped BCOO of :func:`assemble_global` carries.

    Keep the ``(block, r, s)`` walk in sync with
    :func:`assemble_element_block`'s COO emit: changing the iteration
    order there requires the same change here, or any permutation
    pre-computed against this output silently breaks at the next
    Newton iter.
    """
    mesh = fe_problem.mesh
    dof_map = fe_problem.dof_map
    num_blocks = fe_problem.gr.num_residuals
    field_idx_per_block = fe_problem.field_idx_per_block

    rows_all: list[NDArray[np.intp]] = []
    cols_all: list[NDArray[np.intp]] = []
    for block_name in fe_problem.evaluators_by_block:
        elem_indices = mesh.element_blocks[block_name]
        connectivity_block = mesh.connectivity[elem_indices]
        n_elems = connectivity_block.shape[0]
        eq_indices_per_block = [
            _element_eq_indices(
                connectivity_block, dof_map,
                field_idx=field_idx_per_block[r],
            )
            for r in range(num_blocks)
        ]
        for r in range(num_blocks):
            eq_r = eq_indices_per_block[r]
            n_dofs_r = eq_r.shape[1]
            for s in range(num_blocks):
                eq_s = eq_indices_per_block[s]
                n_dofs_s = eq_s.shape[1]
                rows_all.append(np.broadcast_to(
                    eq_r[:, :, None],
                    (n_elems, n_dofs_r, n_dofs_s),
                ).ravel())
                cols_all.append(np.broadcast_to(
                    eq_s[:, None, :],
                    (n_elems, n_dofs_r, n_dofs_s),
                ).ravel())
    return np.concatenate(rows_all), np.concatenate(cols_all)


def assembled_coo_dedup(
        fe_problem: FEProblem,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Deduped ``(rows, cols)`` of the assembled COO + the dedup scatter.

    The assembled COO triplet stream from :func:`assembled_coo_indices`
    carries duplicates: a basis function shared by ``k`` elements
    receives ``k`` triplets at one ``(row, col)``. This helper
    lex-sorts that stream into the ``num_unique`` distinct
    ``(row, col)`` entries and returns, alongside the unique pattern,
    a length-``nnz`` ``coo_dedup_scatter`` mapping each triplet (in
    :func:`assembled_coo_indices` emit order) to its unique slot.

    :func:`assemble_global` segment-sums the ``nnz``-length COO data
    into the ``num_unique``-length deduped buffer through
    ``coo_dedup_scatter`` right at the assembly boundary, so the
    embedded-BC enforcement, the linear solve, and their AD shadows
    never carry the duplicate-laden buffer.

    The unique pattern is lex-sorted ``(row, col)`` major-minor, so
    the deduped BCOO :func:`assemble_global` builds from it satisfies
    ``indices_sorted`` and ``unique_indices``. ``coo_dedup_scatter``
    inherits the ``(block, r, s)`` emit-order contract of
    :func:`assembled_coo_indices`: it is built from that output, so
    it stays in lockstep with :func:`assemble_element_block`'s COO
    emit.
    """
    rows, cols = assembled_coo_indices(fe_problem)
    sort_perm = np.lexsort((cols, rows))
    sorted_rows = rows[sort_perm]
    sorted_cols = cols[sort_perm]

    is_new_group = np.empty(rows.shape[0], dtype=bool)
    is_new_group[0] = True
    is_new_group[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | (
        sorted_cols[1:] != sorted_cols[:-1]
    )
    segment_of_sorted = (np.cumsum(is_new_group) - 1).astype(np.intp)

    unique_rows = sorted_rows[is_new_group].astype(np.intp)
    unique_cols = sorted_cols[is_new_group].astype(np.intp)

    coo_dedup_scatter = np.empty(rows.shape[0], dtype=np.intp)
    coo_dedup_scatter[sort_perm] = segment_of_sorted
    return unique_rows, unique_cols, coo_dedup_scatter
