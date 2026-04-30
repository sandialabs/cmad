"""Element + global FE assembly machinery."""
from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
import scipy.sparse
from jax import vmap
from numpy.typing import NDArray

from cmad.fem.dof import GlobalDofMap, GlobalFieldLayout
from cmad.fem.fe_problem import FEProblem
from cmad.fem.finite_element import EntityType
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
        dof_map: GlobalDofMap,
        connectivity_block: NDArray[np.intp],
) -> list[JaxArray]:
    """Gather per-element basis-coefficient arrays from the flat global U.

    Returns one array per field in ``dof_map.field_layouts``; entry ``f``
    has shape ``(n_elems, num_dofs_per_element_f,
    num_dofs_per_basis_fn[f])`` ready for ``vmap`` over the leading
    element axis. Per-element basis-fn indices come from
    :func:`_element_basis_fns`; the eq formula is
    ``eq = block_offset + basis_fn * num_dofs_per_basis_fn + component``.
    """
    U_jax = jnp.asarray(U_global)
    out: list[JaxArray] = []
    for field_idx, layout in enumerate(dof_map.field_layouts):
        bf_per_elem = _element_basis_fns(layout, connectivity_block)
        ndofs = dof_map.num_dofs_per_basis_fn[field_idx]
        block_offset = dof_map.block_offsets[field_idx]
        k_arr = np.arange(ndofs)
        eq_3d = (
            block_offset
            + bf_per_elem[:, :, None] * ndofs
            + k_arr[None, None, :]
        )
        out.append(U_jax[eq_3d])
    return out


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


def per_element_R_and_K(
        X_elem: JaxArray,
        U_elem: Sequence[JaxArray],
        U_prev_elem: Sequence[JaxArray],
        params: Params,
        quad_xi: NDArray[np.floating] | JaxArray,
        quad_w: NDArray[np.floating] | JaxArray,
        geom_interpolant_fn: Callable[[JaxArray], ShapeFunctionsAtIP],
        per_block_interpolant_fns: Sequence[
            Callable[[JaxArray], ShapeFunctionsAtIP]
        ],
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

    For each IP: evaluate the geometric interpolant to compute the
    isoparametric Jacobian, then evaluate each residual block's field
    interpolant and lift its reference-frame gradients to physical
    frame via the shared ``inv(iso_jac)``. Call
    ``R_and_dR_dU_evaluator`` with the per-block
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

    Vmap-over-elements compatible: only ``X_elem``, ``U_elem``,
    ``U_prev_elem`` carry the leading element axis; the rest are
    element-invariant. The Python-level IP loop unrolls under jit.
    ``ip_set=0`` is always passed to the evaluator; GRs with multiple
    ip_sets dispatch on the trailing arg inside their residual_fn
    body (see :data:`cmad.typing.ResidualFnGR`).

    ``geom_interpolant_fn`` provides the reference-frame shapes for
    the element geometry (typically the mesh's
    ``geometric_finite_element.interpolant_fn``); it feeds the
    isoparametric Jacobian (``iso_jac_det`` for the integration measure
    ``dv = iso_jac_det · w``, ``inv(iso_jac)`` for the chain rule
    that lifts each per-block ``grad_N_ref`` to physical frame) and
    the IP-coordinate map ``coords_ip = N · X_elem`` consumed by
    forcing callables. ``per_block_interpolant_fns[r]`` provides the
    reference-frame shapes for residual block ``r``'s field; the
    physical-frame versions assembled into
    ``field_shapes_phys_per_block`` then drive both the evaluator
    call and the ``f_ext`` test-function read. For isoparametric
    setups (``mesh.geometric_finite_element`` matches every
    ``layout.finite_element``) the geometric and per-block
    interpolants are the same callable; for subparametric (geometry
    FE strictly lower-order than each field FE) or mixed-basis
    (different field FE per block) setups they differ.

    ``xi_dummy`` and ``xi_prev_dummy`` are zero-filled placeholders.
    The CLOSED_FORM evaluator path uses ``model.cauchy_closed_form``,
    which is U-only — state args are unused, so passing zeros is
    correct. When COUPLED mode lands, the per-IP local Newton will
    consume xi (plasticity etc.); these args become load-bearing and
    assembly will need to gather/scatter actual element-IP state.
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
        geom_shapes_ref = geom_interpolant_fn(quad_xi[ip_idx])
        _, iso_jac_det, iso_jac = iso_jac_at_ip(
            geom_shapes_ref.grad_N, X_elem,
        )
        iso_jac_inv = jnp.linalg.inv(iso_jac)
        w_ref = quad_w[ip_idx]
        dv = iso_jac_det

        field_shapes_phys_per_block: list[ShapeFunctionsAtIP] = []
        for r in range(num_blocks):
            field_shapes_ref_r = per_block_interpolant_fns[r](
                quad_xi[ip_idx],
            )
            grad_N_phys_r = field_shapes_ref_r.grad_N @ iso_jac_inv
            field_shapes_phys_per_block.append(ShapeFunctionsAtIP(
                N=field_shapes_ref_r.N, grad_N=grad_N_phys_r,
            ))

        R_ip, dR_dU_ip = R_and_dR_dU_evaluator(
            xi_dummy, xi_prev_dummy, params, U_elem, U_prev_elem,
            field_shapes_phys_per_block, w_ref, dv, 0,
        )

        for r in range(num_blocks):
            R_blocks[r] = R_blocks[r] + R_ip[r]
            for s in range(num_blocks):
                dR_dU_blocks[r][s] = dR_dU_blocks[r][s] + dR_dU_ip[r][s]

        if forcing_fns_by_block_idx:
            coords_ip = geom_shapes_ref.N @ X_elem
            for block_idx, forcing_fn in forcing_fns_by_block_idx.items():
                f_ip = jnp.asarray(forcing_fn(coords_ip, t))
                f_ext = jnp.einsum(
                    "a,k->ak",
                    field_shapes_phys_per_block[block_idx].N, f_ip,
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
    block_shapes = fe_problem.block_shapes
    num_blocks = len(block_shapes)

    quad_rule = fe_problem.assembly_quadrature[mesh.element_family]
    forcing_fns_by_block_idx = fe_problem.forcing_fns_by_block_idx or {}

    assert mesh.geometric_finite_element is not None, (
        "Mesh.geometric_finite_element is resolved in __post_init__"
    )
    geom_interpolant_fn = mesh.geometric_finite_element.interpolant_fn

    field_idx_per_block = fe_problem.field_idx_per_block
    per_block_interpolant_fns: list[
        Callable[[JaxArray], ShapeFunctionsAtIP]
    ] = [
        fl.finite_element.interpolant_fn
        for fl in fe_problem.field_layouts_per_block
    ]

    xi_dummy: StateList = [jnp.zeros_like(b) for b in model._init_xi]
    xi_prev_dummy: StateList = [jnp.zeros_like(b) for b in model._init_xi]

    quad_xi = jnp.asarray(quad_rule.xi)
    quad_w = jnp.asarray(quad_rule.w)

    R_per_elem_blocks, K_per_elem_blocks = vmap(
        lambda X, U, Up: per_element_R_and_K(
            X, U, Up, params,
            quad_xi, quad_w,
            geom_interpolant_fn,
            per_block_interpolant_fns,
            evaluators["R_and_dR_dU"],
            forcing_fns_by_block_idx, block_shapes, t,
            xi_dummy, xi_prev_dummy,
        ),
    )(X_block, U_elem_block, U_prev_elem_block)

    eq_indices_per_block: list[NDArray[np.intp]] = [
        _element_eq_indices(
            connectivity_block, dof_map,
            field_idx=field_idx_per_block[r],
        )
        for r in range(num_blocks)
    ]

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

    ``K_coo`` is the global tangent ``dR/dU`` as a
    ``scipy.sparse.coo_matrix`` of shape ``(n_dofs, n_dofs)``; ``R``
    is the flat residual vector of length ``n_dofs``, dtype float64.
    Nonlinear-FE convention: ``R(U) = R_int(U) - F_ext`` (body-force
    contribution folded into ``R`` at the per-element level, no
    separate ``F`` vector). The Newton driver in
    :func:`cmad.fem.nonlinear_solver.fe_newton_solve` solves
    ``K · dU = -R``; the linear ``K U = F`` form is the degenerate
    one-iter case for a U-linear residual.

    Implementation note: each block's per-element residual and tangent
    are computed by :func:`assemble_element_block`, which returns
    per-block ``(rows, cols, vals, R_block)`` triplet streams; this
    function concatenates the streams across blocks and builds the COO
    matrix once at the bottom (duplicate ``(row, col)`` entries
    accumulate naturally on ``.tocsr()``).
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
