"""Derived-quantity evaluators for FE post-processing.

FE-side helpers that need model-layer access to compute quantities
derivable from a converged ``(U, xi)`` :class:`FEState`. Today hosts
:func:`evaluate_cauchy_at_ips`; strain / Mises / plastic-work-history
helpers layer in here when needed.

Compose with :func:`cmad.io.results.ip_average_to_element` to reduce
``(n_elems, n_ip, *components)`` IP-level results to per-element
integration-measure-weighted means for Exodus element-field output;
the IP-level array is returned unreduced for callers that want the
raw per-Gauss-point data (diagnostics, projection-to-nodes work,
custom reductions).
"""
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.flatten_util import ravel_pytree
from numpy.typing import NDArray

from cmad.fem.assembly import _gather_element_U
from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.global_residuals.interpolation import (
    interpolate_global_fields_at_ip,
)
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.var_types import get_vector_from_sym_tensor


def evaluate_cauchy_at_ips(
        fe_problem: FEProblem,
        fe_state: FEState,
        step: int,
        block_name: str,
) -> NDArray[np.floating]:
    """Cauchy stress at every (elem, IP) of a block.

    Returns ``(n_elems, n_ip, 6)`` in cmad-internal sym-tensor vec
    order ``[xx, xy, xz, yy, yz, zz]``.

    Mode-dispatched on ``fe_problem.modes_by_block[block_name]``:

    - CLOSED_FORM: ``model.cauchy_closed_form(params, U_ip,
      U_ip_prev)``. The xi history is not consulted.
    - COUPLED: ``model.cauchy(xi, xi_prev, params, U_ip, U_ip_prev)``
      with ``xi`` pulled from
      ``fe_state.xi_at(step, block_name)`` and ``xi_prev`` from
      step ``step - 1`` (zeros at ``step == 0``). The flat-trailing
      xi storage is unraveled to the model's StateList pytree per
      IP via ``jax.flatten_util.ravel_pytree(model._init_xi)``.

    ``U`` and ``U_prev`` come from ``fe_state.U_at(step)`` /
    ``fe_state.U_at(step - 1)`` (zeros at ``step == 0``); per-IP
    ``GlobalFieldsAtPoint`` interpolation uses the cached field
    shape values from ``fe_problem.geometry_cache[block_name]``,
    matching the assembly kernels' contract.
    """
    mesh = fe_problem.mesh
    dof_map = fe_problem.dof_map
    elem_indices = mesh.element_blocks[block_name]
    connectivity_block = mesh.connectivity[elem_indices]

    U_global = jnp.asarray(fe_state.U_at(step))
    U_prev_global = (
        jnp.asarray(fe_state.U_at(step - 1)) if step > 0
        else jnp.zeros_like(U_global)
    )
    U_elem_block = _gather_element_U(
        U_global, dof_map, connectivity_block,
    )
    U_prev_elem_block = _gather_element_U(
        U_prev_global, dof_map, connectivity_block,
    )

    model = fe_problem.models_by_block[block_name]
    params = model.parameters.values
    mode = fe_problem.modes_by_block[block_name]
    var_names = fe_problem.gr.var_names
    num_blocks = len(fe_problem.block_shapes)

    geom_cache = fe_problem.geometry_cache[block_name]
    geom_per_elem = geom_cache.per_elem
    geom_shared = geom_cache.shared
    nips = int(geom_shared.quad_w.shape[0])

    if mode == GlobalResidualMode.CLOSED_FORM:
        if model.cauchy_closed_form is None:
            raise AttributeError(
                f"model {type(model).__name__} on block "
                f"'{block_name}' has no cauchy_closed_form callable; "
                f"either bind one in its constructor or use COUPLED "
                f"mode for postprocess cauchy queries"
            )
        cauchy_fn = model.cauchy_closed_form

        def cauchy_per_elem_closed_form(U_e, U_prev_e, gpe):
            cauchy_per_ip = jnp.zeros((nips, 6))
            for ip_idx in range(nips):
                shapes_ip = [
                    ShapeFunctionsAtIP(
                        N=geom_shared.field_N_per_block[r][ip_idx],
                        grad_N=(
                            gpe.field_grad_N_phys_per_block[r][ip_idx]
                        ),
                    )
                    for r in range(num_blocks)
                ]
                U_ip = interpolate_global_fields_at_ip(
                    U_e, shapes_ip, var_names,
                )
                U_prev_ip = interpolate_global_fields_at_ip(
                    U_prev_e, shapes_ip, var_names,
                )
                sigma = cauchy_fn(params, U_ip, U_prev_ip)
                cauchy_per_ip = cauchy_per_ip.at[ip_idx].set(
                    get_vector_from_sym_tensor(sigma, 3),
                )
            return cauchy_per_ip

        cauchy_blocks = vmap(
            cauchy_per_elem_closed_form, in_axes=(0, 0, 0),
        )(U_elem_block, U_prev_elem_block, geom_per_elem)

    elif mode == GlobalResidualMode.COUPLED:
        cauchy_fn = model.cauchy
        xi_history = jnp.asarray(fe_state.xi_at(step, block_name))
        xi_prev_history = (
            jnp.asarray(fe_state.xi_at(step - 1, block_name))
            if step > 0 else jnp.zeros_like(xi_history)
        )
        _, unravel_xi = ravel_pytree(model._init_xi)

        def cauchy_per_elem_coupled(
                U_e, U_prev_e, gpe, xi_per_ip, xi_prev_per_ip,
        ):
            cauchy_per_ip = jnp.zeros((nips, 6))
            for ip_idx in range(nips):
                shapes_ip = [
                    ShapeFunctionsAtIP(
                        N=geom_shared.field_N_per_block[r][ip_idx],
                        grad_N=(
                            gpe.field_grad_N_phys_per_block[r][ip_idx]
                        ),
                    )
                    for r in range(num_blocks)
                ]
                U_ip = interpolate_global_fields_at_ip(
                    U_e, shapes_ip, var_names,
                )
                U_prev_ip = interpolate_global_fields_at_ip(
                    U_prev_e, shapes_ip, var_names,
                )
                xi_blocks = unravel_xi(xi_per_ip[ip_idx])
                xi_prev_blocks = unravel_xi(xi_prev_per_ip[ip_idx])
                sigma = cauchy_fn(
                    xi_blocks, xi_prev_blocks, params, U_ip, U_prev_ip,
                )
                cauchy_per_ip = cauchy_per_ip.at[ip_idx].set(
                    get_vector_from_sym_tensor(sigma, 3),
                )
            return cauchy_per_ip

        cauchy_blocks = vmap(
            cauchy_per_elem_coupled, in_axes=(0, 0, 0, 0, 0),
        )(
            U_elem_block, U_prev_elem_block, geom_per_elem,
            xi_history, xi_prev_history,
        )

    else:
        raise ValueError(f"unsupported GlobalResidualMode: {mode}")

    return np.asarray(cauchy_blocks)
