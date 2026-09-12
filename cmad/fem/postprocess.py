"""Derived-quantity evaluators for FE post-processing.

FE-side helpers that need model-layer access to compute quantities
derivable from a converged ``(U, xi)`` :class:`FEState`: the model's
flux at the integration points (:func:`evaluate_cauchy_at_ips`,
:func:`evaluate_heat_flux_at_ips`) and its local state variables
(:func:`evaluate_state_var_at_ips`).

Compose with :func:`cmad.io.results.ip_average_to_element` to reduce
``(n_elems, n_ip, *components)`` IP-level results to per-element
integration-measure-weighted means for Exodus element-field output;
the IP-level array is returned unreduced for callers that want the
raw per-Gauss-point data (diagnostics, projection-to-nodes work,
custom reductions).
"""
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.flatten_util import ravel_pytree
from numpy.typing import NDArray

from cmad.fem.assembly import _element_eq_indices
from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.global_residuals.balance_laws import cauchy_with_pressure
from cmad.global_residuals.interpolation import (
    interpolate_global_fields_at_ip,
)
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.models.mechanics_model import MechanicsModel
from cmad.models.thermal_model import ThermalModel
from cmad.models.var_types import (
    VarType,
    get_num_eqs,
    get_vector_from_sym_tensor,
)
from cmad.typing import JaxArray


def _evaluate_flux_at_ips(
        fe_problem: FEProblem,
        fe_state: FEState,
        step: int,
        block_name: str,
        flux_closed_form: Callable[..., JaxArray] | None,
        flux: Callable[..., JaxArray],
        n_comp: int,
        post: Callable[[JaxArray, GlobalFieldsAtPoint], JaxArray],
) -> NDArray[np.floating]:
    """A model flux at every (elem, IP) of a block, ``(n_elems, n_ip, n_comp)``.

    ``flux_closed_form(params, U_ip, U_ip_prev)`` serves a CLOSED_FORM
    block and ``flux(xi, xi_prev, params, U_ip, U_ip_prev)`` a COUPLED one
    (``xi`` from ``fe_state.xi_at(step, block_name)`` and ``xi_prev`` from
    the step before, zeros at ``step == 0``); ``post(value, U_ip)`` turns
    the model's value into the ``n_comp`` output components. ``U`` and
    ``U_prev`` come from ``fe_state.U_at(step)`` and the step before
    (zeros at ``step == 0``), interpolated per IP with the cached field
    shape values of ``fe_problem.geometry_cache[block_name]``.
    """
    U_global = jnp.asarray(fe_state.U_at(step))
    U_prev_global = (
        jnp.asarray(fe_state.U_at(step - 1)) if step > 0
        else jnp.zeros_like(U_global)
    )
    # The gather indices are derived from the FE mesh's connectivity
    # rather than read off the kernel arrays: the carrier's element axis
    # is padded for the device sharding (cmad.fem.sharding), and this
    # evaluator works on the stored history, which has the true element
    # counts.
    dof_map = fe_problem.dof_map
    connectivity_block = fe_problem.mesh.connectivity[
        fe_problem.mesh.element_blocks[block_name]
    ]
    n_elems_block = connectivity_block.shape[0]

    def gather(U_jax):
        gathered = []
        for field_idx in range(len(dof_map.field_layouts)):
            ndofs = int(dof_map.num_dofs_per_basis_fn[field_idx])
            eq = _element_eq_indices(
                connectivity_block, dof_map, field_idx=field_idx,
            )
            gathered.append(U_jax[eq.reshape(n_elems_block, -1, ndofs)])
        return gathered

    U_elem_block = gather(U_global)
    U_prev_elem_block = gather(U_prev_global)

    model = fe_problem.models_by_block[block_name]
    params = model.parameters.values
    mode = fe_problem.modes_by_block[block_name]
    var_names = fe_problem.gr.var_names
    num_blocks = len(fe_problem.block_shapes)

    geom_cache = fe_problem.geometry_cache[block_name]
    geom_per_elem = geom_cache.per_elem
    geom_shared = geom_cache.shared
    nips = int(geom_shared.quad_w.shape[0])

    def shapes_at(gpe, ip_idx):
        return [
            ShapeFunctionsAtIP(
                N=geom_shared.field_N_per_block[r][ip_idx],
                grad_N=gpe.field_grad_N_phys_per_block[r][ip_idx],
            )
            for r in range(num_blocks)
        ]

    if mode == GlobalResidualMode.CLOSED_FORM:
        if flux_closed_form is None:
            raise AttributeError(
                f"model {type(model).__name__} on block '{block_name}' "
                f"has no closed-form flux; use COUPLED mode for this "
                f"output"
            )
        closed_form = flux_closed_form

        def per_elem_closed_form(U_e, U_prev_e, gpe):
            values = jnp.zeros((nips, n_comp))
            for ip_idx in range(nips):
                shapes_ip = shapes_at(gpe, ip_idx)
                U_ip = interpolate_global_fields_at_ip(U_e, shapes_ip, var_names)
                U_prev_ip = interpolate_global_fields_at_ip(
                    U_prev_e, shapes_ip, var_names,
                )
                values = values.at[ip_idx].set(
                    post(closed_form(params, U_ip, U_prev_ip), U_ip),
                )
            return values

        values_block = vmap(per_elem_closed_form, in_axes=(0, 0, 0))(
            U_elem_block, U_prev_elem_block, geom_per_elem,
        )

    elif mode == GlobalResidualMode.COUPLED:
        xi_history = jnp.asarray(fe_state.xi_at(step, block_name))
        xi_prev_history = (
            jnp.asarray(fe_state.xi_at(step - 1, block_name))
            if step > 0 else jnp.zeros_like(xi_history)
        )
        _, unravel_xi = ravel_pytree(model._init_xi)

        def per_elem_coupled(U_e, U_prev_e, gpe, xi_per_ip, xi_prev_per_ip):
            values = jnp.zeros((nips, n_comp))
            for ip_idx in range(nips):
                shapes_ip = shapes_at(gpe, ip_idx)
                U_ip = interpolate_global_fields_at_ip(U_e, shapes_ip, var_names)
                U_prev_ip = interpolate_global_fields_at_ip(
                    U_prev_e, shapes_ip, var_names,
                )
                xi_blocks = unravel_xi(xi_per_ip[ip_idx])
                xi_prev_blocks = unravel_xi(xi_prev_per_ip[ip_idx])
                values = values.at[ip_idx].set(post(
                    flux(xi_blocks, xi_prev_blocks, params, U_ip, U_prev_ip),
                    U_ip,
                ))
            return values

        values_block = vmap(per_elem_coupled, in_axes=(0, 0, 0, 0, 0))(
            U_elem_block, U_prev_elem_block, geom_per_elem,
            xi_history, xi_prev_history,
        )

    else:
        raise ValueError(f"unsupported GlobalResidualMode: {mode}")

    return np.asarray(values_block)


def evaluate_cauchy_at_ips(
        fe_problem: FEProblem,
        fe_state: FEState,
        step: int,
        block_name: str,
) -> NDArray[np.floating]:
    """Cauchy stress at every (elem, IP) of a block.

    Returns ``(n_elems, n_ip, n_comp)`` in cmad sym-tensor vec order,
    where ``n_comp = get_num_eqs(SYM_TENSOR, ndims)`` is the in-plane
    component count: 6 (full 3x3) in 3D, 3 (``[xx, xy, yy]``) in 2D. The
    stress is the model's full 3x3 (with the pressure field as its
    hydrostatic part for a mixed formulation) reduced to its in-plane
    block, so a 2D problem writes a genuine 2D sym tensor.
    """
    model = fe_problem.models_by_block[block_name]
    if not isinstance(model, MechanicsModel):
        raise TypeError(
            f"cauchy output on block '{block_name}' needs a mechanics "
            f"model; got {type(model).__name__}"
        )
    is_mixed = getattr(fe_problem.gr, "mixed", False)
    ndims = fe_problem.mesh.nodes.shape[1]

    def post(sigma: JaxArray, U_ip: GlobalFieldsAtPoint) -> JaxArray:
        if is_mixed:
            sigma = cauchy_with_pressure(sigma, U_ip.fields["p"][0])
        return get_vector_from_sym_tensor(sigma[:ndims, :ndims], ndims)

    return _evaluate_flux_at_ips(
        fe_problem, fe_state, step, block_name,
        model.cauchy_closed_form, model.cauchy,
        get_num_eqs(VarType.SYM_TENSOR, ndims), post,
    )


def evaluate_heat_flux_at_ips(
        fe_problem: FEProblem,
        fe_state: FEState,
        step: int,
        block_name: str,
) -> NDArray[np.floating]:
    """Heat flux at every (elem, IP) of a block, ``(n_elems, n_ip, ndims)``."""
    model = fe_problem.models_by_block[block_name]
    if not isinstance(model, ThermalModel):
        raise TypeError(
            f"heat flux output on block '{block_name}' needs a thermal "
            f"model; got {type(model).__name__}"
        )
    ndims = fe_problem.mesh.nodes.shape[1]
    return _evaluate_flux_at_ips(
        fe_problem, fe_state, step, block_name,
        model.heat_flux_closed_form if model.supports_closed_form else None,
        model.heat_flux, ndims, lambda q, U_ip: q,
    )


def evaluate_state_var_at_ips(
        fe_problem: FEProblem,
        fe_state: FEState,
        step: int,
        block_name: str,
        resid_idx: int,
) -> NDArray[np.floating]:
    """One local state variable at every (elem, IP) of a COUPLED block.

    Reads the per-IP flat xi from ``fe_state.xi_at(step, block_name)``,
    unravels it to the model's state list, and returns residual block
    ``resid_idx`` as ``(n_elems, n_ip, n_comp)`` in the model's storage
    order. Compose with :func:`cmad.io.results.ip_average_to_element`.

    State variables are solved only in COUPLED mode, so this is meaningful
    only for COUPLED-bound blocks; the caller gates on mode.
    """
    model = fe_problem.models_by_block[block_name]
    xi_history = jnp.asarray(fe_state.xi_at(step, block_name))
    _, unravel = ravel_pytree(model._init_xi)
    nips = int(
        fe_problem.geometry_cache[block_name].shared.quad_w.shape[0],
    )

    def per_elem(xi_per_ip: JaxArray) -> JaxArray:
        return jnp.stack([
            jnp.atleast_1d(unravel(xi_per_ip[ip])[resid_idx])
            for ip in range(nips)
        ])

    return np.asarray(vmap(per_elem)(xi_history))


@dataclass(frozen=True)
class DerivedOutput:
    """A post-processed (non-state) FE output field.

    ``var_type`` is the field's intrinsic type. ``evaluator`` maps
    ``fe_problem, fe_state, step, block`` to per-IP values of shape
    ``(n_elems, n_ip, n_comp)``, composed with
    :func:`cmad.io.results.ip_average_to_element` by the caller. Adding a
    derived output is one entry in :data:`DERIVED_OUTPUT_REGISTRY`.
    """

    var_type: VarType
    evaluator: Callable[
        [FEProblem, FEState, int, str], NDArray[np.floating]
    ]


DERIVED_OUTPUT_REGISTRY: dict[str, DerivedOutput] = {
    "cauchy": DerivedOutput(VarType.SYM_TENSOR, evaluate_cauchy_at_ips),
    "heat flux": DerivedOutput(VarType.VECTOR, evaluate_heat_flux_at_ips),
}
