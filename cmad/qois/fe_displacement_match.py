"""Time- and space-averaged displacement-mismatch QoI for FE problems."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp

from cmad.fem.assembly import _gather_element_U
from cmad.fem.precompute import compute_ip_quadrature_weights
from cmad.io.qoi_data import load_displacement_data
from cmad.io.registry import register_qoi
from cmad.qois.fe_qoi import FEQoI, StepContribution
from cmad.typing import JaxArray, Params

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem
    from cmad.fem.kernel_arrays import FEKernelArrays


@register_qoi("fe_displacement_match")
class FEDisplacementMatch(FEQoI):
    r"""Time- and space-averaged squared displacement mismatch.

    .. math::

       J = \frac{w}{T \, |\Omega|}
            \sum_n \Delta t_n \int_\Omega |u_n - u^\mathrm{data}_n|^2 \, dV

    Operates on the residual block whose ``var_name`` is ``"u"``. ``w``
    is a scalar deck weight; ``u^\mathrm{data}`` is per-step nodal
    displacement of shape ``(num_steps, num_nodes, ndims)``.
    """

    problem_type: ClassVar[str] = "fe"

    def __init__(
            self,
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
            data: JaxArray,
            weight: float = 1.0,
    ) -> None:
        var_names = list(fe_problem.gr.var_names)
        try:
            r_disp = var_names.index("u")
        except ValueError as exc:
            raise ValueError(
                f"FEDisplacementMatch requires a residual block with "
                f"var_name 'u'; got var_names={var_names}"
            ) from exc

        num_steps = len(t_schedule)
        data_arr = jnp.asarray(data, dtype=jnp.float64)
        if data_arr.shape[0] != num_steps:
            raise ValueError(
                f"FEDisplacementMatch: data has {data_arr.shape[0]} steps "
                f"but the time schedule has {num_steps} (expected one "
                f"displacement field per schedule time, including the "
                f"initial time)"
            )
        data_flat = data_arr.reshape(num_steps, -1)
        num_total_dofs = fe_problem.dof_map.num_total_dofs
        if data_flat.shape[1] != num_total_dofs:
            raise ValueError(
                f"FEDisplacementMatch: data flattens to {data_flat.shape[1]} "
                f"dofs/step but the problem has {num_total_dofs} total dofs; "
                f"this QoI supports single-displacement-field problems "
                f"(num_total_dofs == num_nodes * ndims)"
            )

        ip_weights = compute_ip_quadrature_weights(fe_problem.geometry_cache)
        total_volume = float(sum(arr.sum() for arr in ip_weights.values()))
        T = float(t_schedule[-1]) - float(t_schedule[0])

        self._fe_problem = fe_problem
        self._r_disp = r_disp
        self._field_idx_disp = fe_problem.field_idx_per_block[r_disp]
        self._norm_factor = float(weight) / (T * total_volume)
        self._data_flat = data_flat
        self._t_schedule = jnp.asarray(t_schedule, dtype=jnp.float64)

    @classmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
    ) -> FEDisplacementMatch:
        data = jnp.asarray(
            load_displacement_data(qoi_section), dtype=jnp.float64,
        )
        weight = float(qoi_section.get("weight", 1.0))
        return cls(fe_problem, t_schedule, data, weight)

    def step_contribution(
            self,
            params_by_block: Mapping[str, Params],
            fe_arrays: FEKernelArrays,
    ) -> StepContribution:
        del params_by_block  # params enter only through the solved state U
        fe_problem = self._fe_problem
        r_disp = self._r_disp
        field_idx_disp = self._field_idx_disp
        norm_factor = self._norm_factor
        data_flat = self._data_flat
        t_schedule = self._t_schedule

        block_data: list[tuple[str, JaxArray, JaxArray]] = []
        for block_name in fe_problem.models_by_block:
            geom_cache = fe_arrays.geometry_cache[block_name]
            N_disp = geom_cache.shared.field_N_per_block[r_disp]
            quad_w = geom_cache.shared.quad_w
            iso_jac_det = geom_cache.per_elem.iso_jac_det
            weighted_iso_jac_det = iso_jac_det * quad_w
            block_data.append(
                (block_name, N_disp, weighted_iso_jac_det),
            )

        def _closure(
                U: JaxArray,
                U_prev: JaxArray,
                xi: Mapping[str, JaxArray],
                xi_prev: Mapping[str, JaxArray],
                t: JaxArray,
                t_prev: JaxArray,
        ) -> JaxArray:
            del U_prev, xi, xi_prev
            dt = t - t_prev
            step = jnp.argmin(jnp.abs(t_schedule - t))
            U_data = data_flat[step]
            total_integral = jnp.zeros(())
            for (block_name, N_disp,
                 weighted_iso_jac_det) in block_data:
                U_elem = _gather_element_U(U, fe_arrays, block_name)
                U_data_elem = _gather_element_U(
                    U_data, fe_arrays, block_name,
                )
                diff_per_elem = (
                    U_elem[field_idx_disp] - U_data_elem[field_idx_disp]
                )
                diff_at_ip = jnp.einsum(
                    "pa,eak->epk", N_disp, diff_per_elem,
                )
                diff_sq = jnp.sum(diff_at_ip * diff_at_ip, axis=-1)
                block_integral = jnp.sum(diff_sq * weighted_iso_jac_det)
                total_integral = total_integral + block_integral
            return norm_factor * dt * total_integral

        return _closure
