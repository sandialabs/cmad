"""Time- and space-averaged displacement-mismatch QoI for FE problems."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.assembly import _gather_element_U
from cmad.fem.precompute import compute_ip_quadrature_weights
from cmad.io.qoi_data import load_displacement_data, load_roi
from cmad.io.registry import register_qoi
from cmad.qois.fe_qoi import FEQoI, StepContribution
from cmad.qois.surface_match import (
    surface_groups_and_norm,
    surface_l2_step_closure,
)
from cmad.typing import JaxArray, Params

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem
    from cmad.fem.kernel_arrays import FEKernelArrays
    from cmad.models.global_fields import StepTime


def _element_masks(
        fe_problem: FEProblem, roi: NDArray[np.intp] | None,
) -> dict[str, JaxArray] | None:
    """Per-block 0/1 element masks selecting the region of interest.

    ``None`` when there is no region of interest, which integrates
    everywhere. Shaped ``(n_elems_in_block, 1)`` to broadcast against the
    ``(n_elems, n_ip)`` integration measure.
    """
    if roi is None:
        return None
    keep = np.zeros(fe_problem.mesh.connectivity.shape[0], dtype=bool)
    keep[roi] = True
    return {
        block: jnp.asarray(keep[elems], dtype=jnp.float64)[:, None]
        for block, elems in fe_problem.mesh.element_blocks.items()
    }


@register_qoi("fe_displacement_match")
class FEDisplacementMatch(FEQoI):
    r"""Time- and space-averaged squared displacement mismatch.

    .. math::

       J = \frac{w}{T \, |\Omega|}
            \sum_n \Delta t_n \int_\Omega |u_n - u^\mathrm{data}_n|^2 \, dV

    Operates on the residual block whose ``var_name`` is ``"u"``. ``w``
    is a scalar deck weight; ``u^\mathrm{data}`` is per-step nodal
    displacement of shape ``(num_steps, num_nodes, ndims)``. With a
    ``sideset``, the integral and its normalizing measure are over that
    sideset's surface rather than the volume; with a ``roi``, over those
    elements.
    """

    problem_type: ClassVar[str] = "fe"

    def __init__(
            self,
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
            data: JaxArray,
            weight: float = 1.0,
            sideset: str | None = None,
            roi: NDArray[np.intp] | None = None,
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

        self._fe_problem = fe_problem
        self._r_disp = r_disp
        self._field_idx_disp = fe_problem.field_idx_per_block[r_disp]
        self._data_flat = data_flat
        self._t_schedule = jnp.asarray(t_schedule, dtype=jnp.float64)

        if sideset is None:
            ip_weights = compute_ip_quadrature_weights(
                fe_problem.geometry_cache,
            )
            self._element_mask = _element_masks(fe_problem, roi)
            if self._element_mask is None:
                volume = float(sum(arr.sum() for arr in ip_weights.values()))
            else:
                volume = float(sum(
                    (arr * np.asarray(self._element_mask[block])).sum()
                    for block, arr in ip_weights.items()))
            if volume <= 0.0:
                raise ValueError(
                    "FEDisplacementMatch: the region of interest selects no "
                    "elements, so the mismatch has nothing to average over"
                )
            span = float(t_schedule[-1]) - float(t_schedule[0])
            self._surface_groups = None
            self._norm_factor = float(weight) / (span * volume)
        else:
            if roi is not None:
                raise ValueError(
                    "FEDisplacementMatch: give a sideset or a region of "
                    "interest, not both"
                )
            self._element_mask = None
            self._surface_groups, self._norm_factor = surface_groups_and_norm(
                fe_problem, sideset, "u", weight, t_schedule,
            )

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
        sideset = qoi_section.get("sideset")
        roi = None
        if "roi_file" in qoi_section:
            ndims = fe_problem.ndims
            if ndims != 2:
                raise NotImplementedError(
                    "FEDisplacementMatch: roi_file is supported on 2D meshes, "
                    "where the region of interest is elements. A 3D mesh is "
                    "measured on a face, so its region of interest is sides, "
                    "which needs build_surface_integration_groups to accept "
                    "side pairs rather than a sideset name"
                )
            roi = load_roi(qoi_section, ndims)
        return cls(
            fe_problem, t_schedule, data, weight, sideset=sideset, roi=roi)

    def step_contribution(
            self,
            params_by_block: Mapping[str, Params],
            fe_arrays: FEKernelArrays,
    ) -> StepContribution:
        del params_by_block  # params enter only through the solved state U
        if self._surface_groups is not None:
            return surface_l2_step_closure(
                self._surface_groups, self._data_flat, self._t_schedule,
                self._norm_factor,
            )
        fe_problem = self._fe_problem
        r_disp = self._r_disp
        field_idx_disp = self._field_idx_disp
        norm_factor = self._norm_factor
        data_flat = self._data_flat
        t_schedule = self._t_schedule

        element_mask = self._element_mask
        block_data: list[tuple[str, JaxArray, JaxArray]] = []
        for block_name in fe_problem.models_by_block:
            geom_cache = fe_arrays.geometry_cache[block_name]
            N_disp = geom_cache.shared.field_N_per_block[r_disp]
            quad_w = geom_cache.shared.quad_w
            iso_jac_det = geom_cache.per_elem.iso_jac_det
            weighted_iso_jac_det = iso_jac_det * quad_w
            if element_mask is not None:
                weighted_iso_jac_det = (
                    weighted_iso_jac_det * element_mask[block_name])
            block_data.append(
                (block_name, N_disp, weighted_iso_jac_det),
            )

        def _closure(
                U: JaxArray,
                U_prev: JaxArray,
                xi: Mapping[str, JaxArray],
                xi_prev: Mapping[str, JaxArray],
                step_time: StepTime,
        ) -> JaxArray:
            del U_prev, xi, xi_prev
            dt = step_time.dt
            step = jnp.argmin(jnp.abs(t_schedule - step_time.t))
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
