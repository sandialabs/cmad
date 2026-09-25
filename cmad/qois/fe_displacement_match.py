"""Time- and space-averaged displacement-mismatch QoI for FE problems."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.dof import dof_physical_coords
from cmad.io.qoi_data import (
    calibration_data_roi,
    load_calibration_data,
    load_displacement_data,
    load_match_times,
    load_roi,
)
from cmad.qois.cell_match import (
    CellIntegrationArrays,
    cell_arrays_and_measure,
    cell_squared_mismatch,
)
from cmad.qois.fe_match_term import FEMatchTerm, SquaredMismatch
from cmad.qois.fe_qoi import MatchTimes, StepContribution
from cmad.qois.surface_match import (
    surface_groups_and_area,
    surface_squared_mismatch,
)
from cmad.typing import JaxArray, Params

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem
    from cmad.fem.kernel_arrays import FEKernelArrays


class FEDisplacementMatch(FEMatchTerm):
    r"""Time- and space-averaged squared displacement mismatch.

    .. math::

       J = \frac{1}{T \, |\Omega| \, D}
            \sum_n \Delta t_n \int_\Omega |u_n - u^\mathrm{data}_n|^2 \, dV

    over the match times, :math:`\Delta t_n` being each one's weight
    (:class:`cmad.qois.fe_qoi.MatchTimes`, the time schedule by
    default), :math:`T` their span, and :math:`D` the same average
    applied to :math:`u^\mathrm{data}` alone
    (:class:`cmad.qois.fe_match_term.FEMatchTerm`). Operates on the
    residual block whose ``var_name`` is ``"u"``. ``u^\mathrm{data}`` is
    the nodal displacement at each match time on
    ``node_ids`` (every node when ``None``), shaped
    ``(num_match_times, num_nodes, ndims)``. With a ``sideset``, the
    integral and its normalizing measure are over that sideset's surface
    rather than over the whole mesh; with a ``roi``, over the elements or
    the sides it holds. The integral over elements uses a rule exact for
    the square of a linear field, whatever rule the assembly uses
    (:mod:`cmad.qois.cell_match`).
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
            *,
            match_times: MatchTimes | None = None,
            node_ids: NDArray[np.intp] | None = None,
    ) -> None:
        var_names = list(fe_problem.gr.var_names)
        try:
            r_disp = var_names.index("u")
        except ValueError as exc:
            raise ValueError(
                f"FEDisplacementMatch requires a residual block with "
                f"var_name 'u'; got var_names={var_names}"
            ) from exc

        match = (
            MatchTimes.from_times(t_schedule) if match_times is None
            else match_times
        )
        super().__init__(weight, match)
        num_match = int(match.times.shape[0])
        data_arr = np.asarray(data, dtype=np.float64)
        if data_arr.shape[0] != num_match:
            raise ValueError(
                f"FEDisplacementMatch: data has {data_arr.shape[0]} frames "
                f"but there are {num_match} match times (expected one "
                f"displacement field per match time, the initial time "
                f"included)"
            )
        # The measurement covers the displacement field alone, so it is
        # scattered into that field's equations. Any other field the
        # problem carries, a mixed formulation's pressure among them, is
        # left at zero and never read.
        _coords, eq = dof_physical_coords(
            fe_problem.mesh, fe_problem.dof_map, "u",
        )
        if node_ids is not None:
            eq = eq[np.asarray(node_ids, dtype=np.intp)]
        if data_arr.shape[1:] != eq.shape:
            raise ValueError(
                f"FEDisplacementMatch: data is {data_arr.shape[1:]} per "
                f"frame but field 'u' has {eq.shape} (basis coefficients, "
                f"components) on the given nodes"
            )
        data_flat = np.zeros((num_match, fe_problem.dof_map.num_total_dofs))
        data_flat[:, eq.reshape(-1)] = data_arr.reshape(num_match, -1)

        self._field_idx_disp = fe_problem.field_idx_per_block[r_disp]
        self._data_flat = jnp.asarray(data_flat, dtype=jnp.float64)

        if sideset is not None and roi is not None:
            raise ValueError(
                "FEDisplacementMatch: give a sideset or a region of "
                "interest, not both"
            )
        # A 2D mesh is the measured surface itself, so a region of
        # interest over it is elements, integrated the way the whole mesh
        # is. A 3D mesh is measured on its surface, so its region of
        # interest is sides, integrated over those.
        sides: str | NDArray[np.intp] | None = sideset
        if roi is not None and roi.ndim == 2:
            sides = roi

        self._cell_arrays: dict[str, CellIntegrationArrays] | None
        if sides is None:
            self._surface_groups = None
            self._cell_arrays, measure = cell_arrays_and_measure(
                fe_problem, "u", roi,
            )
        else:
            self._cell_arrays = None
            self._surface_groups, measure = surface_groups_and_area(
                fe_problem, sides, "u",
            )
        self._normalize_by_data_mean_square(
            self._squared_mismatch(fe_problem.kernel_arrays),
            jnp.zeros(fe_problem.dof_map.num_total_dofs),
            measure,
        )

    @classmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
    ) -> FEDisplacementMatch:
        match = MatchTimes.from_times(load_match_times(qoi_section, t_schedule))
        weight = float(qoi_section.get("weight", 1.0))
        sideset = qoi_section.get("sideset")
        if "calibration_data_file" in qoi_section:
            store = load_calibration_data(qoi_section)
            store.check_mesh(int(fe_problem.mesh.nodes.shape[0]))
            frames = store.frames_at(match.times)
            return cls(
                fe_problem, t_schedule, jnp.asarray(store.rows(frames)),
                weight, sideset=sideset,
                roi=calibration_data_roi(store, fe_problem.ndims),
                match_times=match, node_ids=store.node_ids,
            )
        data = jnp.asarray(
            load_displacement_data(qoi_section), dtype=jnp.float64,
        )
        roi = None
        if "roi_file" in qoi_section:
            roi = load_roi(qoi_section, fe_problem.ndims)
        return cls(
            fe_problem, t_schedule, data, weight, sideset=sideset, roi=roi,
            match_times=match,
        )

    def _squared_mismatch(self, fe_arrays: FEKernelArrays) -> SquaredMismatch:
        """``mismatch(U, step)``: the squared difference between ``U`` and
        the data at match time ``step``, integrated over the matched
        region."""
        if self._surface_groups is not None:
            return surface_squared_mismatch(
                self._surface_groups, self._data_flat,
            )
        assert self._cell_arrays is not None
        return cell_squared_mismatch(
            self._cell_arrays, self._field_idx_disp, fe_arrays,
            self._data_flat,
        )

    def step_contribution(
            self,
            params_by_block: Mapping[str, Params],
            fe_arrays: FEKernelArrays,
    ) -> StepContribution:
        del params_by_block  # params enter only through the solved state U
        return self._step_closure(self._squared_mismatch(fe_arrays))
