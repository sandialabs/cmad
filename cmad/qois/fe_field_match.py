"""Base class for the FE QoIs that compare one nodal field against
measured data over the match times and a region."""
from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.dof import dof_physical_coords
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


class FEFieldMatch(FEMatchTerm, ABC):
    """An FE QoI that averages the squared mismatch of the nodal field
    ``field_name`` against data over the match times and a region,
    divided by the same average of the data's change from
    ``initial_field``.

    ``data`` holds the field at each match time on ``node_ids`` (every
    node when ``None``), shaped ``(num_match_times, num_nodes,
    num_components)``; ``initial_field`` is the field on the same nodes
    at the initial time the data changes from, zero when ``None``. With
    a ``sideset``, the integral and its normalizing measure are over that
    sideset's surface rather than over the whole mesh; with a ``roi``,
    over the elements or the sides it holds. The integral over elements
    uses a rule exact for the square of a linear field, whatever rule the
    assembly uses (:mod:`cmad.qois.cell_match`).
    """

    field_name: ClassVar[str]

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
            initial_field: JaxArray | None = None,
    ) -> None:
        name = type(self).__name__
        field = self.field_name
        var_names = list(fe_problem.gr.var_names)
        try:
            r_field = var_names.index(field)
        except ValueError as exc:
            raise ValueError(
                f"{name} requires a residual block with var_name "
                f"'{field}'; got var_names={var_names}"
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
                f"{name}: data has {data_arr.shape[0]} frames but there "
                f"are {num_match} match times (expected one field per "
                f"match time, the initial time included)"
            )
        # The measurement covers one field, so it is scattered into that
        # field's equations. Any other field the problem carries, a mixed
        # formulation's pressure among them, is left at zero and never
        # read.
        _coords, eq = dof_physical_coords(
            fe_problem.mesh, fe_problem.dof_map, field,
        )
        if node_ids is not None:
            eq = eq[np.asarray(node_ids, dtype=np.intp)]
        if data_arr.shape[1:] != eq.shape:
            raise ValueError(
                f"{name}: data is {data_arr.shape[1:]} per frame but "
                f"field '{field}' has {eq.shape} (basis coefficients, "
                f"components) on the given nodes"
            )
        num_total_dofs = fe_problem.dof_map.num_total_dofs
        data_flat = np.zeros((num_match, num_total_dofs))
        data_flat[:, eq.reshape(-1)] = data_arr.reshape(num_match, -1)
        initial_flat = np.zeros(num_total_dofs)
        if initial_field is not None:
            initial_arr = np.asarray(initial_field, dtype=np.float64)
            if initial_arr.shape != eq.shape:
                raise ValueError(
                    f"{name}: the initial field is {initial_arr.shape} but "
                    f"field '{field}' has {eq.shape} on the given nodes"
                )
            initial_flat[eq.reshape(-1)] = initial_arr.reshape(-1)

        self._field_idx = fe_problem.field_idx_per_block[r_field]
        self._data_flat = jnp.asarray(data_flat, dtype=jnp.float64)

        if sideset is not None and roi is not None:
            raise ValueError(
                f"{name}: give a sideset or a region of interest, not both"
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
                fe_problem, field, roi,
            )
        else:
            self._cell_arrays = None
            self._surface_groups, measure = surface_groups_and_area(
                fe_problem, sides, field,
            )
        self._normalize_by_data_mean_square(
            self._squared_mismatch(fe_problem.kernel_arrays),
            jnp.asarray(initial_flat, dtype=jnp.float64),
            measure,
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
            self._cell_arrays, self._field_idx, fe_arrays,
            self._data_flat,
        )

    def step_contribution(
            self,
            params_by_block: Mapping[str, Params],
            fe_arrays: FEKernelArrays,
    ) -> StepContribution:
        del params_by_block  # params enter only through the solved state U
        return self._step_closure(self._squared_mismatch(fe_arrays))
