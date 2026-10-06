"""Time and space averaged temperature mismatch QoI for FE problems."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp

from cmad.io.qoi_data import (
    calibration_data_roi,
    load_calibration_data,
    load_match_times,
)
from cmad.qois.fe_field_match import FEFieldMatch
from cmad.qois.fe_qoi import MatchTimes

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem


class FETemperatureMatch(FEFieldMatch):
    """The squared temperature mismatch averaged over the match times and
    the region, divided by the average of the squared measured
    temperature change from the initial frame, so its value is a
    relative error. Operates on the residual block whose ``var_name`` is
    ``"T"`` and reads a calibration data archive, whose first row is the
    initial frame.
    """

    field_name: ClassVar[str] = "T"

    @classmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
    ) -> FETemperatureMatch:
        match = MatchTimes.from_times(load_match_times(qoi_section, t_schedule))
        store = load_calibration_data(qoi_section)
        store.check_mesh(int(fe_problem.mesh.nodes.shape[0]))
        frames = store.frames_at(match.times)
        return cls(
            fe_problem, t_schedule,
            jnp.asarray(store.rows(frames, field="T")),
            float(qoi_section.get("weight", 1.0)),
            sideset=qoi_section.get("sideset"),
            roi=calibration_data_roi(store, fe_problem.ndims),
            match_times=match, node_ids=store.node_ids,
            initial_field=jnp.asarray(store.rows([0], field="T")[0]),
        )
