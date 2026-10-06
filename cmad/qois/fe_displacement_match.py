"""Time- and space-averaged displacement-mismatch QoI for FE problems."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp

from cmad.io.qoi_data import (
    calibration_data_roi,
    load_calibration_data,
    load_displacement_data,
    load_match_times,
    load_roi,
)
from cmad.qois.fe_field_match import FEFieldMatch
from cmad.qois.fe_qoi import MatchTimes

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem


class FEDisplacementMatch(FEFieldMatch):
    r"""Time- and space-averaged squared displacement mismatch.

    .. math::

       J = \frac{1}{T \, |\Omega| \, D}
            \sum_n \Delta t_n \int_\Omega |u_n - u^\mathrm{data}_n|^2 \, dV

    over the match times, :math:`\Delta t_n` being each one's weight
    (:class:`cmad.qois.fe_qoi.MatchTimes`, the time schedule by
    default), :math:`T` their span, and :math:`D` the same average
    applied to :math:`u^\mathrm{data}` alone
    (:class:`cmad.qois.fe_match_term.FEMatchTerm`). Operates on the
    residual block whose ``var_name`` is ``"u"``; the data, the region,
    and the integration rule are those of
    :class:`cmad.qois.fe_field_match.FEFieldMatch`.
    """

    field_name: ClassVar[str] = "u"

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
