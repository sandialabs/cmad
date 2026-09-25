"""Base class for the FE QoIs that compare a prediction against measured
data."""
from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, TypeAlias

from cmad.qois.fe_qoi import FEQoI, MatchTimes, StepContribution
from cmad.typing import JaxArray

if TYPE_CHECKING:
    from cmad.models.global_fields import StepTime

SquaredMismatch: TypeAlias = Callable[[JaxArray, int | JaxArray], JaxArray]
"""``mismatch(prediction, step)``: a term's squared mismatch against its
data at match time ``step``, integrated over the matched region."""


def compute_data_mean_square(
        mismatch: SquaredMismatch,
        zero_prediction: JaxArray,
        match_times: MatchTimes,
        measure: float,
) -> float:
    """The term's average applied to its data alone: ``mismatch`` at zero
    prediction, summed over the match times with their weights, over the
    span and ``measure``."""
    total = sum(
        float(weight) * float(mismatch(zero_prediction, step))
        for step, weight in enumerate(match_times.weights)
    )
    return total / (match_times.span * measure)


class FEMatchTerm(FEQoI, ABC):
    """An FE QoI that averages a squared mismatch against data over the
    match times and the matched region, divided by the same average
    applied to the data alone, so its value is a relative error."""

    def __init__(self, weight: float, match_times: MatchTimes) -> None:
        super().__init__(weight)
        self._match_times = match_times
        self.data_mean_square: float | None = None

    def _normalize_by_data_mean_square(
            self,
            mismatch: SquaredMismatch,
            zero_prediction: JaxArray,
            measure: float,
    ) -> None:
        """Set the term's scale from its data; raises when the data is
        zero."""
        self.data_mean_square = compute_data_mean_square(
            mismatch, zero_prediction, self._match_times, measure,
        )
        if self.data_mean_square <= 0.0:
            raise ValueError(
                f"{type(self).__name__}: the data is zero over the match "
                "times and the matched region, so the mismatch has no scale"
            )
        self._norm_factor = 1.0 / (
            self._match_times.span * measure * self.data_mean_square
        )

    def _step_closure(self, mismatch: SquaredMismatch) -> StepContribution:
        """The per-step closure: ``mismatch`` evaluated on the field at the
        step's match time, weighted and normalized, zero off the match
        times."""
        norm_factor = self._norm_factor
        match_times = self._match_times

        def _closure(
                U: JaxArray,
                U_prev: JaxArray,
                xi: Mapping[str, JaxArray],
                xi_prev: Mapping[str, JaxArray],
                step_time: StepTime,
        ) -> JaxArray:
            del U_prev, xi, xi_prev
            step, weight = match_times.index_and_weight(step_time.t)
            return norm_factor * weight * mismatch(U, step)

        return _closure

    def data_mean_squares(self) -> dict[str, float]:
        if self.data_mean_square is None:
            return {}
        (name,) = self.accumulated_qoi_names()
        return {name: self.data_mean_square}
