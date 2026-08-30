"""Reactive time step refinement.

A quasi-static solve that fails a step is driven again on a schedule
with that step cut back: its interval is split, and each cut may move
onto one of the times the caller supplies, when one lies inside the
interval. The rule is here; the loops that apply it live with the
callers that drive the solve.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TimeRefinement:
    """How far to cut back: a failed step's interval is split into
    ``factor`` pieces, at most ``max_depth`` times, zero turning the
    refinement off."""

    factor: int = 2
    max_depth: int = 3

    @classmethod
    def from_deck(cls, section: dict[str, Any] | None) -> TimeRefinement:
        section = section or {}
        return cls(
            factor=int(section.get("factor", 2)),
            max_depth=int(section.get("max depth", 3)),
        )


def refine_schedule(
        schedule: Sequence[float] | NDArray[np.float64],
        step: int,
        factor: int,
        snap_to: NDArray[np.float64] | None = None,
        rtol: float = 1.0e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split the interval ending at ``step`` into ``factor`` pieces.

    Each cut moves to the nearest of ``snap_to`` strictly inside the
    interval, within ``rtol`` of the schedule span, and stays where it
    is when there is none. Returns the new schedule and the times it
    gained.
    """
    times = np.asarray(schedule, dtype=np.float64)
    tol = rtol * float(times[-1] - times[0])
    t_lo, t_hi = float(times[step]), float(times[step + 1])
    inside = (
        np.empty(0) if snap_to is None
        else snap_to[(snap_to > t_lo + tol) & (snap_to < t_hi - tol)]
    )
    cuts: list[float] = []
    for fraction in np.arange(1, factor) / factor:
        cut = t_lo + fraction * (t_hi - t_lo)
        if inside.size:
            cut = float(inside[np.argmin(np.abs(inside - cut))])
        cuts.append(cut)
    refined = np.unique(np.concatenate([times, cuts]))
    inserted = refined[~np.isin(refined, times)]
    return refined, inserted
