"""Temporal interpolation for a nodal history."""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.finite_element import FiniteElement
from cmad.typing import JaxArray, Scalar


@dataclass(frozen=True)
class PrescribedField:
    """A field given at every node over time, read beside the unknown fields.

    ``data`` is ``(num_times, num_nodes, num_components)`` and ``times``
    its strictly increasing times.
    """

    name: str
    finite_element: FiniteElement
    data: NDArray[np.floating]
    times: NDArray[np.floating]

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(
                f"PrescribedField '{self.name}' data must be (num_times, "
                f"num_nodes, num_components); got shape {self.data.shape}"
            )
        if self.data.shape[0] != self.times.shape[0]:
            raise ValueError(
                f"PrescribedField '{self.name}' has {self.data.shape[0]} "
                f"entries in data but {self.times.shape[0]} times"
            )
        gaps = np.diff(self.times)
        if gaps.size and not np.all(gaps > 0.0):
            raise ValueError(
                f"PrescribedField '{self.name}' times must be strictly "
                f"increasing; found a step of {gaps.min():.6g}"
            )


def interpolate_nodal_history(
        data: JaxArray, times: JaxArray, t: Scalar,
) -> JaxArray:
    """Interpolate a nodal history linearly in time, the ends held.

    ``data`` is ``(num_times, N, num_components)`` and ``times`` the times
    the history is given at. One entry holds for the whole history.
    """
    num_times = data.shape[0]
    if num_times == 1:
        return data[0]
    last = num_times - 2
    lo = jnp.clip(jnp.searchsorted(times, t, side="right") - 1, 0, last)
    span = times[lo + 1] - times[lo]
    weight = jnp.clip((t - times[lo]) / span, 0.0, 1.0)
    before = jnp.take(data, lo, axis=0)
    after = jnp.take(data, lo + 1, axis=0)
    return before + weight * (after - before)
