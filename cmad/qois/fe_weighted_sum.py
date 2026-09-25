"""Composite FE QoI: a sum of sub-QoIs, each carrying its own weight."""
from __future__ import annotations

import jax.numpy as jnp

from cmad.qois.fe_term_sum import FETermSum
from cmad.typing import JaxArray


class FEWeightedSum(FETermSum):
    """Sum of FE sub-QoIs, each carrying its own deck ``weight``.

    Each entry in the deck ``terms`` list builds a sub-QoI via its own
    ``from_deck``; this QoI's value is the weighted sum of theirs.
    Relative weighting between heterogeneous terms (e.g. a displacement
    match and a load match) is each term's own ``weight``, so the composite
    holds no separate weight.
    """

    def combine(self, accumulated_qois: JaxArray) -> JaxArray:
        return jnp.sum(self._weights * accumulated_qois)
