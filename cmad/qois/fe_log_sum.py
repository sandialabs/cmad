"""Composite FE QoI: a weighted sum of the logarithms of sub-QoIs."""
from __future__ import annotations

import jax.numpy as jnp

from cmad.qois.fe_term_sum import FETermSum
from cmad.typing import JaxArray


class FELogSum(FETermSum):
    """Sum of the logarithms of FE sub-QoIs, each log carrying its own
    deck ``weight``.

    Its gradient is that of the weighted sum with each term's weight
    divided by the term's own value, so equal weights give the optimum
    that rebalancing the terms at the optimum would reach.
    """

    def combine(self, accumulated_qois: JaxArray) -> JaxArray:
        return jnp.sum(self._weights * jnp.log(accumulated_qois))
