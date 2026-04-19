from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.io.registry import register_qoi
from cmad.models.model import Model
from cmad.qois.qoi import QoI
from cmad.typing import (
    CauchyFn,
    GlobalList,
    JaxArray,
    Params,
    StateList,
    Step,
)


@register_qoi("calibration")
class Calibration(QoI):
    def __init__(
            self, model: Model,
            data: NDArray[np.floating], weight: NDArray[np.floating],
    ) -> None:
        self._model = model
        self._data = data
        # assume weight constant and same shape as cauchy stress
        assert weight.shape == (3, 3)
        self._weight = weight
        partial_qoi = partial(self._qoi, cauchy_fun=model.cauchy)
        super().__init__(partial_qoi)


    @classmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            model: Model,
            data: NDArray[np.floating],
            weight: NDArray[np.floating],
    ) -> "Calibration":
        return cls(model, data, weight)


    def data_at_step(self, step: Step) -> NDArray[np.floating]:
        return self._data[..., step]


    def weight_at_step(self, step: Step) -> NDArray[np.floating]:
        return self._weight


    # _qoi not an abstract class, as signatures may be class-specific
    @staticmethod
    def _qoi(
            xi: StateList, xi_prev: StateList, params: Params,
            u: GlobalList, u_prev: GlobalList,
            data_at_step: JaxArray, weight_at_step: JaxArray,
            cauchy_fun: CauchyFn,
    ) -> JaxArray:

        cauchy = cauchy_fun(xi, xi_prev, params, u, u_prev)
        mismatch = weight_at_step * (cauchy - data_at_step)
        return 0.5 * jnp.sum(mismatch * mismatch)
