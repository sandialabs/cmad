from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.io.registry import register_qoi
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.models.model import Model
from cmad.qois.qoi import QoI
from cmad.typing import (
    CauchyFn,
    JaxArray,
    Params,
    StateList,
    Step,
)


@register_qoi("uniaxial_calibration")
class UniaxialCalibration(QoI):
    def __init__(
            self, model: Model,
            data: NDArray[np.floating], weight: NDArray[np.floating],
            uniaxial_stress_idx: int, stretch_var_idx: int,
    ) -> None:
        self._model = model
        assert data.shape == weight.shape
        self._data = data
        self._weight = weight
        partial_qoi = partial(self._qoi,
                              cauchy_fun=model.cauchy,
                              uniaxial_stress_idx=uniaxial_stress_idx,
                              stretch_var_idx=stretch_var_idx)
        super().__init__(partial_qoi)


    @classmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            model: Model,
            data: NDArray[np.floating],
            weight: NDArray[np.floating],
    ) -> "UniaxialCalibration":
        return cls(
            model=model,
            data=data,
            weight=weight,
            uniaxial_stress_idx=qoi_section["uniaxial_stress_idx"],
            stretch_var_idx=qoi_section["stretch_var_idx"],
        )


    def update_data(self, data: NDArray[np.floating]) -> None:
        old_shape = self._data.shape
        assert data.shape == old_shape
        self._data = data


    def data_at_step(self, step: Step) -> NDArray[np.floating]:
        return self._data[..., step]


    def weight_at_step(self, step: Step) -> NDArray[np.floating]:
        return self._weight[:, step]


    @staticmethod
    def _qoi(
            xi: StateList, xi_prev: StateList, params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            data_at_step: JaxArray, weight_at_step: JaxArray,
            cauchy_fun: CauchyFn,
            uniaxial_stress_idx: int, stretch_var_idx: int,
    ) -> JaxArray:

        sigma = cauchy_fun(xi, xi_prev, params, U, U_prev)
        uniaxial_sigma = sigma[uniaxial_stress_idx, uniaxial_stress_idx]
        off_axis_strains = jnp.r_[xi[stretch_var_idx][0] - 1.,
            xi[stretch_var_idx][1] - 1.]
        pred = jnp.r_[uniaxial_sigma, off_axis_strains]
        mismatch = (pred - data_at_step) * weight_at_step
        return 0.5 * jnp.sum(mismatch * mismatch)
