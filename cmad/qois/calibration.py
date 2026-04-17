from functools import partial

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

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


class Calibration(QoI):
    def __init__(
            self, model: Model, global_state: NDArray[np.floating],
            data: NDArray[np.floating], weight: NDArray[np.floating],
    ) -> None:
        self._model = model
        assert global_state.shape[-1] == data.shape[-1]
        self._global_state = global_state
        self._data = data
        # assume weight constant and same shape as cauchy stress
        assert weight.shape == (3, 3)
        self._weight = weight
        partial_qoi = partial(self._qoi, cauchy_fun=model.cauchy)
        super().__init__(partial_qoi)


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


class UniaxialCalibration(QoI):
    def __init__(
            self, model: Model, global_state: NDArray[np.floating],
            data: NDArray[np.floating], weight: NDArray[np.floating],
            uniaxial_stress_idx: int, stretch_var_idx: int,
    ) -> None:
        self._model = model
        assert global_state.shape[-1] == data.shape[-1]
        assert data.shape == weight.shape
        self._global_state = global_state
        self._data = data
        self._weight = weight
        partial_qoi = partial(self._qoi,
                              cauchy_fun=model.cauchy,
                              uniaxial_stress_idx=uniaxial_stress_idx,
                              stretch_var_idx=stretch_var_idx)
        super().__init__(partial_qoi)


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
            u: GlobalList, u_prev: GlobalList,
            data_at_step: JaxArray, weight_at_step: JaxArray,
            cauchy_fun: CauchyFn,
            uniaxial_stress_idx: int, stretch_var_idx: int,
    ) -> JaxArray:

        sigma = cauchy_fun(xi, xi_prev, params, u, u_prev)
        uniaxial_sigma = sigma[uniaxial_stress_idx, uniaxial_stress_idx]
        off_axis_strains = jnp.r_[xi[stretch_var_idx][0] - 1.,
            xi[stretch_var_idx][1] - 1.]
        pred = jnp.r_[uniaxial_sigma, off_axis_strains]
        mismatch = (pred - data_at_step) * weight_at_step
        return 0.5 * jnp.sum(mismatch * mismatch)
