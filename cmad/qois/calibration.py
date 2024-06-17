import numpy as np
from functools import partial

import jax.numpy as jnp
from jax import Array

from cmad.qois.qoi import QoI

from cmad.models.var_types import get_sym_tensor_from_vector


class Calibration(QoI):
    def __init__(self, model, global_state, data, weight):
        self._model = model
        assert global_state.shape[-1] == data.shape[-1]
        self._global_state = global_state
        self._data = data
        partial_qoi = partial(self._qoi, cauchy_fun=model.cauchy,
                              weight=jnp.asarray(weight, dtype=np.float64))
        super().__init__(partial_qoi)


    def data_at_step(self, step):
        return self._data[..., step]

    # _qoi not an abstract class, as signatures may be class-specific
    @staticmethod
    def _qoi(xi, xi_prev, params, u, u_prev,
             data_at_step, cauchy_fun, weight) -> Array:

        cauchy = cauchy_fun(xi, xi_prev, params, u, u_prev)
        mismatch = weight @ (cauchy - data_at_step)
        return 0.5 * jnp.sum(mismatch * mismatch)

class UniaxialCalibration(Calibration):
    # _qoi not an abstract class, as signatures may be class-specific
    @staticmethod
    def _qoi(xi, xi_prev, params, u, u_prev,
             data_at_step, cauchy_fun, weight) -> Array:

        sig_11 = cauchy_fun(xi, xi_prev, params, u, u_prev)[1,1]
        pred = jnp.stack([sig_11,xi[2][0]-1.,xi[2][1]-1.],axis=0)
        mismatch = (pred - data_at_step)*weight
        return 0.5 * jnp.sum(mismatch * mismatch)

