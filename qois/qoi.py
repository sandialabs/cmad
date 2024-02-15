import numpy as np
import jax.numpy as jnp

from abc import ABC, abstractmethod
from jax import jit, jacfwd, jacrev

from cmad.models.deriv_types import DerivType


class QoI(ABC):
    def __init__(self, qoi_fun):
        self._qoi = jit(qoi_fun)
        self._dqoi = [jit(jacfwd(qoi_fun, argnums=DerivType.DXI)),
                      jit(jacfwd(qoi_fun, argnums=DerivType.DXI_PREV)),
                      jit(jacrev(qoi_fun, argnums=DerivType.DPARAMS))]

    def evaluate(self, step):
        """
        Evaluate the QoI (J) or its derivatives (dJ) at a step.
        """

        variables = self._model.variables()
        deriv_mode = self._model.deriv_mode()
        data_at_step = self._data[:, :, step]

        if deriv_mode == DerivType.DNONE:
            self._J = np.asarray(self._qoi(*variables, data_at_step),
                                 dtype=np.float64)
            self._dJ = None
        elif deriv_mode == DerivType.DPARAMS:
            dJ = self._dqoi[deriv_mode](*variables, data_at_step)
            self._dJ = \
                np.asarray(
                    self._model.parameters.qoi_active_params_jacobian(dJ),
                    dtype=np.float64)
        else:
            self._dJ = \
                np.atleast_2d(np.hstack(self._dqoi[deriv_mode](*variables,
                                                               data_at_step)))

    def J(self):
        return self._J

    def dJ(self):
        return self._dJ

    def model(self):
        return self._model

    def global_state(self):
        return self._global_state

    def data(self):
        return self._data

    def weight(self):
        return self._weight
