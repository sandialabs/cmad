import numpy as np
import jax.numpy as jnp

from abc import ABC, abstractmethod
from jax import hessian, jit, jacfwd, jacrev
from jax.tree_util import tree_flatten

from cmad.models.deriv_types import DerivType


class QoI(ABC):
    def __init__(self, qoi_fun):
        self._qoi = jit(qoi_fun)
        self._dqoi = [jit(jacfwd(qoi_fun, argnums=DerivType.DXI)),
                      jit(jacfwd(qoi_fun, argnums=DerivType.DXI_PREV)),
                      jit(jacrev(qoi_fun, argnums=DerivType.DPARAMS))]
        self._d2qoi = jit(hessian(qoi_fun, argnums=(DerivType.DXI,
                                                    DerivType.DPARAMS)))

        self._hessian_xi_xi = jit(hessian(qoi_fun,
                                          argnums=DerivType.DXI))

        self._hessian_xi_params = jit(jacrev(jacfwd(qoi_fun,
                                           argnums=DerivType.DXI_PREV),
                                           argnums=DerivType.DPARAMS))

        self._hessian_params_params = jit(hessian(qoi_fun,
                                          argnums=DerivType.DPARAMS))

    def evaluate(self, step):
        """
        Evaluate the QoI (J) or its derivatives (dJ) at a step.
        """

        variables = self._model.variables()
        deriv_mode = self._model.deriv_mode()
        data_at_step = self.data_at_step(step)

        if deriv_mode == DerivType.DNONE:
            self._J = np.asarray(self._qoi(*variables, data_at_step),
                                 dtype=self.model().dtype)
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

    # consider jitting
    def unpack_state_hessian(self,
            pytree_hessian):

        num_residuals = self._model.num_residuals
        hessian = np.block([[np.asarray(
            pytree_hessian[row_res_idx][col_res_idx])
            for col_res_idx in range(num_residuals)]
            for row_res_idx in range(num_residuals)]
        )

        return hessian


    # consider jitting
    def unpack_params_hessian(self,
            pytree_hessian,
            first_deriv_type):

        num_param_names = len(self._model.parameters._names)
        active_idx = self._model.parameters.active_idx

        if first_deriv_type == DerivType.DPARAMS:
            offsets = num_param_names * np.arange(num_param_names)
            block_shapes = self._model.parameters.block_shapes
        else:
            num_residuals = self._model.num_residuals
            offsets = num_param_names * np.arange(num_residuals)
            block_shapes = self._model.parameters.mixed_block_shapes

        flat_hessian, _ = tree_flatten(pytree_hessian)
        hessian = np.block([
            [np.asarray(flat_hessian[idx]).reshape(*block_shapes[idx])
            for idx in range(offset, offset + num_param_names)]
            for offset in offsets])[:, active_idx]

        if first_deriv_type == DerivType.DPARAMS:
            hessian = hessian[active_idx, :]

        return hessian

    def evaluate_hessians(self, step):
        """
        Evaluate the Hessians of the QoI
        """

        variables = self._model.variables()
        data_at_step = self.data_at_step(step)
        hessian_params_params = \
            self._hessian_params_params(*variables, data_at_step)
        hessian_xi_params = \
            self._hessian_xi_params(*variables, data_at_step)
        hessian_xi_xi = \
            self._hessian_xi_xi(*variables, data_at_step)

        self.d2J_dparams2 = self.unpack_params_hessian(hessian_params_params,
            DerivType.DPARAMS)
        self.d2J_dxi_dparams = self.unpack_params_hessian(hessian_xi_params,
            DerivType.DXI)
        self.d2J_dxi2 = self.unpack_state_hessian(hessian_xi_xi)

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
