from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import jax.numpy as jnp

from jax import hessian, jit, jacfwd, jacrev
from jax.tree_util import tree_flatten

from numpy.typing import NDArray

from cmad.models.deriv_types import DerivType
from cmad.models.model import Model
from cmad.typing import JaxArray, PyTree, QoIFn, Step


class QoI(ABC):
    # ---- set in QoI.__init__ ----
    _qoi: Callable[..., JaxArray]
    _dqoi: list[Callable[..., PyTree]]
    _d2qoi: Callable[..., PyTree]
    _hessian_xi_xi: Callable[..., PyTree]
    _hessian_xi_params: Callable[..., PyTree]
    _hessian_params_params: Callable[..., PyTree]

    # ---- expected to be set by subclass before super().__init__() ----
    _model: Model
    _global_state: NDArray[np.floating]
    _data: NDArray[np.floating]
    _weight: NDArray[np.floating]

    # ---- populated by evaluate() / evaluate_hessians() ----
    _J: NDArray[np.floating]
    _dJ: NDArray[np.floating] | None
    d2J_dxi2: NDArray[np.floating]
    d2J_dparams2: NDArray[np.floating]
    d2J_dxi_dparams: NDArray[np.floating]

    def __init__(self, qoi_fun: QoIFn) -> None:
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

    def evaluate(self, step: Step) -> None:
        """
        Evaluate the QoI (J) or its derivatives (dJ) at a step.
        """

        variables = self._model.variables()
        deriv_mode = self._model.deriv_mode()
        data_at_step = self.data_at_step(step)
        weight_at_step = self.weight_at_step(step)

        if deriv_mode == DerivType.DNONE:
            self._J = np.asarray(self._qoi(*variables,
                                 data_at_step,
                                 weight_at_step),
                                 dtype=self.model().dtype)
            self._dJ = None
        elif deriv_mode == DerivType.DPARAMS:
            dJ = self._dqoi[deriv_mode](*variables,
                                        data_at_step,
                                        weight_at_step)
            self._dJ = \
                np.asarray(
                    self._model.parameters.qoi_active_params_jacobian(dJ),
                    dtype=np.float64)
        else:
            self._dJ = \
                np.atleast_2d(np.hstack(self._dqoi[deriv_mode](*variables,
                                                               data_at_step,
                                                               weight_at_step)))

    # consider jitting
    def unpack_state_hessian(
            self, pytree_hessian: PyTree,
    ) -> NDArray[np.floating]:

        num_residuals = self._model.num_residuals
        hessian = np.block([[np.asarray(
            pytree_hessian[row_res_idx][col_res_idx])
            for col_res_idx in range(num_residuals)]
            for row_res_idx in range(num_residuals)]
        )

        return hessian


    # consider jitting
    def unpack_params_hessian(
            self, pytree_hessian: PyTree, first_deriv_type: int,
    ) -> NDArray[np.floating]:

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

    def evaluate_hessians(self, step: Step) -> None:
        """
        Evaluate the Hessians of the QoI
        """

        variables = self._model.variables()
        data_at_step = self.data_at_step(step)
        weight_at_step = self.weight_at_step(step)
        hessian_params_params = \
            self._hessian_params_params(*variables,
                                        data_at_step,
                                        weight_at_step)
        hessian_xi_params = \
            self._hessian_xi_params(*variables,
                                    data_at_step,
                                    weight_at_step)
        hessian_xi_xi = \
            self._hessian_xi_xi(*variables,
                                data_at_step,
                                weight_at_step)

        self.d2J_dparams2 = self.unpack_params_hessian(hessian_params_params,
            DerivType.DPARAMS)
        self.d2J_dxi_dparams = self.unpack_params_hessian(hessian_xi_params,
            DerivType.DXI)
        self.d2J_dxi2 = self.unpack_state_hessian(hessian_xi_xi)

    def J(self) -> NDArray[np.floating]:
        return self._J

    def dJ(self) -> NDArray[np.floating] | None:
        return self._dJ

    def model(self) -> Model:
        return self._model

    def global_state(self) -> NDArray[np.floating]:
        return self._global_state

    def data(self) -> NDArray[np.floating]:
        return self._data

    def weight(self) -> NDArray[np.floating]:
        return self._weight

    def data_at_step(self, step: Step) -> NDArray[np.floating]:
        """
        Return the measured data at a given step.
        Subclasses must override.
        """
        raise NotImplementedError

    def weight_at_step(self, step: Step) -> NDArray[np.floating]:
        """
        Return the measurement weight at a given step.
        Subclasses must override.
        """
        raise NotImplementedError
