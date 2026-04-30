from abc import ABC
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from jax import hessian, jacfwd, jacrev, jit
from jax.tree_util import tree_flatten
from numpy.typing import NDArray

from cmad.models.deriv_types import DerivType
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.models.model import Model
from cmad.typing import JaxArray, Params, PyTree, QoIFn, StateList, Step


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
    _data: NDArray[np.floating]
    _weight: NDArray[np.floating]

    # ---- populated by evaluate() / evaluate_hessians() ----
    _J: NDArray[np.number]
    _dJ: NDArray[np.floating] | None
    d2J_dxi2: NDArray[np.floating]
    d2J_dparams2: NDArray[np.floating]
    d2J_dxi_dparams: NDArray[np.floating]

    def __init__(self, qoi_fun: QoIFn) -> None:
        self._qoi = jit(qoi_fun)
        self._dqoi = [jit(jacfwd(qoi_fun, argnums=DerivType.DXI)),
                      jit(jacfwd(qoi_fun, argnums=DerivType.DXI_PREV)),
                      jit(jacrev(qoi_fun, argnums=DerivType.DPARAMS)),
                      jit(jacfwd(qoi_fun, argnums=DerivType.DU)),
                      jit(jacfwd(qoi_fun, argnums=DerivType.DU_PREV))]
        self._d2qoi = jit(hessian(qoi_fun, argnums=(DerivType.DXI,
                                                    DerivType.DPARAMS)))

        self._hessian_xi_xi = jit(hessian(qoi_fun,
                                          argnums=DerivType.DXI))

        self._hessian_xi_params = jit(jacrev(jacfwd(qoi_fun,
                                           argnums=DerivType.DXI_PREV),
                                           argnums=DerivType.DPARAMS))

        self._hessian_params_params = jit(hessian(qoi_fun,
                                          argnums=DerivType.DPARAMS))

    @classmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            model: Model,
            data: NDArray[np.floating],
            weight: NDArray[np.floating],
    ) -> "QoI":
        """Build a QoI instance from a parsed deck's ``qoi:`` section.

        The driver resolves ``data_file`` / ``weight`` / ``weight_file``
        relative to the deck and passes the loaded arrays in; the
        classmethod translates any subclass-specific extras from
        ``qoi_section`` (e.g. ``uniaxial_stress_idx``) into constructor
        kwargs. The base stub exists so the registry's ``type[QoI]``
        return is statically callable via ``cls.from_deck(...)``;
        subclasses must override.
        """
        raise NotImplementedError

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
            dJ_pytree = cast(
                list[JaxArray],
                self._dqoi[deriv_mode](*variables, data_at_step,
                                       weight_at_step),
            )
            self._dJ = np.atleast_2d(np.hstack(dJ_pytree))

    # consider jitting
    def unpack_state_hessian(
            self, pytree_hessian: PyTree,
    ) -> NDArray[np.floating]:

        # JAX hessian-of-residual returns nested list/list/array
        ph = cast(list[list[JaxArray]], pytree_hessian)
        num_residuals = self._model.num_residuals
        hessian = np.block([[np.asarray(ph[row_res_idx][col_res_idx])
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

    def J(self) -> NDArray[np.number]:
        return self._J

    def dJ(self) -> NDArray[np.floating]:
        assert self._dJ is not None, \
            "dJ() requires a non-DNONE deriv mode (seed_xi/xi_prev/params)"
        return self._dJ

    def model(self) -> Model:
        return self._model

    def data(self) -> NDArray[np.floating]:
        return self._data

    def weight(self) -> NDArray[np.floating]:
        return self._weight

    def dJ_dxi(
            self,
            xi: StateList,
            xi_prev: StateList,
            params: Params,
            U: GlobalFieldsAtPoint,
            U_prev: GlobalFieldsAtPoint,
            data: JaxArray,
            weight: JaxArray,
    ) -> PyTree:
        return self._dqoi[DerivType.DXI](
            xi, xi_prev, params, U, U_prev, data, weight,
        )

    def dJ_dxi_prev(
            self,
            xi: StateList,
            xi_prev: StateList,
            params: Params,
            U: GlobalFieldsAtPoint,
            U_prev: GlobalFieldsAtPoint,
            data: JaxArray,
            weight: JaxArray,
    ) -> PyTree:
        return self._dqoi[DerivType.DXI_PREV](
            xi, xi_prev, params, U, U_prev, data, weight,
        )

    def dJ_dp(
            self,
            xi: StateList,
            xi_prev: StateList,
            params: Params,
            U: GlobalFieldsAtPoint,
            U_prev: GlobalFieldsAtPoint,
            data: JaxArray,
            weight: JaxArray,
    ) -> PyTree:
        return self._dqoi[DerivType.DPARAMS](
            xi, xi_prev, params, U, U_prev, data, weight,
        )

    def dJ_dU(
            self,
            xi: StateList,
            xi_prev: StateList,
            params: Params,
            U: GlobalFieldsAtPoint,
            U_prev: GlobalFieldsAtPoint,
            data: JaxArray,
            weight: JaxArray,
    ) -> PyTree:
        return self._dqoi[DerivType.DU](
            xi, xi_prev, params, U, U_prev, data, weight,
        )

    def dJ_dU_prev(
            self,
            xi: StateList,
            xi_prev: StateList,
            params: Params,
            U: GlobalFieldsAtPoint,
            U_prev: GlobalFieldsAtPoint,
            data: JaxArray,
            weight: JaxArray,
    ) -> PyTree:
        return self._dqoi[DerivType.DU_PREV](
            xi, xi_prev, params, U, U_prev, data, weight,
        )

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
