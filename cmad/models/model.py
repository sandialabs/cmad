from abc import ABC
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, cast

import numpy as np
from jax import hessian, jacfwd, jacrev, jit
from jax.tree_util import tree_flatten
from numpy.typing import NDArray

from cmad.models.deriv_types import DerivType
from cmad.models.var_types import VarType
from cmad.parameters.parameters import Parameters
from cmad.typing import (
    CauchyFn,
    GlobalField,
    GlobalList,
    JaxArray,
    Params,
    PyTree,
    ResidualFn,
    StateBlock,
    StateList,
)


class Model(ABC):
    """Material-point constitutive model contract.

    Subclasses set up a residual function and a Cauchy stress function
    (both pure, no side effects) and pass them to super().__init__().
    See ResidualFn and CauchyFn in cmad.typing for the required callable
    signatures.

    Both functions take (xi, xi_prev, params, u, u_prev) and return a
    JaxArray. They are jit-compiled and AD-derivative-cached at
    construction; access via self._residual / self.cauchy.
    """

    # ---- attributes the subclass must set before super().__init__() ----
    parameters: Parameters
    dtype: type
    _is_complex: bool
    _num_eqs: NDArray[np.intp]
    _var_types: NDArray[np.intp]
    _init_xi: StateList
    _ndims: int

    # ---- attributes set by self._init_residuals() ----
    num_residuals: int
    resid_names: list[str | None]

    # ---- attributes set by self._init_state_variables() ----
    _xi: StateList
    _xi_prev: StateList
    num_dofs: int
    _delta_xi_offsets: NDArray[np.intp]

    # ---- attributes set by self.gather_global() / self.gather_xi() ----
    _u: GlobalList
    _u_prev: GlobalList

    # ---- attributes set by Model.__init__() ----
    # _residual and cauchy are documented in the class docstring; their
    # types come from ResidualFn / CauchyFn in cmad.typing.
    _jacobian: list[Callable[..., PyTree]]
    _hessian_states: Callable[..., PyTree]
    _hessian_xi_params: Callable[..., PyTree]
    _hessian_xi_prev_params: Callable[..., PyTree]
    _hessian_params_params: Callable[..., PyTree]
    dcauchy: list[Callable[..., PyTree]]
    _deriv_mode: int  # backed by DerivType

    # ---- attributes populated by evaluate* methods ----
    _C: NDArray[np.floating]
    _Jac: NDArray[np.floating] | None
    _Sigma: NDArray[np.floating]
    _dSigma: NDArray[np.floating] | None
    d2C_dxi2: NDArray[np.floating]
    d2C_dxi_dxi_prev: NDArray[np.floating]
    d2C_dxi_prev2: NDArray[np.floating]
    d2C_dparams2: NDArray[np.floating]
    d2C_dxi_dparams: NDArray[np.floating]
    d2C_dxi_prev_dparams: NDArray[np.floating]

    @classmethod
    def from_deck(
            cls,
            model_section: dict[str, Any],
            parameters: Parameters,
    ) -> "Model":
        """Build a :class:`Model` instance from the deck's ``model:`` section.

        Concrete subclasses translate deck fields (``def_type``,
        ``effective_stress``, etc.) into constructor kwargs. The base stub
        exists so the registry's ``type[Model]`` return is statically
        callable via ``cls.from_deck(...)``; subclasses must override.
        """
        raise NotImplementedError

    def __init__(
            self, residual_fun: ResidualFn, cauchy_fun: CauchyFn,
    ) -> None:
        self._residual = jit(residual_fun)
        self._jacobian = [jit(jacfwd(residual_fun, argnums=DerivType.DXI,
                          holomorphic=self._is_complex)),
                          jit(jacfwd(residual_fun, argnums=DerivType.DXI_PREV)),
                          jit(jacrev(residual_fun, argnums=DerivType.DPARAMS))]


        self._hessian_states = jit(hessian(residual_fun,
                                   argnums=(DerivType.DXI,
                                            DerivType.DXI_PREV)))

        # not sure if jacfwd or jacrev is preferred
        self._hessian_xi_params = jit(jacrev(jacfwd(residual_fun,
                                      argnums=DerivType.DXI),
                                      argnums=DerivType.DPARAMS))

        self._hessian_xi_prev_params = jit(jacrev(jacfwd(residual_fun,
                                           argnums=DerivType.DXI_PREV),
                                           argnums=DerivType.DPARAMS))

        self._hessian_params_params = jit(hessian(residual_fun,
                                          argnums=DerivType.DPARAMS))

        self.cauchy = jit(cauchy_fun)
        self.dcauchy = [jit(jacfwd(cauchy_fun, argnums=DerivType.DXI)),
                        jit(jacfwd(cauchy_fun, argnums=DerivType.DXI_PREV)),
                        jit(jacrev(cauchy_fun, argnums=DerivType.DPARAMS))]

        self._deriv_mode = DerivType.DNONE

        self.parameters.model_active_params_jacobian = \
            jit(self.parameters.model_active_params_jacobian,
                static_argnums=1)

        self.parameters.compute_mixed_block_shapes(self._num_eqs)

    def evaluate(self) -> None:
        """
        Evaluate the residual (C) or its jacobian (Jac).
        """

        variables = self.variables()
        deriv_mode = self._deriv_mode

        if deriv_mode == DerivType.DNONE:
            self._C = np.asarray(self._residual(*variables), dtype=self.dtype)
            self._Jac = None
        elif deriv_mode == DerivType.DPARAMS:
            Jac = self._jacobian[deriv_mode](*variables)
            self._Jac = np.asarray(
                self.parameters.model_active_params_jacobian(
                    Jac, self.num_dofs), dtype=np.float64)
        else:
            jac_pytree = cast(
                list[JaxArray],
                self._jacobian[deriv_mode](*variables),
            )
            self._Jac = np.hstack(jac_pytree)


    # consider jitting
    def unpack_state_hessian(
            self,
            pytree_hessian: PyTree,
            first_deriv_type: int,
            second_deriv_type: int,
    ) -> NDArray[np.floating]:

        # JAX block-Hessian over multiple argnums returns nested
        # tuple/list/list/array-of-array structure
        ph = cast(list[list[list[list[JaxArray]]]], pytree_hessian)
        num_residuals = self.num_residuals
        hessian = np.block([[np.asarray(
            ph[first_deriv_type][row_res_idx]
              [second_deriv_type][col_res_idx])
            for col_res_idx in range(num_residuals)]
            for row_res_idx in range(num_residuals)]
        )

        return hessian


    # consider jitting
    def unpack_params_hessian(
            self,
            pytree_hessian: PyTree,
            first_deriv_type: int,
    ) -> NDArray[np.floating]:

        num_dofs = self.num_dofs
        num_param_names = len(self.parameters._names)
        active_idx = self.parameters.active_idx

        if first_deriv_type == DerivType.DPARAMS:
            offsets = num_param_names * np.arange(num_param_names)
            block_shapes = self.parameters.block_shapes
        else:
            num_residuals = self.num_residuals
            offsets = num_param_names * np.arange(num_residuals)
            block_shapes = self.parameters.mixed_block_shapes

        flat_hessian, _ = tree_flatten(pytree_hessian)
        hessian = np.block([
            [np.asarray(flat_hessian[idx]).reshape(num_dofs, *block_shapes[idx])
            for idx in range(offset, offset + num_param_names)]
            for offset in offsets])[:, :, active_idx]

        if first_deriv_type == DerivType.DPARAMS:
            return hessian[:, active_idx, :]
        else:
            return hessian


    def evaluate_hessians(self) -> None:
        """
        Evaluate the Hessians of the residual
        """

        variables = self.variables()

        hessian_states = self._hessian_states(*variables)
        hessian_params_params = self._hessian_params_params(*variables)
        hessian_xi_params = self._hessian_xi_params(*variables)
        hessian_xi_prev_params = self._hessian_xi_prev_params(*variables)

        self.d2C_dxi2 = self.unpack_state_hessian(hessian_states,
            DerivType.DXI, DerivType.DXI)
        self.d2C_dxi_dxi_prev = self.unpack_state_hessian(hessian_states,
            DerivType.DXI, DerivType.DXI_PREV)
        self.d2C_dxi_prev2 = self.unpack_state_hessian(hessian_states,
            DerivType.DXI_PREV, DerivType.DXI_PREV)

        self.d2C_dparams2 = self.unpack_params_hessian(hessian_params_params,
            DerivType.DPARAMS)
        self.d2C_dxi_dparams = self.unpack_params_hessian(hessian_xi_params,
            DerivType.DXI)
        self.d2C_dxi_prev_dparams = \
            self.unpack_params_hessian(hessian_xi_prev_params,
            DerivType.DXI_PREV)


    def evaluate_cauchy(self) -> None:
        """
        Evaluate the cauchy stress (Sigma) or its derivatives (dSigma).
        """

        variables = self.variables()
        deriv_mode = self._deriv_mode

        if deriv_mode == DerivType.DNONE:
            self._Sigma = np.asarray(self.cauchy(*variables), dtype=np.float64)
            self._dSigma = None
        elif deriv_mode == DerivType.DPARAMS:
            dSigma = self.dcauchy[deriv_mode](*variables)
            self._dSigma = np.asarray(
                self.parameters.model_active_params_jacobian(dSigma, 9),
                dtype=np.float64)
        else:
            dsigma_pytree = cast(
                list[JaxArray], self.dcauchy[deriv_mode](*variables),
            )
            self._dSigma = np.dstack(dsigma_pytree)

    def set_xi_to_init_vals(self) -> None:
        for ii in range(self.num_residuals):
            self._xi[ii] = self._init_xi[ii].copy().astype(self.dtype)
            self._xi_prev[ii] = self._init_xi[ii].copy().astype(self.dtype)

    def C(self) -> NDArray[np.floating]:
        return self._C

    def Jac(self) -> NDArray[np.floating]:
        assert self._Jac is not None, \
            "Jac() requires a non-DNONE deriv mode (seed_xi/xi_prev/params)"
        return self._Jac

    def Sigma(self) -> NDArray[np.floating]:
        return self._Sigma

    def dSigma(self) -> NDArray[np.floating]:
        assert self._dSigma is not None, \
            "dSigma() requires a non-DNONE deriv mode (seed_xi/xi_prev/params)"
        return self._dSigma

    def variables(
            self,
    ) -> tuple[StateList, StateList, Params, GlobalList, GlobalList]:
        return self._xi, self._xi_prev, self.parameters.values, \
            self._u, self._u_prev

    def _init_residuals(self, num_residuals: int) -> None:
        self.num_residuals = num_residuals
        self._num_eqs = np.zeros(num_residuals, dtype=int)
        self._var_types = np.zeros(num_residuals, dtype=int)
        self.resid_names = [None] * num_residuals

    def _init_state_variables(self) -> None:
        """
        Initialize the list for the state variables xi and xi_prev
        and the offsets for derivative array indexing.
        """
        num_residuals = self.num_residuals
        # slots are populated by subclass setup or set_xi_to_init_vals
        self._xi = cast(StateList, [None] * num_residuals)
        self._xi_prev = cast(StateList, [None] * num_residuals)

        self.num_dofs = 0
        self._delta_xi_offsets = np.zeros(num_residuals, dtype=int)
        for ii in range(num_residuals):
            self._delta_xi_offsets[ii] += self.num_dofs
            self.num_dofs += self._num_eqs[ii]

    def delta_xi_offset(self, res_idx: int, eq_idx: int) -> int:
        return self._delta_xi_offsets[res_idx] + eq_idx

    def var_type(self, residual: int) -> int:
        return self._var_types[residual]

    def resid_name(self, residual: int) -> str | None:
        return self.resid_names[residual]

    @property
    def ndims(self) -> int:
        return self._ndims

    def gather_global(
            self, u: Sequence[GlobalField], u_prev: Sequence[GlobalField],
    ) -> None:
        self._u = list(u)
        self._u_prev = list(u_prev)

    def gather_xi(
            self, xi: Sequence[StateBlock], xi_prev: Sequence[StateBlock],
    ) -> None:
        self._xi = list(xi)
        self._xi_prev = list(xi_prev)

    def seed_xi(self) -> None:
        self._deriv_mode = DerivType.DXI

    def seed_xi_prev(self) -> None:
        self._deriv_mode = DerivType.DXI_PREV

    def seed_params(self) -> None:
        self._deriv_mode = DerivType.DPARAMS

    def seed_none(self) -> None:
        self._deriv_mode = DerivType.DNONE

    def deriv_mode(self) -> int:
        return self._deriv_mode

    def xi(self) -> StateList:
        return self._xi

    def xi_prev(self) -> StateList:
        return self._xi_prev

    def advance_xi(self) -> None:
        for ii in range(self.num_residuals):
            self._xi_prev[ii] = self._xi[ii].copy()

    def set_scalar_xi(self, idx: int, xi: JaxArray) -> None:
        self._xi[idx] = xi.copy()

    def set_vector_xi(self, idx: int, xi: JaxArray) -> None:
        self._xi[idx] = xi.copy()

    def set_sym_tensor_xi(self, idx: int, xi: JaxArray) -> None:
        if self._num_eqs[idx] == 6:
            self._xi[idx][0] = xi[0, 0]
            self._xi[idx][1] = xi[0, 1]
            self._xi[idx][2] = xi[0, 2]
            self._xi[idx][3] = xi[1, 1]
            self._xi[idx][4] = xi[1, 2]
            self._xi[idx][5] = xi[2, 2]
        elif self._num_eqs[idx] == 3:
            self._xi[idx][0] = xi[0, 0]
            self._xi[idx][1] = xi[0, 1]
            self._xi[idx][2] = xi[1, 1]
        elif self._num_eqs[idx] == 1:
            self._xi[idx][0] = xi[0, 0]

    _NDIM_BY_NUM_EQS: ClassVar[dict[int, int]] = {9: 3, 4: 2, 1: 1}

    @staticmethod
    def get_tensor_ndim(num_eqs: int) -> int:
        try:
            return Model._NDIM_BY_NUM_EQS[num_eqs]
        except KeyError as e:
            raise ValueError(
                f"Unknown num_eqs for tensor variable: {num_eqs}"
            ) from e

    def set_tensor_xi(self, idx: int, xi: JaxArray) -> None:
        ndim = Model.get_tensor_ndim(self._num_eqs[idx])
        eq = 0
        for ii in range(ndim):
            for jj in range(ndim):
                self._xi[idx][eq] = xi[ii, jj]
                eq += 1

    def add_to_xi(self, delta_xi: NDArray[np.floating]) -> None:
        for idx in range(self.num_residuals):
            var_type = self._var_types[idx]
            if var_type != VarType.SCALAR:
                for eq in range(self._num_eqs[idx]):
                    offset = self.delta_xi_offset(idx, eq)
                    self._xi[idx][eq] += delta_xi[offset]
            else:
                offset = self.delta_xi_offset(idx, 0)
                self._xi[idx] += delta_xi[offset]

    def set_scalar_xi_prev(self, idx: int, xi_prev: JaxArray) -> None:
        self._xi_prev[idx] = xi_prev.copy()

    def set_vector_xi_prev(self, idx: int, xi_prev: JaxArray) -> None:
        self._xi_prev[idx] = xi_prev.copy()

    def set_sym_tensor_xi_prev(self, idx: int, xi_prev: JaxArray) -> None:
        if self._num_eqs[idx] == 6:
            self._xi_prev[idx][0] = xi_prev[0, 0]
            self._xi_prev[idx][1] = xi_prev[0, 1]
            self._xi_prev[idx][2] = xi_prev[0, 2]
            self._xi_prev[idx][3] = xi_prev[1, 1]
            self._xi_prev[idx][4] = xi_prev[1, 2]
            self._xi_prev[idx][5] = xi_prev[2, 2]
        elif self._num_eqs[idx] == 3:
            self._xi_prev[idx][0] = xi_prev[0, 0]
            self._xi_prev[idx][1] = xi_prev[0, 1]
            self._xi_prev[idx][2] = xi_prev[1, 1]
        elif self._num_eqs[idx] == 1:
            self._xi_prev[idx][0] = xi_prev[0, 0]

    def set_tensor_xi_prev(self, idx: int, xi_prev: JaxArray) -> None:
        ndim = Model.get_tensor_ndim(self._num_eqs[idx])
        eq = 0
        for ii in range(ndim):
            for jj in range(ndim):
                self._xi_prev[idx][eq] = xi_prev[ii, jj]
                eq += 1

    @staticmethod
    def store_xi(
            xi_list: list[StateList],
            xi_val: StateList,
            step: int,
    ) -> None:
        for idx in range(len(xi_list[step])):
            xi_list[step][idx] = xi_val[idx].copy()
