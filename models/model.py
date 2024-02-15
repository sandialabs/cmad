import numpy as np

import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array

from abc import ABC, abstractmethod
from functools import partial

from cmad.models.deriv_types import DerivType
from cmad.models.var_types import VarType


class Model(ABC):
    def __init__(self, residual_fun, cauchy_fun):
        self._residual = jit(residual_fun)
        self._jacobian = [jit(jacfwd(residual_fun, argnums=DerivType.DXI)),
                          jit(jacfwd(residual_fun, argnums=DerivType.DXI_PREV)),
                          jit(jacrev(residual_fun, argnums=DerivType.DPARAMS))]

        self.cauchy = jit(cauchy_fun)
        self.dcauchy = [jit(jacfwd(cauchy_fun, argnums=DerivType.DXI)),
                        jit(jacfwd(cauchy_fun, argnums=DerivType.DXI_PREV)),
                        jit(jacrev(cauchy_fun, argnums=DerivType.DPARAMS))]

        self._deriv_mode = DerivType.DNONE

        self.parameters.model_active_params_jacobian = \
            jit(self.parameters.model_active_params_jacobian,
                static_argnums=1)

    @staticmethod
    def _residual(xi, xi_prev, params, u, u_prev) -> Array:
        """
        The residual function.
        No side effects allowed!
        """
        raise NotImplementedError

    @staticmethod
    def cauchy(xi, xi_prev, params, u, u_prev) -> Array:
        """
        The cauchy stress function.
        No side effects allowed!
        """
        raise NotImplementedError

    def evaluate(self):
        """
        Evaluate the residual (C) or its jacobian (Jac).
        """

        variables = self.variables()
        deriv_mode = self._deriv_mode

        if deriv_mode == DerivType.DNONE:
            self._C = np.asarray(self._residual(*variables), dtype=np.float64)
            self._Jac = None
        elif deriv_mode == DerivType.DPARAMS:
            Jac = self._jacobian[deriv_mode](*variables)
            self._Jac = np.asarray(
                self.parameters.model_active_params_jacobian(
                    Jac, self.num_dofs), dtype=np.float64)
        else:
            self._Jac = np.hstack(self._jacobian[deriv_mode](*variables))

    def evaluate_cauchy(self):
        """
        Evaluate the cauchy stress (Sigma) or its derivatives (dSigma).
        """

        variables = self.variables()
        deriv_mode = self._deriv_mode

        if deriv_mode == DerivType.DNONE:
            self._Sigma = np.asarray(self.cauchy(*variables), dtype=np.float64)
            self._dSigma = None
        elif deriv_mode == DerivType.DPARAMS:
            dSigma = self._jacobian[deriv_mode](*variables)
            self._dSigma = \
                np.asarray(self.parameters.active_params_jacobian(dSigma),
                           dtype=np.float64)
        else:
            self._dSigma = np.dstack(self.dcauchy[deriv_mode](*variables))

    def set_xi_to_init_vals(self):
        for ii in range(self.num_residuals):
            self._xi[ii] = self._init_xi[ii].copy()
            self._xi_prev[ii] = self._init_xi[ii].copy()

    def C(self):
        return self._C

    def Jac(self):
        return self._Jac

    def Sigma(self):
        return self._Sigma

    def dSigma(self):
        return self._dSigma

    def variables(self):
        return self._xi, self._xi_prev, self.parameters.values, \
            self._u, self._u_prev

    def _init_residuals(self, num_residuals: int):
        self.num_residuals = num_residuals
        self._num_eqs = np.zeros(num_residuals, dtype=int)
        self._var_types = np.zeros(num_residuals, dtype=int)
        self.resid_names = [None] * num_residuals

    def _init_state_variables(self):
        """
        Initialize the list for the state variables xi and xi_prev
        and the offsets for derivative array indexing.
        """
        num_residuals = self.num_residuals
        self._xi = [None] * num_residuals
        self._xi_prev = [None] * num_residuals

        self.num_dofs = 0
        self._delta_xi_offsets = np.zeros(num_residuals, dtype=int)
        for ii in range(num_residuals):
            self._delta_xi_offsets[ii] += self.num_dofs
            self.num_dofs += self._num_eqs[ii]

    def delta_xi_offset(self, res_idx: int, eq_idx: int) -> int:
        return self._delta_xi_offsets[res_idx] + eq_idx

    def var_type(self, residual: int) -> int:
        return self._var_types[residual]

    def resid_name(self, residual: int) -> str:
        return self.resid_names[residual]

    def gather_global(self, u, u_prev):
        self._u = u.copy()
        self._u_prev = u_prev.copy()

    def gather_xi(self, xi, xi_prev):
        self._xi = xi.copy()
        self._xi_prev = xi_prev.copy()

    def seed_xi(self):
        self._deriv_mode = DerivType.DXI

    def seed_xi_prev(self):
        self._deriv_mode = DerivType.DXI_PREV

    def seed_params(self):
        self._deriv_mode = DerivType.DPARAMS

    def seed_none(self):
        self._deriv_mode = DerivType.DNONE

    def deriv_mode(self):
        return self._deriv_mode

    def xi(self) -> list:
        return self._xi

    def xi_prev(self) -> list:
        return self._xi_prev

    def advance_xi(self):
        for ii in range(self.num_residuals):
            self._xi_prev[ii] = self._xi[ii].copy()

    def set_scalar_xi(self, idx, xi):
        self._xi[idx] = xi.copy()

    def set_vector_xi(self, idx, xi):
        self._xi[idx] = xi.copy()

    def set_sym_tensor_xi(self, idx, xi):
        if self._ndims == 3:
            self._xi[idx][0] = xi[0, 0]
            self._xi[idx][1] = xi[0, 1]
            self._xi[idx][2] = xi[0, 2]
            self._xi[idx][3] = xi[1, 1]
            self._xi[idx][4] = xi[1, 2]
            self._xi[idx][5] = xi[2, 2]
        if self._ndims == 2:
            self._xi[idx][0] = xi[0, 0]
            self._xi[idx][1] = xi[0, 1]
            self._xi[idx][2] = xi[1, 1]
        if self._ndims == 1:
            self._xi[idx][0] = xi[0, 0]

    def set_tensor_xi(self, idx, xi):
        eq = 0
        for ii in range(self._ndims):
            for jj in range(self._ndims):
                self._xi[idx][eq] = xi[ii, jj]
                eq += 1

    def add_to_xi(self, delta_xi):
        for idx in range(self.num_residuals):
            var_type = self._var_types[idx]
            if var_type == VarType.SCALAR:
                self.add_to_scalar_xi(idx, delta_xi)
            if var_type == VarType.VECTOR:
                self.add_to_vector_xi(idx, delta_xi)
            if var_type == VarType.SYM_TENSOR:
                self.add_to_sym_tensor_xi(idx, delta_xi)
            if var_type == VarType.TENSOR:
                self.add_to_tensor_xi(idx, delta_xi)

    def add_to_scalar_xi(self, idx, delta_xi):
        assert self._var_types[idx] == VarType.SCALAR
        offset = self.delta_xi_offset(idx, 0)
        self._xi[idx] += delta_xi[offset]

    def add_to_vector_xi(self, idx, delta_xi):
        assert self._var_types[idx] == VarType.VECTOR
        for eq in range(self._ndims):
            offset = self.delta_xi_offset(idx, eq)
            self._xi[idx][eq] += delta_xi[offset]

    def add_to_sym_tensor_xi(self, idx, delta_xi):
        assert self._var_types[idx] == VarType.SYM_TENSOR
        if self._ndims == 3:
            for eq in range(6):
                offset = self.delta_xi_offset(idx, eq)
                self._xi[idx][eq] += delta_xi[offset]
        if self._ndims == 2:
            for eq in range(3):
                offset = self.delta_xi_offset(idx, eq)
                self._xi[idx][eq] += delta_xi[offset]
        if self._ndims == 1:
            offset = self.delta_xi_offset(idx, 0)
            self._xi[idx][0] += delta_xi[offset]

    def add_to_tensor_xi(self, idx, delta_xi):
        assert self._var_types[idx] == VarType.TENSOR
        eq = 0
        for ii in range(self._ndims):
            for jj in range(self._ndims):
                offset = self.delta_xi_offsets(idx, eq)
                self._xi[idx][eq] += delta_xi[offset]
                eq += 1

    def set_scalar_xi_prev(self, idx, xi_prev):
        self._xi_prev[idx] = xi_prev.copy()

    def set_vector_xi_prev(self, idx, xi_prev):
        self._xi_prev[idx] = xi_prev.copy()

    def set_sym_tensor_xi_prev(self, idx, xi_prev):
        if self._ndims == 3:
            self._xi_prev[idx][0] = xi_prev[0, 0]
            self._xi_prev[idx][1] = xi_prev[0, 1]
            self._xi_prev[idx][2] = xi_prev[0, 2]
            self._xi_prev[idx][3] = xi_prev[1, 1]
            self._xi_prev[idx][4] = xi_prev[1, 2]
            self._xi_prev[idx][5] = xi_prev[2, 2]
        if self._ndims == 2:
            self._xi_prev[idx][0] = xi_prev[0, 0]
            self._xi_prev[idx][1] = xi_prev[0, 1]
            self._xi_prev[idx][2] = xi_prev[1, 1]
        if self._ndims == 1:
            self._xi_prev[idx][0] = xi_prev[0, 0]

    def set_tensor_xi_prev(self, idx, xi_prev):
        eq = 0
        for ii in range(self._ndims):
            for jj in range(self._ndims):
                self._xi_prev[idx][eq] = xi_prev[ii, jj]
                eq += 1

    def set_scalar_u(self, idx, u):
        self._u[idx] = u.copy()

    def set_vector_u(self, idx, u):
        self._u[idx] = u.copy()

    def set_sym_tensor_u(self, idx, u):
        if self._ndims == 3:
            self.u[idx][0] = u[0, 0]
            self.u[idx][1] = u[0, 1]
            self.u[idx][2] = u[0, 2]
            self.u[idx][3] = u[1, 1]
            self.u[idx][4] = u[1, 2]
            self.u[idx][5] = u[2, 2]
        if self._ndims == 2:
            self.u[idx][0] = u[0, 0]
            self.u[idx][1] = u[0, 1]
            self.u[idx][2] = u[1, 1]
        if self._ndims == 1:
            self.u[idx][0] = u[0, 0]

    def set_tensor_u(self, idx, u):
        eq = 0
        for ii in range(self._ndims):
            for jj in range(self._ndims):
                self._u[idx][eq] = u[ii, jj]
                eq += 1

    def set_scalar_u_prev(self, idx, u_prev):
        self._u_prev[idx] = u_prev.copy()

    def set_vector_u_prev(self, idx, u_prev):
        self._u_prev[idx] = u_prev.copy()

    def set_sym_tensor_u_prev(self, idx, u_prev):
        if self._ndims == 3:
            self.u_prev[idx][0] = u_prev[0, 0]
            self.u_prev[idx][1] = u_prev[0, 1]
            self.u_prev[idx][2] = u_prev[0, 2]
            self.u_prev[idx][3] = u_prev[1, 1]
            self.u_prev[idx][4] = u_prev[1, 2]
            self.u_prev[idx][5] = u_prev[2, 2]
        if self._ndims == 2:
            self.u_prev[idx][0] = u_prev[0, 0]
            self.u_prev[idx][1] = u_prev[0, 1]
            self.u_prev[idx][2] = u_prev[1, 1]
        if self._ndims == 1:
            self.u_prev[idx][0] = u_prev[0, 0]

    def set_tensor_u_prev(self, idx, u_prev):
        eq = 0
        for ii in range(self._ndims):
            for jj in range(self._ndims):
                self._u_prev[idx][eq] = u_prev[ii, jj]
                eq += 1

    @staticmethod
    def store_xi(xi_list, xi_val, step):
        for idx in range(len(xi_list[step])):
            xi_list[step][idx] = xi_val[idx].copy()
