from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar

import jax.numpy as jnp
import numpy as np

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.elastic_constants import ElasticConstants
from cmad.models.elastic_stress import (
    conventional_elastic_stress_fun,
    isotropic_linear_elastic_cauchy_stress,
    stress_fun_is_finite,
    two_mu_scale_factor,
)
from cmad.models.global_fields import GlobalFieldsAtPoint, StepTime
from cmad.models.kinematics import gather_F
from cmad.models.mechanics_model import MechanicsModel
from cmad.models.var_types import (
    VarType,
    get_num_eqs,
    get_sym_tensor_from_vector,
    get_vector_from_sym_tensor,
)
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray, Scalar, StateList


class Elastic(MechanicsModel):
    """
    General elastic model
    """

    supports_closed_form: ClassVar[bool] = True
    supports_mixed: ClassVar[bool] = True

    _def_type: int
    _ndims: int

    def __init__(
            self, parameters: Parameters,
            elastic_stress_fun: Callable[
                ..., JaxArray] = isotropic_linear_elastic_cauchy_stress,
            def_type: int = DefType.FULL_3D,
            is_complex: bool = False,
    ) -> None:

        self._is_complex = is_complex
        self.dtype = float
        if is_complex:
            self.dtype = complex
        self.is_finite_deformation = stress_fun_is_finite(elastic_stress_fun)

        self._def_type = def_type
        ndims = def_type_ndims(def_type)
        self._ndims = ndims

        if def_type == DefType.FULL_3D or def_type == DefType.PLANE_STRAIN:
            num_residuals = 1

        elif def_type == DefType.PLANE_STRESS \
                or def_type == DefType.UNIAXIAL_STRESS:
            num_residuals = 2

        else:
            raise NotImplementedError

        self._init_residuals(num_residuals)

        # cauchy stress tensor
        self.var_names[0] = "cauchy"
        self._var_types[0] = VarType.SYM_TENSOR
        self._num_eqs[0] = get_num_eqs(VarType.SYM_TENSOR, 3)
        init_vec_cauchy = np.zeros(self._num_eqs[0])

        self._init_xi = [init_vec_cauchy]

        if def_type == DefType.PLANE_STRESS:
            # out of plane stretch
            self.var_names[1] = "out of plane stretch"
            self._var_types[1] = VarType.SCALAR
            self._num_eqs[1] = get_num_eqs(VarType.SCALAR, ndims)
            init_oop_stretch = np.ones(self._num_eqs[1])
            self._oop_stretch_idx = 1

            self._init_xi += [init_oop_stretch]

        # may want to allow for some idx ([0, 1 ,2]) to be the uniaxial
        # stress idx later
        elif def_type == DefType.UNIAXIAL_STRESS:
            # off-axis stretches
            self.var_names[1] = "off-axis stretches"
            self._var_types[1] = VarType.VECTOR
            self._num_eqs[1] = get_num_eqs(VarType.VECTOR, 2)
            init_off_axis_stretches = np.ones(self._num_eqs[1])

            self._init_xi += [init_off_axis_stretches]

        # set the initial values for xi and xi_prev
        self._init_state_variables()
        self.set_xi_to_init_vals()

        # TODO: check that the parameters make sense for this model
        # self._check_params(parameters)
        self.parameters = parameters

        residual = partial(self._residual_fn,
                           def_type=def_type,
                           elastic_stress=elastic_stress_fun)

        cauchy = partial(self._cauchy_fn, def_type=def_type)

        if def_type == DefType.FULL_3D or def_type == DefType.PLANE_STRAIN:
            cauchy_closed_form = partial(self._cauchy_closed_form_fn,
                                         def_type=def_type,
                                         elastic_stress=elastic_stress_fun)
            super().__init__(residual, cauchy,
                             cauchy_closed_form_fun=cauchy_closed_form)
        else:
            super().__init__(residual, cauchy)

    @classmethod
    def from_deck(
            cls,
            model_section: dict[str, Any],
            parameters: Parameters,
            def_type: int,
    ) -> "Elastic":
        elastic_stress = model_section.get("elastic_stress", "isotropic_linear")
        return cls(
            parameters=parameters,
            def_type=def_type,
            elastic_stress_fun=conventional_elastic_stress_fun(elastic_stress),
        )

    def derived_output_field_names(self) -> list[str]:
        return ["cauchy"]

    @staticmethod
    def _residual_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            step_time: StepTime,
            def_type: int, elastic_stress: Callable[..., JaxArray],
    ) -> JaxArray:

        # state variables for the model
        cauchy = get_sym_tensor_from_vector(xi[0], 3)

        # global state variables
        F = gather_F(xi, U, def_type, 1)  # 3D deformation gradient

        # elastic residual
        scale_factor = two_mu_scale_factor(params)
        C_elastic_cauchy_tensor = cauchy - elastic_stress(F, params)
        C_elastic_cauchy = \
            get_vector_from_sym_tensor(C_elastic_cauchy_tensor, 3) \
            / scale_factor

        if def_type == DefType.FULL_3D or def_type == DefType.PLANE_STRAIN:
            C_elastic = C_elastic_cauchy

        elif def_type == DefType.PLANE_STRESS or \
                def_type == DefType.UNIAXIAL_STRESS:

            if def_type == DefType.PLANE_STRESS:
                C_stretch = cauchy[2, 2] / scale_factor

            elif def_type == DefType.UNIAXIAL_STRESS:
                C_stretch = jnp.r_[cauchy[1, 1], cauchy[2, 2]] \
                    / scale_factor

            C_elastic = jnp.r_[C_elastic_cauchy, C_stretch]

        return C_elastic

    def _check_params(self, parameters: Parameters) -> None:
        raise NotImplementedError

    @staticmethod
    def _cauchy_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            def_type: int,
    ) -> JaxArray:

        return get_sym_tensor_from_vector(xi[0], 3)

    @staticmethod
    def _cauchy_closed_form_fn(
            params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            def_type: int,
            elastic_stress: Callable[..., JaxArray],
    ) -> JaxArray:

        grad_u = U.grad_fields["u"]
        if def_type == DefType.PLANE_STRAIN:
            # 2D grad_u embedded with the out of plane stretch fixed to 1
            # (zero out of plane strain). elastic_stress returns the full
            # 3x3 stress; the GR contracts its leading 2x2 block.
            F_2D = jnp.eye(2) + grad_u
            F = jnp.r_[jnp.c_[F_2D, jnp.zeros((2, 1))],
                       jnp.c_[jnp.zeros((1, 2)), 1.0]]
        else:
            F = jnp.eye(3) + grad_u
        return elastic_stress(F, params)

    @staticmethod
    def pressure_scale_factor(params: dict[str, Any]) -> Scalar:
        return ElasticConstants.from_params(params["elastic"]).kappa

    @staticmethod
    def shear_scale_factor(params: dict[str, Any]) -> Scalar:
        return ElasticConstants.from_params(params["elastic"]).mu
