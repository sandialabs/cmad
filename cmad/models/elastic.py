import numpy as np
import jax.numpy as jnp

from functools import partial

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.elastic_stress import (isotropic_linear_elastic_cauchy_stress,
                                        two_mu_scale_factor)
from cmad.models.kinematics import gather_F
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.models.var_types import (
    VarType,
    get_num_eqs,
    get_sym_tensor_from_vector,
    get_vector_from_sym_tensor)


class Elastic(Model):
    """
    General elastic model
    """

    def __init__(self, parameters: Parameters,
                 elastic_stress_fun=isotropic_linear_elastic_cauchy_stress,
                 def_type=DefType.FULL_3D):

        self._def_type = def_type
        ndims = def_type_ndims(def_type)
        self._ndims = ndims

        if def_type == DefType.FULL_3D:
            num_residuals = 1

        elif def_type == DefType.PLANE_STRESS \
                or def_type == DefType.UNIAXIAL_STRESS:
            num_residuals = 2

        else:
            raise NotImplementedError

        self._init_residuals(num_residuals)

        # cauchy stress tensor
        self.resid_names[0] = "cauchy"
        self._var_types[0] = VarType.SYM_TENSOR
        self._num_eqs[0] = get_num_eqs(VarType.SYM_TENSOR, 3)
        init_vec_cauchy = np.zeros(self._num_eqs[0])

        self._init_xi = [init_vec_cauchy]

        if def_type == DefType.PLANE_STRESS:
            # out of plane stretch
            self.resid_names[1] = "out of plane stretch"
            self._var_types[1] = VarType.SCALAR
            self._num_eqs[1] = get_num_eqs(VarType.SCALAR, ndims)
            init_oop_stretch = np.ones(self._num_eqs[1])

            self._init_xi += [init_oop_stretch]

        # may want to allow for some idx ([0, 1 ,2]) to be the uniaxial
        # stress idx later
        elif def_type == DefType.UNIAXIAL_STRESS:
            # off-axis stretches
            self.resid_names[1] = "off-axis stretches"
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

        residual = partial(self._residual,
                           def_type=def_type,
                           elastic_stress=elastic_stress_fun)

        cauchy = partial(self.cauchy, def_type=def_type)

        super().__init__(residual, cauchy)

    @staticmethod
    def _residual(xi, xi_prev, params, u, u_prev,
                  def_type, elastic_stress) -> jnp.array:

        # state variables for the model
        cauchy = get_sym_tensor_from_vector(xi[0], 3)

        # global state variables
        F = gather_F(xi, u, def_type, 1)  # 3D deformation gradient
        # T = u[1] # temperature

        # elastic residual
        scale_factor = two_mu_scale_factor(params)
        C_elastic_cauchy_tensor = cauchy - elastic_stress(F, u, params)
        C_elastic_cauchy = \
            get_vector_from_sym_tensor(C_elastic_cauchy_tensor, 3) \
            / scale_factor

        if def_type == def_type.FULL_3D:
            C_elastic = C_elastic_cauchy

        elif def_type == def_type.PLANE_STRESS or \
                def_type == def_type.UNIAXIAL_STRESS:

            if def_type == def_type.PLANE_STRESS:
                C_stretch = cauchy[2, 2] / scale_factor

            elif def_type == def_type.UNIAXIAL_STRESS:
                C_stretch = jnp.r_[cauchy[1, 1], cauchy[2, 2]] \
                    / scale_factor

            C_elastic = jnp.r_[C_elastic_cauchy, C_stretch]

        return C_elastic

    def _check_params(self, parameters):
        raise NotImplementedError

    @staticmethod
    def cauchy(xi, xi_prev, params, u, u_prev,
               def_type) -> jnp.array:

        return get_sym_tensor_from_vector(xi[0], 3)
