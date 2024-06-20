import numpy as np
import jax.numpy as jnp

from jax import grad

from functools import partial

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.elastic_stress import (isotropic_linear_elastic_stress,
                                        two_mu_scale_factor)
from cmad.models.effective_stress import effective_stress_fun
from cmad.models.hardening import combined_hardening_fun, get_hardening_funs
from cmad.models.kinematics import gather_F, off_axis_idx
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.models.paths import cond_residual
from cmad.models.var_types import (
    VarType,
    get_num_eqs,
    get_scalar,
    get_sym_tensor_from_vector,
    get_vector_from_sym_tensor,
    get_tensor_from_vector,
    get_vector
)


def compute_delta_strain(xi, xi_prev, params, u, u_prev, def_type,
        uniaxial_stress_idx):

    local_var_idx = 2
    F = gather_F(xi, u, def_type, local_var_idx, uniaxial_stress_idx)
    F_prev = gather_F(xi_prev, u_prev, def_type, local_var_idx,
        uniaxial_stress_idx)

    I = jnp.eye(3)
    grad_u = F - I
    grad_u_prev = F_prev - I

    epsilon = 0.5 * (grad_u + grad_u.T)
    epsilon_prev = 0.5 * (grad_u_prev + grad_u_prev.T)
    delta_epsilon = epsilon - epsilon_prev

    # Q is a rotation from material coordinates to global coordinates
    # Q_{ij} = e_i (global) \dot e_j (material)
    Q = params["rotation matrix"]

    if def_type == DefType.UNIAXIAL_STRESS:
        off_axis_delta_strain = get_vector(xi[3], 3)
        constrained_delta_epsilon = jnp.array([
            [delta_epsilon[0, 0],
            off_axis_delta_strain[0],
            off_axis_delta_strain[1]],
            [off_axis_delta_strain[0],
            delta_epsilon[1, 1],
            off_axis_delta_strain[2]],
            [off_axis_delta_strain[1],
            off_axis_delta_strain[2],
            delta_epsilon[2, 2]]
        ])
        material_delta_epsilon = Q.T @ constrained_delta_epsilon @ Q
    else:
        material_delta_epsilon = Q.T @ delta_epsilon @ Q


    return material_delta_epsilon


def compute_yield_fun_and_normal(xi, params, def_type,
                                 effective_stress, hardening, is_complex):

    ndims = def_type_ndims(def_type)

    plastic_params = params["plastic"]
    Y = plastic_params["flow stress"]["initial yield"]["Y"]
    hardening_params = plastic_params["flow stress"]["hardening"]

    cauchy = get_sym_tensor_from_vector(xi[0], 3)
    phi = effective_stress(cauchy, plastic_params)

    alpha = get_scalar(xi[1])
    sigma_flow = Y + hardening(alpha, hardening_params)

    yield_fun = (phi - sigma_flow) / two_mu_scale_factor(params)
    yield_normal = grad(effective_stress, holomorphic=is_complex)(cauchy, plastic_params)

    return yield_fun, yield_normal


class SmallRateElasticPlastic(Model):
    """
    Small strain rate form elastic-plastic model:
    Elastic: Modular linear elasticity
    Plastic: Modular effective stress and hardening
    """

    def __init__(self, parameters: Parameters,
                 def_type=DefType.FULL_3D,
                 elastic_stress_fun=isotropic_linear_elastic_stress,
                 hardening_funs: dict = get_hardening_funs(),
                 yield_tol=1e-14, uniaxial_stress_idx=0, is_complex=False):

        self._is_complex = is_complex
        self.dtype = float
        if is_complex:
            self.dtype = complex

        self._def_type = def_type
        ndims = def_type_ndims(def_type)
        self._ndims = ndims

        if def_type == DefType.FULL_3D:
            num_residuals = 2

        elif def_type == DefType.PLANE_STRESS:
            num_residuals = 3

        elif def_type == DefType.UNIAXIAL_STRESS:
            num_residuals = 4

        else:
            raise NotImplementedError

        self._init_residuals(num_residuals)

        # cauchy stress tensor
        self.resid_names[0] = "cauchy"
        self._var_types[0] = VarType.SYM_TENSOR
        self._num_eqs[0] = get_num_eqs(VarType.SYM_TENSOR, 3)
        init_vec_cauchy = np.zeros(self._num_eqs[0])

        # isotropic hardening variable
        self.resid_names[1] = "alpha"
        self._var_types[1] = VarType.SCALAR
        self._num_eqs[1] = get_num_eqs(VarType.SCALAR, 3)
        init_alpha = np.zeros(self._num_eqs[1])

        self._init_xi = [init_vec_cauchy, init_alpha]

        if def_type == DefType.PLANE_STRESS:
            # out of plane stretch
            self.resid_names[2] = "out of plane stretch"
            self._var_types[2] = VarType.SCALAR
            self._num_eqs[2] = get_num_eqs(VarType.SCALAR, ndims)
            init_oop_stretch = np.ones(self._num_eqs[2])

            self._init_xi += [init_oop_stretch]

        elif def_type == DefType.UNIAXIAL_STRESS:
            # off-axis stretches
            self.resid_names[2] = "off-axis stretches"
            self._var_types[2] = VarType.VECTOR
            self._num_eqs[2] = get_num_eqs(VarType.VECTOR, 2)
            init_off_axis_stretches = np.ones(self._num_eqs[2])

            # off-axis delta strains
            self.resid_names[3] = "off-axis delta strains"
            self._var_types[3] = VarType.VECTOR
            self._num_eqs[3] = get_num_eqs(VarType.VECTOR, 3)
            init_off_axis_delta_strains = np.zeros(self._num_eqs[3])

            self._init_xi += [init_off_axis_stretches] \
                           + [init_off_axis_delta_strains]

        # set the initial values for xi and xi_prev
        self._init_state_variables()
        self.set_xi_to_init_vals()

        # TODO: check that the parameters make sense for this model
        # self._check_params(parameters)
        self.parameters = parameters

        effective_stress_type = \
            list(parameters.values["plastic"]["effective stress"])[0]

        residual = partial(self._residual, def_type=def_type,
                           elastic_stress=elastic_stress_fun,
                           effective_stress=effective_stress_fun(
                               effective_stress_type),
                           hardening=partial(combined_hardening_fun,
                                             hardening_funs=hardening_funs),
                           yield_tol=yield_tol,
                           uniaxial_stress_idx=uniaxial_stress_idx, is_complex=is_complex)

        cauchy = partial(self.cauchy, def_type=def_type)

        super().__init__(residual, cauchy)

    @staticmethod
    def _residual(xi, xi_prev, params, u, u_prev,
                  def_type, elastic_stress, effective_stress, hardening,
                  yield_tol, uniaxial_stress_idx, is_complex) -> jnp.array:

        # state variables for the model
        cauchy = get_sym_tensor_from_vector(xi[0], 3)
        cauchy_prev = get_sym_tensor_from_vector(xi_prev[0], 3)
        alpha = get_scalar(xi[1])
        alpha_prev = get_scalar(xi_prev[1])

        trial_delta_strain \
            = compute_delta_strain(xi, xi_prev, params, u, u_prev, def_type,
                                   uniaxial_stress_idx)
        trial_delta_cauchy \
            = elastic_stress(trial_delta_strain, params)
        delta_gamma = alpha - alpha_prev
        scale_factor = two_mu_scale_factor(params)

        # elastic residual
        C_elastic_cauchy_tensor = cauchy - cauchy_prev \
            - trial_delta_cauchy
        C_elastic_cauchy = \
            get_vector_from_sym_tensor(C_elastic_cauchy_tensor, 3) \
            / scale_factor
        C_elastic_alpha = delta_gamma

        # plastic residual
        yield_fun, yield_normal = \
            compute_yield_fun_and_normal(xi, params, def_type,
                                         effective_stress, hardening, is_complex)
        delta_plastic_strain = delta_gamma * yield_normal
        delta_cauchy = trial_delta_cauchy \
            - elastic_stress(delta_plastic_strain, params)
        C_plastic_cauchy_tensor = cauchy - cauchy_prev \
            - delta_cauchy
        C_plastic_cauchy = \
            get_vector_from_sym_tensor(C_plastic_cauchy_tensor, 3) \
            / scale_factor
        C_plastic_alpha = yield_fun

        if def_type == def_type.FULL_3D:
            C_elastic = jnp.r_[C_elastic_cauchy, C_elastic_alpha]
            C_plastic = jnp.r_[C_plastic_cauchy, C_plastic_alpha]

        elif def_type == def_type.PLANE_STRESS or \
                def_type == def_type.UNIAXIAL_STRESS:

            Q = params["rotation matrix"]
            global_trial_delta_cauchy = Q @ trial_delta_cauchy @ Q.T
            global_delta_cauchy = Q @ delta_cauchy @ Q.T

            if def_type == def_type.PLANE_STRESS:
                C_elastic_stretch = global_trial_delta_cauchy[2, 2] \
                    / scale_factor
                C_plastic_stretch = global_delta_cauchy[2, 2] / scale_factor

                C_elastic = jnp.r_[C_elastic_cauchy, C_elastic_alpha,
                                   C_elastic_stretch]
                C_plastic = jnp.r_[C_plastic_cauchy, C_plastic_alpha,
                                   C_plastic_stretch]

            elif def_type == def_type.UNIAXIAL_STRESS:
                off_axis_stress_idx = off_axis_idx(uniaxial_stress_idx)
                first_idx = off_axis_stress_idx[0]
                second_idx = off_axis_stress_idx[1]

                C_elastic_stretch = jnp.r_[
                    global_trial_delta_cauchy[first_idx, first_idx],
                    global_trial_delta_cauchy[second_idx, second_idx]] \
                    / scale_factor
                C_plastic_stretch = jnp.r_[
                    global_delta_cauchy[first_idx, first_idx],
                    global_delta_cauchy[second_idx, second_idx]] \
                    / scale_factor
                C_elastic_delta_strain = jnp.r_[
                    global_trial_delta_cauchy[0, 1],
                    global_trial_delta_cauchy[0, 2],
                    global_trial_delta_cauchy[1, 2]
                ] / scale_factor
                C_plastic_delta_strain = jnp.r_[
                    global_delta_cauchy[0, 1],
                    global_delta_cauchy[0, 2],
                    global_delta_cauchy[1, 2]
                ] / scale_factor

                C_elastic = jnp.r_[C_elastic_cauchy, C_elastic_alpha,
                                   C_elastic_stretch, C_elastic_delta_strain]
                C_plastic = jnp.r_[C_plastic_cauchy, C_plastic_alpha,
                                   C_plastic_stretch, C_plastic_delta_strain]

        return cond_residual(yield_fun, C_elastic, C_plastic, yield_tol)

    def _check_params(self, parameters):
        raise NotImplementedError

    @staticmethod
    def cauchy(xi, xi_prev, params, u, u_prev, def_type) -> jnp.array:

        Q = params["rotation matrix"]
        global_cauchy = Q @ get_sym_tensor_from_vector(xi[0], 3) @ Q.T
        return global_cauchy
