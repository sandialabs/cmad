import numpy as np
import jax.numpy as jnp

from jax import grad

from functools import partial

from cmad.models.deformation_types import (DefType, def_type_ndims)
from cmad.models.effective_stress import effective_stress_fun
from cmad.models.elastic_constants import compute_lambda, compute_mu
from cmad.models.hardening import combined_hardening_fun, get_hardening_funs
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters, unpack_elastic_params
from cmad.models.paths import cond_residual
from cmad.models.var_types import (
    VarType,
    get_num_eqs,
    get_scalar,
    get_sym_tensor_from_vector,
    get_vector_from_sym_tensor,
    get_tensor_from_vector,
    put_tensor_into_3D,
    get_tensor_from_3D)


def residual_scale_factor(params):
    E, nu = unpack_elastic_params(params)
    lame_mu = compute_mu(E, nu)
    return 2. * lame_mu


def compute_delta_eps(u, u_prev):
    # global state variables
    F = u[0]
    F_prev = u_prev[0]

    # compute linearized strain tensors from F and F_prev
    ndims = F.shape[0]
    I = jnp.eye(ndims)

    grad_u = F - I
    eps = 0.5 * (grad_u + grad_u.T)

    grad_u_prev = F_prev - I
    eps_prev = 0.5 * (grad_u_prev + grad_u_prev.T)

    delta_eps = eps - eps_prev

    return delta_eps


def compute_yield_fun_and_normal(xi, params, effective_stress, hardening,
                                 def_type):

    ndims = def_type_ndims(def_type)
    cauchy = get_sym_tensor_from_vector(xi[0], ndims)
    alpha = get_scalar(xi[1])

    # model parameters
    E, nu, = unpack_elastic_params(params)
    lame_mu = compute_mu(E, nu)
    plastic_params = params["plastic"]
    Y = plastic_params["flow stress"]["initial yield"]["Y"]
    hardening_params = plastic_params["flow stress"]["hardening"]

    # yield function and normal
    cauchy_3D = put_tensor_into_3D(cauchy, def_type)
    phi = effective_stress(cauchy_3D, plastic_params)
    sigma_flow = Y + hardening(alpha, hardening_params)
    scale_factor = residual_scale_factor(params)
    yield_fun = (phi - sigma_flow) / scale_factor
    yield_normal_3D = grad(effective_stress)(cauchy_3D, plastic_params)
    yield_normal = get_tensor_from_3D(yield_normal_3D, def_type)

    return yield_fun, yield_normal


def residual_components(xi, xi_prev, params, u, u_prev,
                        def_type, effective_stress, hardening, yield_tol):

    ndims = def_type_ndims(def_type)
    I = jnp.eye(ndims)

    # state variables for the model
    cauchy = get_sym_tensor_from_vector(xi[0], ndims)
    cauchy_prev = get_sym_tensor_from_vector(xi_prev[0], ndims)
    alpha = get_scalar(xi[1])
    alpha_prev = get_scalar(xi_prev[1])

    # elastic constants
    E, nu = unpack_elastic_params(params)
    lame_lambda = compute_lambda(E, nu)
    lame_mu = compute_mu(E, nu)

    # intermediate quantities
    delta_eps = compute_delta_eps(u, u_prev)
    delta_gamma = alpha - alpha_prev
    yield_fun, yield_normal = \
        compute_yield_fun_and_normal(xi, params, effective_stress, hardening,
                                     def_type)

    # elastic residual
    C_elastic_cauchy_tensor = cauchy - cauchy_prev \
        - lame_lambda * jnp.trace(delta_eps) * I \
        - 2. * lame_mu * delta_eps
    C_elastic_alpha = delta_gamma

    # plastic residual
    C_plastic_cauchy_tensor = C_elastic_cauchy_tensor \
        + 2. * lame_mu * delta_gamma * yield_normal
    C_plastic_alpha = yield_fun

    return (C_elastic_cauchy_tensor, C_elastic_alpha,
            C_plastic_cauchy_tensor, C_plastic_alpha)


class SmallRateElasticPlastic(Model):
    """
    Small strain rate form elastic-plastic model:
    Elastic: Linear elasticity
    Plastic: Modular effective stress and hardening
    """

    def __init__(self, parameters: Parameters,
                 def_type=DefType.FULL_3D,
                 hardening_funs: dict = get_hardening_funs(),
                 yield_tol=1e-14):

        self._def_type = def_type
        ndims = def_type_ndims(def_type)
        self._ndims = ndims

        num_residuals = 2
        self._init_residuals(num_residuals)

        # cauchy stress tensor
        self.resid_names[0] = "cauchy"
        self._var_types[0] = VarType.SYM_TENSOR
        self._num_eqs[0] = get_num_eqs(VarType.SYM_TENSOR, ndims)
        self._cauchy_idx = 0

        # isotropic hardening variable
        self.resid_names[1] = "alpha"
        self._var_types[1] = VarType.SCALAR
        self._num_eqs[1] = get_num_eqs(VarType.SCALAR, ndims)
        self._alpha_idx = 1

        # set the initial values for xi and xi_prev
        self._init_state_variables()
        init_vec_cauchy = np.zeros(self._num_eqs[self._cauchy_idx])
        init_alpha = np.zeros(self._num_eqs[self._alpha_idx])
        self._init_xi = [init_vec_cauchy, init_alpha]
        self.set_xi_to_init_vals()

        # TODO: check that the parameters make sense for this model
        # self._check_params(parameters)
        self.parameters = parameters

        effective_stress_type = \
            list(parameters.values["plastic"]["effective stress"])[0]

        if def_type == DefType.FULL_3D:
            residual_fun = self._residual_3D
        elif def_type == DefType.PLANE_STRESS:
            residual_fun = self._residual_plane_stress
        elif def_type == DefType.UNIAXIAL_STRESS:
            residual_fun = self._residual_uniaxial_stress
        else:
            raise NotImplementedError

        residual = partial(residual_fun, def_type=def_type,
                           effective_stress=effective_stress_fun(
                               effective_stress_type),
                           hardening=partial(combined_hardening_fun,
                                             hardening_funs=hardening_funs),
                           yield_tol=yield_tol)

        cauchy = partial(self.cauchy, def_type=def_type)

        super().__init__(residual, cauchy)

    @staticmethod
    def _residual_3D(xi, xi_prev, params, u, u_prev,
            def_type, effective_stress, hardening, yield_tol) -> jnp.array:

        ndims = def_type_ndims(def_type)

        residuals = residual_components(xi, xi_prev, params, u, u_prev,
            def_type, effective_stress, hardening, yield_tol)

        C_elastic_cauchy_tensor = residuals[0]
        C_elastic_alpha = residuals[1]
        C_plastic_cauchy_tensor = residuals[2]
        C_plastic_alpha = residuals[3]

        scale_factor = residual_scale_factor(params)

        C_elastic_cauchy = \
            get_vector_from_sym_tensor(C_elastic_cauchy_tensor, ndims)
        C_elastic = jnp.r_[C_elastic_cauchy / scale_factor, C_elastic_alpha]

        C_plastic_cauchy = \
            get_vector_from_sym_tensor(C_plastic_cauchy_tensor, ndims)
        C_plastic = jnp.r_[C_plastic_cauchy / scale_factor, C_plastic_alpha]

        return cond_residual(C_plastic_alpha, C_elastic, C_plastic, yield_tol)

    @staticmethod
    def _residual_plane_stress(xi, xi_prev, params, u, u_prev,
            def_type, effective_stress, hardening, yield_tol) -> jnp.array:

        ndims = def_type_ndims(def_type)
        I = jnp.eye(ndims)

        residuals = residual_components(xi, xi_prev, params, u, u_prev,
            def_type, effective_stress, hardening, yield_tol)

        C_elastic_cauchy_tensor = residuals[0]
        C_elastic_alpha = residuals[1]
        C_plastic_cauchy_tensor = residuals[2]
        C_plastic_alpha = residuals[3]

        scale_factor = residual_scale_factor(params)

        # elastic constants
        E, nu = unpack_elastic_params(params)
        lame_lambda = compute_lambda(E, nu)
        lame_mu = compute_mu(E, nu)

        # compute the out of plane strain increment (elastic only)
        delta_eps = compute_delta_eps(u, u_prev)
        delta_elastic_eps_33 = -(lame_lambda * jnp.trace(delta_eps)) \
            / (lame_lambda + 2. * lame_mu)

        C_elastic_cauchy_tensor_plane_stress = \
            C_elastic_cauchy_tensor - lame_lambda * delta_elastic_eps_33 * I
        C_elastic_cauchy = \
            get_vector_from_sym_tensor(C_elastic_cauchy_tensor_plane_stress,
                                       ndims)
        C_elastic = jnp.r_[C_elastic_cauchy / scale_factor, C_elastic_alpha]

        # compute the out of plane strain increment (elastic + plastic)
        yield_fun, yield_normal = \
            compute_yield_fun_and_normal(xi, params, effective_stress,
                                         hardening, def_type)
        delta_gamma = C_elastic_alpha
        delta_plastic_eps_33 = delta_elastic_eps_33 - jnp.trace(yield_normal) \
            * delta_gamma * 2. * lame_mu / (2. * lame_mu + lame_lambda)

        C_plastic_cauchy_tensor_plane_stress = \
            C_plastic_cauchy_tensor - lame_lambda * delta_plastic_eps_33 * I
        C_plastic_cauchy = \
            get_vector_from_sym_tensor(C_plastic_cauchy_tensor_plane_stress,
                                       ndims)
        C_plastic = jnp.r_[C_plastic_cauchy / scale_factor, C_plastic_alpha]

        return cond_residual(yield_fun, C_elastic, C_plastic, yield_tol)

    @staticmethod
    def _residual_uniaxial_stress(
        xi,
        xi_prev,
        params,
        u,
        u_prev,
        def_type,
        effective_stress,
        hardening,
            yield_tol) -> jnp.array:

        ndims = def_type_ndims(def_type)

        residuals = residual_components(xi, xi_prev, params, u, u_prev,
            def_type, effective_stress, hardening, yield_tol)

        C_elastic_cauchy_tensor = residuals[0]
        C_elastic_alpha = residuals[1]
        C_plastic_cauchy_tensor = residuals[2]
        C_plastic_alpha = residuals[3]

        scale_factor = residual_scale_factor(params)

        # elastic constants
        E, nu = unpack_elastic_params(params)
        lame_lambda = compute_lambda(E, nu)
        lame_mu = compute_mu(E, nu)

        # compute the elastic component of the lateral strain increment
        delta_eps_11 = compute_delta_eps(u, u_prev)
        delta_elastic_eps_lat = -nu * delta_eps_11

        C_elastic_cauchy_tensor_uniaxial_stress = \
            C_elastic_cauchy_tensor - 2. * lame_lambda * delta_elastic_eps_lat
        C_elastic_cauchy = \
            get_vector_from_sym_tensor(C_elastic_cauchy_tensor_uniaxial_stress,
                                       ndims)
        C_elastic = jnp.r_[C_elastic_cauchy / scale_factor, C_elastic_alpha]

        # compute the plastic component of the lateral strain increment
        yield_fun, yield_normal = \
            compute_yield_fun_and_normal(xi, params, effective_stress,
                                         hardening, def_type)
        delta_gamma = C_elastic_alpha
        delta_plastic_eps = delta_gamma * yield_normal
        delta_plastic_eps_lat = -lame_mu * delta_plastic_eps \
            / (2. * (lame_mu + lame_lambda))
        delta_eps_lat = delta_elastic_eps_lat + delta_plastic_eps_lat

        C_plastic_cauchy_tensor_uniaxial_stress = \
            C_plastic_cauchy_tensor - 2. * lame_lambda * delta_eps_lat
        C_plastic_cauchy = \
            get_vector_from_sym_tensor(C_plastic_cauchy_tensor_uniaxial_stress,
                                       ndims)
        C_plastic = jnp.r_[C_plastic_cauchy / scale_factor, C_plastic_alpha]

        return cond_residual(yield_fun, C_elastic, C_plastic, yield_tol)

    def _check_params(self, parameters):
        raise NotImplementedError

    @staticmethod
    def cauchy(xi, xi_prev, params, u, u_prev, def_type) -> jnp.array:
        """
        Returns the cauchy stress tensor.
        """
        ndims = def_type_ndims(def_type)
        cauchy = get_sym_tensor_from_vector(xi[0], ndims)
        return put_tensor_into_3D(cauchy, def_type)
