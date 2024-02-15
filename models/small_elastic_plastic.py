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


def yield_fun_scale_factor(params):
    E, nu = unpack_elastic_params(params)
    lame_mu = compute_mu(E, nu)
    return 2. * lame_mu


def compute_eps(u):
    F = u[0]
    ndims = F.shape[0]
    I = jnp.eye(ndims)
    grad_u = F - I
    eps = 0.5 * (grad_u + grad_u.T)

    return eps


def compute_cauchy_3D(xi, params, u):
    ndims = 3
    I = jnp.eye(ndims)

    E, nu = unpack_elastic_params(params)
    lame_lambda = compute_lambda(E, nu)
    lame_mu = compute_mu(E, nu)

    eps = compute_eps(u)
    pstrain = get_sym_tensor_from_vector(xi[0], ndims)

    cauchy = lame_lambda * jnp.trace(eps) * I \
        + 2. * lame_mu * (eps - pstrain)

    return cauchy


def compute_cauchy_plane_stress(xi, params, u):
    ndims = 2
    I = jnp.eye(ndims)

    E, nu = unpack_elastic_params(params)
    lame_lambda = compute_lambda(E, nu)
    lame_mu = compute_mu(E, nu)

    eps = compute_eps(u)
    pstrain = get_sym_tensor_from_vector(xi[0], ndims)

    eps_33 = -(lame_lambda * jnp.trace(eps)
               + 2. * lame_mu * jnp.trace(pstrain)) \
        / (lame_lambda + 2. * lame_mu)

    cauchy = lame_lambda * (jnp.trace(eps) + eps_33) * I \
        + 2. * lame_mu * (eps - pstrain)

    cauchy_3D = put_tensor_into_3D(cauchy, DefType.PLANE_STRESS)

    return cauchy_3D


def compute_cauchy_uniaxial_stress(xi, params, u):
    ndims = 1

    E, nu = unpack_elastic_params(params)
    lame_lambda = compute_lambda(E, nu)
    lame_mu = compute_mu(E, nu)

    eps = compute_eps(u)
    pstrain = get_sym_tensor_from_vector(xi[0], ndims)

    cauchy = E * (eps - pstrain)
    cauchy_3D = put_tensor_into_3D(cauchy, DefType.UNIAXIAL_STRESS)

    return cauchy_3D


def compute_yield_fun_and_normal(xi, xi_prev, params, u, u_prev, cauchy_fun,
                                 effective_stress, hardening, def_type):

    # model parameters
    E, nu = unpack_elastic_params(params)
    lame_mu = compute_mu(E, nu)
    plastic_params = params["plastic"]
    Y = plastic_params["flow stress"]["initial yield"]["Y"]
    hardening_params = plastic_params["flow stress"]["hardening"]

    # yield function and normal
    pstrain = get_sym_tensor_from_vector(xi[0], 1)
    alpha = get_scalar(xi[1])
    cauchy_3D = cauchy_fun(xi, xi_prev, params, u, u_prev)
    phi = effective_stress(cauchy_3D, plastic_params)
    sigma_flow = Y + hardening(alpha, hardening_params)
    scale_factor = yield_fun_scale_factor(params)
    yield_fun = (phi - sigma_flow) / scale_factor
    yield_normal_3D = grad(effective_stress)(cauchy_3D, plastic_params)
    yield_normal = get_tensor_from_3D(yield_normal_3D, def_type)

    return yield_fun, yield_normal


class SmallElasticPlastic(Model):
    """
    Small strain elastic-plastic model:
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

        # linearized plastic strain tensor
        self.resid_names[0] = "pstrain"
        self._var_types[0] = VarType.SYM_TENSOR
        self._num_eqs[0] = get_num_eqs(VarType.SYM_TENSOR, ndims)
        self._pstrain_idx = 0

        # isotropic hardening variable
        self.resid_names[1] = "alpha"
        self._var_types[1] = VarType.SCALAR
        self._num_eqs[1] = get_num_eqs(VarType.SCALAR, ndims)
        self._alpha_idx = 1

        # set the initial values for xi and xi_prev
        self._init_state_variables()
        init_vec_pstrain = np.zeros(self._num_eqs[self._pstrain_idx])
        init_alpha = np.zeros(self._num_eqs[self._alpha_idx])
        self._init_xi = [init_vec_pstrain, init_alpha]
        self.set_xi_to_init_vals()

        # TODO: check that the parameters make sense for this model
        # self._check_params(parameters)
        self.parameters = parameters

        effective_stress_type = \
            list(parameters.values["plastic"]["effective stress"])[0]

        cauchy = partial(self.cauchy, def_type=def_type)

        residual = partial(self._residual,
                           cauchy_fun=cauchy,
                           def_type=def_type,
                           effective_stress=effective_stress_fun(
                               effective_stress_type),
                           hardening=partial(combined_hardening_fun,
                                             hardening_funs=hardening_funs),
                           yield_tol=yield_tol)

        super().__init__(residual, cauchy)

    @staticmethod
    def _residual(xi, xi_prev, params, u, u_prev, cauchy_fun, 
            def_type, effective_stress, hardening, yield_tol) -> jnp.array:

        ndims = def_type_ndims(def_type)

        # state variables for the model
        pstrain = get_sym_tensor_from_vector(xi[0], ndims)
        pstrain_prev = get_sym_tensor_from_vector(xi_prev[0], ndims)
        alpha = get_scalar(xi[1])
        alpha_prev = get_scalar(xi_prev[1])

        # intermediate quantities
        delta_gamma = alpha - alpha_prev
        yield_fun, yield_normal = compute_yield_fun_and_normal(
            xi, xi_prev, params, u, u_prev, cauchy_fun, effective_stress,
            hardening, def_type)

        # elastic residual
        C_elastic_pstrain_tensor = pstrain - pstrain_prev
        C_elastic_pstrain = \
            get_vector_from_sym_tensor(C_elastic_pstrain_tensor, ndims)
        C_elastic_alpha = delta_gamma
        C_elastic = jnp.r_[C_elastic_pstrain, C_elastic_alpha]

        # plastic residual
        C_plastic_pstrain_tensor = C_elastic_pstrain_tensor \
            - delta_gamma * yield_normal
        C_plastic_pstrain = \
            get_vector_from_sym_tensor(C_plastic_pstrain_tensor, ndims)
        C_plastic_alpha = yield_fun
        C_plastic = jnp.r_[C_plastic_pstrain, C_plastic_alpha]

        return cond_residual(yield_fun, C_elastic, C_plastic, yield_tol)

    def _check_params(self, parameters):
        raise NotImplementedError

    @staticmethod
    def cauchy(xi, xi_prev, params, u, u_prev, def_type) -> jnp.array:
        """
        Returns the cauchy stress tensor.
        """
        if def_type == DefType.FULL_3D:
            return compute_cauchy_3D(xi, params, u)
        elif def_type == DefType.PLANE_STRESS:
            return compute_cauchy_plane_stress(xi, params, u)
        elif def_type == DefType.UNIAXIAL_STRESS:
            return compute_cauchy_uniaxial_stress(xi, params, u)
        else:
            raise NotImplementedError
