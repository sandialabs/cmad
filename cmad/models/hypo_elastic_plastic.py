"""Finite deformation rate form (hypoelastic) elastic-plastic model.

Integrates the unrotated stress rate law ``σ̇ = ℂ:(D − γ̇ n)`` in the
corotational frame of the polar rotation ``R = polar(F)``: the unrotated rate
of deformation ``D``, the unrotated Cauchy stress ``TC = xi[0]``, and the yield
normal ``n`` live in that frame, and the output stress rotates back as
``σ = R TC Rᵀ``. The plastic multiplier rate ``γ̇ = Δα/dt`` and the stress rate
``(TC − TC_prev)/dt`` carry the step increment ``dt`` from ``step_time``, so the
residual is the true rate form; under rate independent (von Mises) flow its
``dt`` cancels from the solution. State ``[unrotated_cauchy TC, alpha]``, FULL_3D.
"""
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, cast

import jax.numpy as jnp
import numpy as np
from jax import grad, jit

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.effective_stress import conventional_effective_stress_fun
from cmad.models.elastic_constants import ElasticConstants
from cmad.models.elastic_stress import (
    isotropic_linear_elastic_stress,
    two_mu_scale_factor,
)
from cmad.models.global_fields import GlobalFieldsAtPoint, StepTime
from cmad.models.hardening import combined_hardening_fun, get_hardening_funs
from cmad.models.kinematics import (
    gather_F,
    polar_rotation,
    unrotated_rate_of_deformation,
)
from cmad.models.mechanics_model import MechanicsModel
from cmad.models.paths import cond_residual, yield_threshold
from cmad.models.var_types import (
    VarType,
    get_num_eqs,
    get_scalar,
    get_sym_tensor_from_vector,
    get_vector_from_sym_tensor,
)
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray, Scalar, StateList


def elastic_predictor(
        xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
        step_time: StepTime,
        def_type: int, elastic_stress: Callable[..., JaxArray],
) -> StateList:
    """Elastic predictor state ``[TC_prev + dt * C:D, alpha_prev]``, the
    closed form root of the elastic branch.
    """
    dt = step_time.dt
    F = gather_F(xi_prev, U, def_type, -1)
    F_prev = gather_F(xi_prev, U_prev, def_type, -1)
    D = unrotated_rate_of_deformation(F, F_prev, dt)

    TC_prev = get_sym_tensor_from_vector(xi_prev[0], 3)
    TC_trial = TC_prev + dt * elastic_stress(D, params)

    return [get_vector_from_sym_tensor(TC_trial, 3), xi_prev[1]]


def compute_yield_fun(
        xi: StateList, params: dict[str, Any],
        effective_stress: Callable[..., JaxArray],
        hardening: Callable[..., JaxArray],
) -> JaxArray:
    """Yield value on the unrotated stress ``TC = xi[0]``."""
    plastic_params = params["plastic"]
    Y = plastic_params["flow stress"]["initial yield"]["Y"]
    hardening_params = plastic_params["flow stress"]["hardening"]

    TC = get_sym_tensor_from_vector(xi[0], 3)
    phi = effective_stress(TC, plastic_params)

    alpha = get_scalar(xi[1])
    sigma_flow = Y + hardening(alpha, hardening_params)

    return (phi - sigma_flow) / two_mu_scale_factor(params)


def compute_yield_fun_and_normal(
        xi: StateList, params: dict[str, Any],
        effective_stress: Callable[..., JaxArray],
        hardening: Callable[..., JaxArray],
        is_complex: bool,
) -> tuple[JaxArray, JaxArray]:
    """Yield value and flow normal on the unrotated stress ``TC = xi[0]``."""
    TC = get_sym_tensor_from_vector(xi[0], 3)
    yield_normal = grad(effective_stress, holomorphic=is_complex)(
        TC, params["plastic"])

    return compute_yield_fun(
        xi, params, effective_stress, hardening), yield_normal


class HypoElasticPlastic(MechanicsModel):
    """Finite deformation rate form elastic-plastic model.

    Elastic: isotropic linear response on the unrotated rate of deformation.
    Plastic: modular effective stress + hardening, rate independent
    (consistency). State ``[unrotated_cauchy TC, alpha]``.
    """

    supports_mixed: ClassVar[bool] = True
    is_finite_deformation = True

    _def_type: int
    _ndims: int

    def __init__(
            self, parameters: Parameters,
            def_type: int = DefType.FULL_3D,
            elastic_stress_fun: Callable[
                ..., JaxArray] = isotropic_linear_elastic_stress,
            effective_stress_fun: Callable[..., JaxArray] | None = None,
            hardening_funs: dict | None = None,
            yield_tol: float = 1e-12,
            is_complex: bool = False,
    ) -> None:

        if def_type != DefType.FULL_3D:
            raise NotImplementedError(
                "hypo_elastic_plastic currently supports FULL_3D")
        if hardening_funs is None:
            hardening_funs = get_hardening_funs()

        self._is_complex = is_complex
        self.dtype = complex if is_complex else float

        self._def_type = def_type
        self._ndims = def_type_ndims(def_type)

        self._init_residuals(2)

        # unrotated Cauchy stress state
        self.var_names[0] = "unrotated_cauchy"
        self.resid_names[0] = "unrotated stress"
        self._var_types[0] = VarType.SYM_TENSOR
        self._num_eqs[0] = get_num_eqs(VarType.SYM_TENSOR, 3)
        init_vec_cauchy = np.zeros(self._num_eqs[0])

        # isotropic hardening variable
        self.var_names[1] = "alpha"
        self.resid_names[1] = "yield surface"
        self._var_types[1] = VarType.SCALAR
        self._num_eqs[1] = get_num_eqs(VarType.SCALAR, 3)
        init_alpha = np.zeros(self._num_eqs[1])

        self._init_xi = [init_vec_cauchy, init_alpha]

        self._init_state_variables()
        self.set_xi_to_init_vals()

        self.parameters = parameters

        if effective_stress_fun is None:
            plastic_subtree = cast(dict[str, Any], parameters.values["plastic"])
            effective_stress_type = \
                next(iter(plastic_subtree["effective stress"]))
            effective_stress_fun = \
                conventional_effective_stress_fun(effective_stress_type)

        residual = partial(
            self._residual_fn, def_type=def_type,
            elastic_stress=elastic_stress_fun,
            effective_stress=effective_stress_fun,
            hardening=partial(combined_hardening_fun,
                              hardening_funs=hardening_funs),
            yield_tol=yield_tol, is_complex=is_complex)

        cauchy = partial(self._cauchy_fn, def_type=def_type)

        self.initial_guess_fn = jit(partial(
            elastic_predictor,
            def_type=def_type, elastic_stress=elastic_stress_fun))

        super().__init__(residual, cauchy)

    @classmethod
    def from_deck(
            cls,
            model_section: dict[str, Any],
            parameters: Parameters,
            def_type: int,
    ) -> "HypoElasticPlastic":
        return cls(parameters=parameters, def_type=def_type)

    def derived_output_field_names(self) -> list[str]:
        return ["cauchy"]

    @staticmethod
    def _residual_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            step_time: StepTime,
            def_type: int,
            elastic_stress: Callable[..., JaxArray],
            effective_stress: Callable[..., JaxArray],
            hardening: Callable[..., JaxArray],
            yield_tol: float, is_complex: bool,
    ) -> JaxArray:

        dt = step_time.dt
        TC = get_sym_tensor_from_vector(xi[0], 3)
        TC_prev = get_sym_tensor_from_vector(xi_prev[0], 3)
        alpha = get_scalar(xi[1])
        alpha_prev = get_scalar(xi_prev[1])

        F = gather_F(xi, U, def_type, -1)
        F_prev = gather_F(xi_prev, U_prev, def_type, -1)
        D = unrotated_rate_of_deformation(F, F_prev, dt)

        delta_gamma = alpha - alpha_prev
        gamma_dot = delta_gamma / dt
        scale_factor = two_mu_scale_factor(params)

        stress_rate = (TC - TC_prev) / dt
        trial_stress_rate = elastic_stress(D, params)

        # elastic residual
        C_elastic_cauchy_tensor = stress_rate - trial_stress_rate
        C_elastic_cauchy = \
            get_vector_from_sym_tensor(C_elastic_cauchy_tensor, 3) \
            / scale_factor
        C_elastic_alpha = delta_gamma

        # plastic residual
        yield_fun, yield_normal = compute_yield_fun_and_normal(
            xi, params, effective_stress, hardening, is_complex)
        xi_elastic = elastic_predictor(
            xi_prev, params, U, U_prev, step_time, def_type, elastic_stress)
        trial_yield_fun = compute_yield_fun(
            xi_elastic, params, effective_stress, hardening)
        plastic_rate = gamma_dot * yield_normal
        response_stress_rate = trial_stress_rate \
            - elastic_stress(plastic_rate, params)
        C_plastic_cauchy_tensor = stress_rate - response_stress_rate
        C_plastic_cauchy = \
            get_vector_from_sym_tensor(C_plastic_cauchy_tensor, 3) \
            / scale_factor
        C_plastic_alpha = yield_fun

        C_elastic = jnp.r_[C_elastic_cauchy, C_elastic_alpha]
        C_plastic = jnp.r_[C_plastic_cauchy, C_plastic_alpha]

        return cond_residual(trial_yield_fun, C_elastic, C_plastic,
                             yield_threshold(yield_tol, params))

    @staticmethod
    def _cauchy_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint, def_type: int,
    ) -> JaxArray:

        TC = get_sym_tensor_from_vector(xi[0], 3)
        F = gather_F(xi, U, def_type, -1)
        R = polar_rotation(F)
        return R @ TC @ R.T

    def dev_cauchy(
            self,
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> JaxArray:
        cauchy = self.cauchy(xi, xi_prev, params, U, U_prev)
        return cauchy - jnp.trace(cauchy) / 3. * jnp.eye(3)

    def hydro_cauchy(
            self,
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> Scalar:
        cauchy = self.cauchy(xi, xi_prev, params, U, U_prev)
        return jnp.trace(cauchy) / 3.

    @staticmethod
    def pressure_scale_factor(params: dict[str, Any]) -> Scalar:
        return ElasticConstants.from_params(params["elastic"]).kappa

    @staticmethod
    def shear_scale_factor(params: dict[str, Any]) -> Scalar:
        return ElasticConstants.from_params(params["elastic"]).mu
