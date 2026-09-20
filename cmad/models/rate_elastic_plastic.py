"""Rate form elastic-plastic model, small strain or finite deformation.

Integrates the stress rate equation ``σ̇ = ℂ:(D − γ̇ n)`` in the material
frame. Its input is the increment ``ε − ε_prev`` for small strain and
``D Δt = Rᵀ sym((F − F_prev) F_mid⁻¹) R`` for finite deformation, taken
into the material frame as ``Qᵀ (·) Q`` when the material has a
``rotation matrix`` ``Q`` (material to global). The stress comes out in
reverse, ``σ = R (Q s Qᵀ) Rᵀ`` with ``R = polar_rotation(F)``.

Every term of the stress rate equation is a rate, so the equation times
``Δt`` has no ``Δt`` left, and the stress residual is written in
increments. ``Δt`` enters through ``α̇ = Δα/Δt`` in the yield function
only. A stress equation with a term that is not a rate (viscoelastic
relaxation) must keep its ``Δt``.
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
from cmad.models.flow_stress import make_yield_function
from cmad.models.global_fields import (
    GlobalFieldsAtPoint,
    StepTime,
    temperature_at_point,
)
from cmad.models.kinematics import (
    gather_F,
    off_axis_idx,
    polar_rotation,
    small_strain_increment,
    unrotated_rate_of_deformation_increment,
)
from cmad.models.mechanics_model import MechanicsModel, require_def_type
from cmad.models.paths import cond_residual, yield_threshold
from cmad.models.var_types import (
    VarType,
    get_num_eqs,
    get_scalar,
    get_sym_tensor_from_vector,
    get_vector,
    get_vector_from_sym_tensor,
)
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray, Scalar, StateList


def material_frame_increment(
        xi: StateList, xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
        def_type: int, uniaxial_stress_idx: int,
        finite_deformation: bool, has_material_rotation: bool,
) -> JaxArray:
    """Input of the stress equation in the material frame: ``ε − ε_prev``,
    or ``D Δt`` for finite deformation."""

    local_var_idx = 2
    F = gather_F(xi, U, def_type, local_var_idx, uniaxial_stress_idx)
    F_prev = gather_F(xi_prev, U_prev, def_type, local_var_idx,
        uniaxial_stress_idx)

    if finite_deformation:
        increment = unrotated_rate_of_deformation_increment(
            F, F_prev, def_type)
    else:
        increment = small_strain_increment(F, F_prev)

    if def_type == DefType.UNIAXIAL_STRESS:
        off_axis = get_vector(xi[3], 3)
        increment = jnp.array([
            [increment[0, 0], off_axis[0], off_axis[1]],
            [off_axis[0], increment[1, 1], off_axis[2]],
            [off_axis[1], off_axis[2], increment[2, 2]],
        ])

    if has_material_rotation:
        # Q is a rotation from material coordinates to global coordinates
        # Q_{ij} = e_i (global) \dot e_j (material)
        Q = params["rotation matrix"]
        return Q.T @ increment @ Q

    return increment


def rotate_out_of_material_frame(
        stress: JaxArray, params: dict[str, Any],
        has_material_rotation: bool,
) -> JaxArray:
    """``Q s Qᵀ`` when the material has a rotation matrix, ``s``
    otherwise."""
    if has_material_rotation:
        Q = params["rotation matrix"]
        return Q @ stress @ Q.T

    return stress


def elastic_predictor(
        xi: StateList, xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
        def_type: int, uniaxial_stress_idx: int,
        finite_deformation: bool, has_material_rotation: bool,
        elastic_stress: Callable[..., JaxArray],
) -> StateList:
    """Elastic predictor state ``[cauchy_prev + C:increment,
    alpha_prev]``, the closed form root of the elastic branch.
    """
    increment = material_frame_increment(
        xi, xi_prev, params, U, U_prev, def_type, uniaxial_stress_idx,
        finite_deformation, has_material_rotation)
    cauchy_prev = get_sym_tensor_from_vector(xi_prev[0], 3)
    cauchy_trial = cauchy_prev + elastic_stress(increment, params)

    return [
        get_vector_from_sym_tensor(cauchy_trial, 3),
        xi_prev[1],
        *xi[2:],
    ]


def initial_guess(
        xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
        step_time: StepTime,
        def_type: int, uniaxial_stress_idx: int,
        finite_deformation: bool, has_material_rotation: bool,
        elastic_stress: Callable[..., JaxArray],
) -> StateList:
    """Starting state for the local Newton: the elastic predictor taken at
    the previous stretches, no current iterate existing yet.
    """
    return elastic_predictor(
        xi_prev, xi_prev, params, U, U_prev, def_type, uniaxial_stress_idx,
        finite_deformation, has_material_rotation, elastic_stress)


def compute_yield_fun(
        xi: StateList, xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, step_time: StepTime, def_type: int,
        effective_stress: Callable[..., JaxArray],
        yield_function: Callable[..., JaxArray],
) -> JaxArray:

    def_type_ndims(def_type)

    plastic_params = params["plastic"]

    cauchy = get_sym_tensor_from_vector(xi[0], 3)
    phi = effective_stress(cauchy, plastic_params)

    alpha = get_scalar(xi[1])
    alpha_prev = get_scalar(xi_prev[1])
    alpha_dot = (alpha - alpha_prev) / step_time.dt
    yield_fun = yield_function(phi, alpha, alpha_dot, temperature_at_point(U),
                               plastic_params["flow stress"])

    return yield_fun / two_mu_scale_factor(params)


def compute_yield_fun_and_normal(
        xi: StateList, xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, step_time: StepTime, def_type: int,
        effective_stress: Callable[..., JaxArray],
        yield_function: Callable[..., JaxArray],
        is_complex: bool,
) -> tuple[JaxArray, JaxArray]:

    cauchy = get_sym_tensor_from_vector(xi[0], 3)
    yield_normal = grad(effective_stress, holomorphic=is_complex)(
        cauchy, params["plastic"])

    return compute_yield_fun(
        xi, xi_prev, params, U, step_time, def_type, effective_stress,
        yield_function), yield_normal


class RateElasticPlastic(MechanicsModel):
    """
    Rate form elastic-plastic model, small strain or finite deformation:
    Elastic: Modular linear elasticity
    Plastic: Modular effective stress and hardening
    """

    supports_mixed: ClassVar[bool] = True

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
            uniaxial_stress_idx: int = 0,
            is_complex: bool = False,
            finite_deformation: bool = False,
    ) -> None:

        self.is_finite_deformation = finite_deformation
        has_material_rotation = "rotation matrix" in parameters.values

        self._is_complex = is_complex
        self.dtype = float
        if is_complex:
            self.dtype = complex

        self._def_type = def_type
        ndims = def_type_ndims(def_type)
        self._ndims = ndims

        if def_type == DefType.FULL_3D or def_type == DefType.PLANE_STRAIN:
            num_residuals = 2

        elif def_type == DefType.PLANE_STRESS:
            num_residuals = 3

        elif def_type == DefType.UNIAXIAL_STRESS:
            num_residuals = 4

        else:
            raise NotImplementedError

        self._init_residuals(num_residuals)

        # unrotated (material-frame) cauchy stress state
        self.var_names[0] = "unrotated_cauchy"
        self.resid_names[0] = "material stress"
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

        if def_type == DefType.PLANE_STRESS:
            # out of plane stretch
            self.var_names[2] = "out of plane stretch"
            self.resid_names[2] = "cauchy_33"
            self._var_types[2] = VarType.SCALAR
            self._num_eqs[2] = get_num_eqs(VarType.SCALAR, ndims)
            init_oop_stretch = np.ones(self._num_eqs[2])
            self._oop_stretch_idx = 2

            self._init_xi += [init_oop_stretch]

        elif def_type == DefType.UNIAXIAL_STRESS:
            # off-axis stretches
            self.var_names[2] = "off-axis stretches"
            self.resid_names[2] = "off-axis normal stress"
            self._var_types[2] = VarType.VECTOR
            self._num_eqs[2] = get_num_eqs(VarType.VECTOR, 2)
            init_off_axis_stretches = np.ones(self._num_eqs[2])

            # off-axis delta strains
            self.var_names[3] = "off-axis delta strains"
            self.resid_names[3] = "off-axis shear stress"
            self._var_types[3] = VarType.VECTOR
            self._num_eqs[3] = get_num_eqs(VarType.VECTOR, 3)
            init_off_axis_delta_strains = np.zeros(self._num_eqs[3])

            self._init_xi += [init_off_axis_stretches,
                              init_off_axis_delta_strains]

        # set the initial values for xi and xi_prev
        self._init_state_variables()
        self.set_xi_to_init_vals()

        # TODO: check that the parameters make sense for this model
        # self._check_params(parameters)
        self.parameters = parameters

        plastic_subtree = cast(dict[str, Any], parameters.values["plastic"])
        if effective_stress_fun is None:
            effective_stress_type = \
                next(iter(plastic_subtree["effective stress"]))
            effective_stress_fun = \
                conventional_effective_stress_fun(effective_stress_type)
        yield_function = make_yield_function(
            plastic_subtree["flow stress"], hardening_funs)

        residual = partial(self._residual_fn, def_type=def_type,
                           elastic_stress=elastic_stress_fun,
                           effective_stress=effective_stress_fun,
                           yield_function=yield_function,
                           yield_tol=yield_tol,
                           uniaxial_stress_idx=uniaxial_stress_idx,
                           is_complex=is_complex,
                           finite_deformation=finite_deformation,
                           has_material_rotation=has_material_rotation)

        cauchy = partial(self._cauchy_fn, def_type=def_type,
                         finite_deformation=finite_deformation,
                         has_material_rotation=has_material_rotation)

        self.initial_guess_fn = jit(partial(
            initial_guess, def_type=def_type,
            uniaxial_stress_idx=uniaxial_stress_idx,
            finite_deformation=finite_deformation,
            has_material_rotation=has_material_rotation,
            elastic_stress=elastic_stress_fun))

        super().__init__(residual, cauchy)

    @classmethod
    def from_deck(
            cls,
            model_section: dict[str, Any],
            parameters: Parameters,
            def_type: int | None,
    ) -> "RateElasticPlastic":
        return cls(
            parameters=parameters,
            def_type=require_def_type(def_type, cls.__name__),
            uniaxial_stress_idx=model_section.get("uniaxial_stress_idx", 0),
            finite_deformation=model_section.get("finite deformation", False),
        )

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
            yield_function: Callable[..., JaxArray],
            yield_tol: float, uniaxial_stress_idx: int, is_complex: bool,
            finite_deformation: bool, has_material_rotation: bool,
    ) -> JaxArray:

        # state variables for the model
        cauchy = get_sym_tensor_from_vector(xi[0], 3)
        cauchy_prev = get_sym_tensor_from_vector(xi_prev[0], 3)
        alpha = get_scalar(xi[1])
        alpha_prev = get_scalar(xi_prev[1])

        increment = material_frame_increment(
            xi, xi_prev, params, U, U_prev, def_type, uniaxial_stress_idx,
            finite_deformation, has_material_rotation)
        trial_delta_cauchy = elastic_stress(increment, params)
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
            compute_yield_fun_and_normal(xi, xi_prev, params, U, step_time,
                                         def_type, effective_stress,
                                         yield_function, is_complex)
        xi_elastic = elastic_predictor(
            xi, xi_prev, params, U, U_prev, def_type, uniaxial_stress_idx,
            finite_deformation, has_material_rotation, elastic_stress)
        trial_yield_fun = \
            compute_yield_fun(xi_elastic, xi_prev, params, U, step_time,
                              def_type, effective_stress, yield_function)
        plastic_increment = delta_gamma * yield_normal
        delta_cauchy = trial_delta_cauchy \
            - elastic_stress(plastic_increment, params)
        C_plastic_cauchy_tensor = cauchy - cauchy_prev \
            - delta_cauchy
        C_plastic_cauchy = \
            get_vector_from_sym_tensor(C_plastic_cauchy_tensor, 3) \
            / scale_factor
        C_plastic_alpha = yield_fun

        if def_type == DefType.FULL_3D or def_type == DefType.PLANE_STRAIN:
            C_elastic = jnp.r_[C_elastic_cauchy, C_elastic_alpha]
            C_plastic = jnp.r_[C_plastic_cauchy, C_plastic_alpha]

        elif def_type == DefType.PLANE_STRESS or \
                def_type == DefType.UNIAXIAL_STRESS:

            global_trial_delta_cauchy = rotate_out_of_material_frame(
                trial_delta_cauchy, params, has_material_rotation)
            global_delta_cauchy = rotate_out_of_material_frame(
                delta_cauchy, params, has_material_rotation)

            if def_type == DefType.PLANE_STRESS:
                C_elastic_stretch = global_trial_delta_cauchy[2, 2] \
                    / scale_factor
                C_plastic_stretch = global_delta_cauchy[2, 2] / scale_factor

                C_elastic = jnp.r_[C_elastic_cauchy, C_elastic_alpha,
                                   C_elastic_stretch]
                C_plastic = jnp.r_[C_plastic_cauchy, C_plastic_alpha,
                                   C_plastic_stretch]

            elif def_type == DefType.UNIAXIAL_STRESS:
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

        return cond_residual(trial_yield_fun, C_elastic, C_plastic,
                             yield_threshold(yield_tol, params, yield_function))

    def _check_params(self, parameters: Parameters) -> None:
        raise NotImplementedError

    @staticmethod
    def _cauchy_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint, def_type: int,
            finite_deformation: bool, has_material_rotation: bool,
    ) -> JaxArray:

        cauchy = rotate_out_of_material_frame(
            get_sym_tensor_from_vector(xi[0], 3), params,
            has_material_rotation)
        if finite_deformation:
            R = polar_rotation(gather_F(xi, U, def_type, 2), def_type)
            cauchy = R @ cauchy @ R.T

        return cauchy

    @staticmethod
    def pressure_scale_factor(params: dict[str, Any]) -> Scalar:
        return ElasticConstants.from_params(params["elastic"]).kappa

    @staticmethod
    def shear_scale_factor(params: dict[str, Any]) -> Scalar:
        return ElasticConstants.from_params(params["elastic"]).mu
