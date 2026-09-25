from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, cast

import jax.numpy as jnp
import numpy as np
from jax import grad, jit

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.effective_stress import conventional_effective_stress_fun
from cmad.models.elastic_stress import isotropic_linear_elastic_stress
from cmad.models.flow_stress import make_yield_function
from cmad.models.global_fields import (
    GlobalFieldsAtPoint,
    StepTime,
    temperature_at_point,
)
from cmad.models.kinematics import gather_F, off_axis_idx
from cmad.models.material_frame import (
    require_in_plane_rotation,
    rotate_into_material_frame,
    rotate_out_of_material_frame,
)
from cmad.models.mechanics_model import MechanicsModel, require_def_type
from cmad.models.paths import compute_yield_threshold, cond_residual
from cmad.models.radial_return import (
    plastic_multiplier_increment,
    resolve_initial_guess,
)
from cmad.models.var_types import (
    VarType,
    get_num_eqs,
    get_scalar,
    get_sym_tensor_from_vector,
    get_vector,
    get_vector_from_sym_tensor,
    put_2D_tensor_into_3D,
)
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray, StateList


def plastic_strain_33_idx(def_type: int) -> int:
    return 2 if def_type == DefType.PLANE_STRAIN else 3


def plastic_strain_from_state(
        xi: StateList, def_type: int, has_material_rotation: bool,
) -> JaxArray:
    """Material frame plastic strain as a 3x3, zero where the def type
    stores no component."""
    if def_type == DefType.PLANE_STRAIN or def_type == DefType.PLANE_STRESS:
        in_plane = put_2D_tensor_into_3D(get_sym_tensor_from_vector(xi[0], 2))
        plastic_strain_33 = get_scalar(xi[plastic_strain_33_idx(def_type)])
        return in_plane.at[2, 2].set(plastic_strain_33[0])
    if def_type == DefType.UNIAXIAL_STRESS and not has_material_rotation:
        return jnp.diag(get_vector(xi[0], 3))
    return get_sym_tensor_from_vector(xi[0], 3)


def stored_plastic_strain_components(
        A: JaxArray, def_type: int, has_material_rotation: bool,
) -> JaxArray:
    if def_type == DefType.PLANE_STRAIN or def_type == DefType.PLANE_STRESS:
        return get_vector_from_sym_tensor(A[:2, :2], 2)
    if def_type == DefType.UNIAXIAL_STRESS and not has_material_rotation:
        return jnp.diag(A)
    return get_vector_from_sym_tensor(A, 3)


def compute_elastic_strain(
        xi: StateList, params: dict[str, Any], U: GlobalFieldsAtPoint,
        def_type: int, uniaxial_stress_idx: int,
        has_material_rotation: bool,
) -> JaxArray:
    local_var_idx = 2
    F = gather_F(xi, U, def_type, local_var_idx, uniaxial_stress_idx)
    plastic_strain = plastic_strain_from_state(
        xi, def_type, has_material_rotation)
    grad_u = F - jnp.eye(3)
    global_total_strain = 0.5 * (grad_u + grad_u.T)

    if def_type == DefType.UNIAXIAL_STRESS:
        off_axis_global_plastic_strain = rotate_out_of_material_frame(
            plastic_strain, params, has_material_rotation)
        constrained_global_total_strain = jnp.array([
            [global_total_strain[0, 0],
            off_axis_global_plastic_strain[0, 1],
            off_axis_global_plastic_strain[0, 2]],
            [off_axis_global_plastic_strain[1, 0],
            global_total_strain[1, 1],
            off_axis_global_plastic_strain[1, 2]],
            [off_axis_global_plastic_strain[2, 0],
            off_axis_global_plastic_strain[2, 1],
            global_total_strain[2, 2]]
        ])
        material_total_strain = rotate_into_material_frame(
            constrained_global_total_strain, params, has_material_rotation)
    else:
        material_total_strain = rotate_into_material_frame(
            global_total_strain, params, has_material_rotation)

    return material_total_strain - plastic_strain


def compute_yield_fun(
        xi: StateList, xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, step_time: StepTime,
        def_type: int,
        elastic_stress: Callable[..., JaxArray],
        effective_stress: Callable[..., JaxArray],
        yield_function: Callable[..., JaxArray],
        shear_scale_factor: float,
        uniaxial_stress_idx: int, has_material_rotation: bool,
) -> tuple[JaxArray, JaxArray]:

    plastic_params = params["plastic"]

    elastic_strain = compute_elastic_strain(xi, params, U, def_type,
        uniaxial_stress_idx, has_material_rotation)
    cauchy = elastic_stress(elastic_strain, params)
    phi = effective_stress(cauchy, plastic_params)

    alpha = get_scalar(xi[1])
    alpha_prev = get_scalar(xi_prev[1])
    alpha_dot = (alpha - alpha_prev) / step_time.dt
    yield_fun = yield_function(phi, alpha, alpha_dot, temperature_at_point(U),
                               plastic_params["flow stress"])

    return cauchy, yield_fun / shear_scale_factor


def compute_yield_fun_and_normal(
        xi: StateList, xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
        step_time: StepTime,
        def_type: int,
        elastic_stress: Callable[..., JaxArray],
        effective_stress: Callable[..., JaxArray],
        yield_function: Callable[..., JaxArray],
        shear_scale_factor: float,
        uniaxial_stress_idx: int, is_complex: bool,
        has_material_rotation: bool,
) -> tuple[JaxArray, JaxArray, JaxArray]:

    cauchy, yield_fun = compute_yield_fun(
        xi, xi_prev, params, U, step_time, def_type, elastic_stress,
        effective_stress, yield_function, shear_scale_factor,
        uniaxial_stress_idx, has_material_rotation)
    yield_normal = grad(effective_stress, holomorphic=is_complex)(
        cauchy, params["plastic"])

    return cauchy, yield_fun, yield_normal


def start_from_radial_return(
        xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
        step_time: StepTime,
        def_type: int,
        elastic_stress: Callable[..., JaxArray],
        effective_stress: Callable[..., JaxArray],
        yield_function: Callable[..., JaxArray],
        shear_scale_factor: float, yield_threshold: float,
        uniaxial_stress_idx: int, has_material_rotation: bool,
) -> StateList:
    """Starting state for the local Newton: the previous state, its stress
    returned to the yield surface along its own normal when it lies
    outside."""
    cauchy_trial, _ = compute_yield_fun(
        xi_prev, xi_prev, params, U, step_time, def_type, elastic_stress,
        effective_stress, yield_function, shear_scale_factor,
        uniaxial_stress_idx, has_material_rotation)
    normal_trial = grad(effective_stress)(cauchy_trial, params["plastic"])
    pstrain_prev = plastic_strain_from_state(
        xi_prev, def_type, has_material_rotation)
    alpha_prev = get_scalar(xi_prev[1])

    def returned_state(delta_gamma: JaxArray) -> StateList:
        pstrain = pstrain_prev + delta_gamma * normal_trial
        returned = [
            stored_plastic_strain_components(
                pstrain, def_type, has_material_rotation),
            alpha_prev + delta_gamma,
        ]
        if def_type == DefType.PLANE_STRAIN:
            returned.append(jnp.array([pstrain[2, 2]]))
        return returned

    def g(delta_gamma: JaxArray) -> JaxArray:
        return compute_yield_fun(
            returned_state(delta_gamma), xi_prev, params, U, step_time,
            def_type, elastic_stress, effective_stress, yield_function,
            shear_scale_factor, uniaxial_stress_idx,
            has_material_rotation)[1][0]

    delta_gamma = plastic_multiplier_increment(g, yield_threshold)
    is_plastic = g(jnp.zeros(())) > yield_threshold

    return [
        jnp.where(is_plastic, returned_block, trial_block)
        for returned_block, trial_block in zip(
            returned_state(delta_gamma), xi_prev, strict=True)
    ]


class SmallElasticPlastic(MechanicsModel):
    """
    Small strain elastic-plastic model:
    Elastic: Modular linear elasticity
    Plastic: Modular effective stress and hardening
    """

    supports_mixed: ClassVar[bool] = True

    _def_type: int
    _ndims: int
    _uniaxial_stress_idx: int

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
            initial_guess: str | None = None,
    ) -> None:

        has_material_rotation = "rotation matrix" in parameters.values
        require_in_plane_rotation(
            parameters, def_type, "small_elastic_plastic")
        initial_guess = resolve_initial_guess(
            initial_guess, def_type, "small_elastic_plastic")

        self._is_complex = is_complex
        self.dtype = float
        if is_complex:
            self.dtype = complex

        self._def_type = def_type
        ndims = def_type_ndims(def_type)
        self._ndims = ndims

        is_2D = def_type in (DefType.PLANE_STRAIN, DefType.PLANE_STRESS)

        if def_type == DefType.FULL_3D:
            num_residuals = 2

        elif def_type == DefType.PLANE_STRAIN \
                or def_type == DefType.UNIAXIAL_STRESS:
            num_residuals = 3

        elif def_type == DefType.PLANE_STRESS:
            num_residuals = 4

        else:
            raise NotImplementedError

        self._init_residuals(num_residuals)

        # linearized plastic strain tensor in material coordinates
        self.var_names[0] = "plastic strain"
        self.resid_names[0] = "flow rule"
        if is_2D:
            self._var_types[0] = VarType.SYM_TENSOR
            self._num_eqs[0] = get_num_eqs(VarType.SYM_TENSOR, 2)
        elif def_type == DefType.UNIAXIAL_STRESS \
                and not has_material_rotation:
            self._var_types[0] = VarType.VECTOR
            self._num_eqs[0] = get_num_eqs(VarType.VECTOR, 3)
        else:
            self._var_types[0] = VarType.SYM_TENSOR
            self._num_eqs[0] = get_num_eqs(VarType.SYM_TENSOR, 3)
        init_vec_pstrain = np.zeros(self._num_eqs[0])

        # isotropic hardening variable
        self.var_names[1] = "alpha"
        self.resid_names[1] = "yield surface"
        self._var_types[1] = VarType.SCALAR
        self._num_eqs[1] = get_num_eqs(VarType.SCALAR, ndims)
        init_alpha = np.zeros(self._num_eqs[1])

        self._init_xi = [init_vec_pstrain, init_alpha]

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
            self._uniaxial_stress_idx = uniaxial_stress_idx

            self._init_xi += [init_off_axis_stretches]

        if is_2D:
            idx = plastic_strain_33_idx(def_type)
            self.var_names[idx] = "plastic strain 33"
            self.resid_names[idx] = "flow rule 33"
            self._var_types[idx] = VarType.SCALAR
            self._num_eqs[idx] = get_num_eqs(VarType.SCALAR, ndims)
            init_pstrain_33 = np.zeros(self._num_eqs[idx])

            self._init_xi += [init_pstrain_33]

        # set the initial values for xi and xi_prev
        self._init_state_variables()
        self.set_xi_to_init_vals()

        # TODO: check that the parameters make sense for this model
        # self._check_params(parameters)
        self.parameters = parameters
        self._init_scale_factors()

        plastic_subtree = cast(dict[str, Any], parameters.values["plastic"])
        if effective_stress_fun is None:
            effective_stress_type = \
                next(iter(plastic_subtree["effective stress"]))
            effective_stress_fun = \
                conventional_effective_stress_fun(effective_stress_type)
        yield_function = make_yield_function(
            plastic_subtree["flow stress"], hardening_funs)
        yield_threshold = compute_yield_threshold(
            yield_tol, parameters.values, yield_function,
            self.shear_scale_factor)

        residual = partial(self._residual_fn,
                           def_type=def_type,
                           elastic_stress=elastic_stress_fun,
                           effective_stress=effective_stress_fun,
                           yield_function=yield_function,
                           shear_scale_factor=self.shear_scale_factor,
                           yield_threshold=yield_threshold,
                           uniaxial_stress_idx=uniaxial_stress_idx,
                           is_complex=is_complex,
                           has_material_rotation=has_material_rotation)

        cauchy = partial(self._cauchy_fn,
                         def_type=def_type,
                         elastic_stress=elastic_stress_fun,
                         uniaxial_stress_idx=uniaxial_stress_idx,
                         has_material_rotation=has_material_rotation)

        # The elastic predictor is the previous state.
        if initial_guess == "radial return":
            self.initial_guess_fn = jit(partial(
                start_from_radial_return, def_type=def_type,
                elastic_stress=elastic_stress_fun,
                effective_stress=effective_stress_fun,
                yield_function=yield_function,
                shear_scale_factor=self.shear_scale_factor,
                yield_threshold=yield_threshold,
                uniaxial_stress_idx=uniaxial_stress_idx,
                has_material_rotation=has_material_rotation))

        super().__init__(residual, cauchy)

    @classmethod
    def from_deck(
            cls,
            model_section: dict[str, Any],
            parameters: Parameters,
            def_type: int | None,
    ) -> "SmallElasticPlastic":
        return cls(
            parameters=parameters,
            def_type=require_def_type(def_type, cls.__name__),
            uniaxial_stress_idx=model_section.get("uniaxial_stress_idx", 0),
            initial_guess=model_section.get("initial guess"),
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
            shear_scale_factor: float, yield_threshold: float,
            uniaxial_stress_idx: int, is_complex: bool,
            has_material_rotation: bool,
    ) -> JaxArray:

        # state variables for the model
        pstrain = plastic_strain_from_state(
            xi, def_type, has_material_rotation)
        pstrain_prev = plastic_strain_from_state(
            xi_prev, def_type, has_material_rotation)
        alpha = get_scalar(xi[1])
        alpha_prev = get_scalar(xi_prev[1])

        # intermediate quantities
        delta_gamma = alpha - alpha_prev
        material_cauchy, yield_fun, yield_normal = compute_yield_fun_and_normal(
            xi, xi_prev, params, U, U_prev, step_time, def_type,
            elastic_stress, effective_stress, yield_function,
            shear_scale_factor, uniaxial_stress_idx, is_complex,
            has_material_rotation)

        # the elastic predictor freezes the plastic strain and the hardening
        # at the previous step; the stretch blocks stay current because
        # compute_elastic_strain reads them
        xi_elastic = [xi_prev[0], xi_prev[1], *xi[2:]]
        if def_type == DefType.PLANE_STRAIN \
                or def_type == DefType.PLANE_STRESS:
            idx = plastic_strain_33_idx(def_type)
            xi_elastic[idx] = xi_prev[idx]
        _cauchy, trial_yield_fun = compute_yield_fun(
            xi_elastic, xi_prev, params, U, step_time, def_type,
            elastic_stress, effective_stress, yield_function,
            shear_scale_factor, uniaxial_stress_idx, has_material_rotation)

        # elastic residual
        C_elastic_pstrain_tensor = pstrain - pstrain_prev
        C_elastic_pstrain = stored_plastic_strain_components(
            C_elastic_pstrain_tensor, def_type, has_material_rotation)
        C_elastic_alpha = delta_gamma

        # plastic residual
        C_plastic_pstrain_tensor = C_elastic_pstrain_tensor \
            - delta_gamma * yield_normal
        C_plastic_pstrain = stored_plastic_strain_components(
            C_plastic_pstrain_tensor, def_type, has_material_rotation)
        C_plastic_alpha = yield_fun

        if def_type == DefType.FULL_3D:
            C_elastic = jnp.r_[C_elastic_pstrain, C_elastic_alpha]
            C_plastic = jnp.r_[C_plastic_pstrain, C_plastic_alpha]

        elif def_type == DefType.PLANE_STRAIN:
            C_elastic = jnp.r_[C_elastic_pstrain, C_elastic_alpha,
                               C_elastic_pstrain_tensor[2, 2]]
            C_plastic = jnp.r_[C_plastic_pstrain, C_plastic_alpha,
                               C_plastic_pstrain_tensor[2, 2]]

        elif def_type == DefType.PLANE_STRESS or \
                def_type == DefType.UNIAXIAL_STRESS:

            global_cauchy = rotate_out_of_material_frame(
                material_cauchy, params, has_material_rotation)

            if def_type == DefType.PLANE_STRESS:
                C_stretch = global_cauchy[2, 2] / shear_scale_factor
                C_elastic = jnp.r_[C_elastic_pstrain, C_elastic_alpha,
                                   C_stretch, C_elastic_pstrain_tensor[2, 2]]
                C_plastic = jnp.r_[C_plastic_pstrain, C_plastic_alpha,
                                   C_stretch, C_plastic_pstrain_tensor[2, 2]]

            elif def_type == DefType.UNIAXIAL_STRESS:
                off_axis_stress_idx = off_axis_idx(uniaxial_stress_idx)
                first_idx = off_axis_stress_idx[0]
                second_idx = off_axis_stress_idx[1]
                C_stretch = jnp.r_[global_cauchy[first_idx, first_idx],
                                   global_cauchy[second_idx, second_idx]] \
                            / shear_scale_factor
                C_elastic = jnp.r_[C_elastic_pstrain, C_elastic_alpha,
                                   C_stretch]
                C_plastic = jnp.r_[C_plastic_pstrain, C_plastic_alpha,
                                   C_stretch]

        return cond_residual(
            trial_yield_fun, C_elastic, C_plastic, yield_threshold)

    def _check_params(self, parameters: Parameters) -> None:
        raise NotImplementedError

    @staticmethod
    def _cauchy_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            def_type: int, elastic_stress: Callable[..., JaxArray],
            uniaxial_stress_idx: int, has_material_rotation: bool,
    ) -> JaxArray:

        elastic_strain = compute_elastic_strain(xi, params, U, def_type,
            uniaxial_stress_idx, has_material_rotation)
        material_cauchy = elastic_stress(elastic_strain, params)

        return rotate_out_of_material_frame(
            material_cauchy, params, has_material_rotation)
