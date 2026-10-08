"""Finite deformation elastic-plastic model (be_bar return mapping).

Multiplicative split with a neohookean elastic response. The elastic left
Cauchy-Green ``be_bar`` is carried (split into its deviator ``zeta`` and a
hydrostatic part ``Ie``), advanced by the relative deformation gradient,
and returned to the yield surface. The yield is von Mises (J2) on the
deviatoric Kirchhoff stress, which is what the be_bar formulation
supports; the hardening is modular. Runs in FULL_3D, in either 2D form,
or in uniaxial stress.

For plane strain the relative deformation gradient embeds ``F_33 = 1``,
so the 3D return map carries the out of plane ``be_bar`` with no extra
local unknown. Plane stress instead solves for ``F_33`` as a fourth
local unknown, fixed by ``sigma_33 = 0``, which the return map cannot
supply on its own. Uniaxial stress is the same trade one dimension
lower: only the stretch along the loading axis is prescribed, so the two
off-axis stretches are a fourth local unknown, a vector fixed by the two
off-axis normal stresses vanishing. The uniaxial ``gather_F`` is
diagonal, so ``be_bar`` stays diagonal and fits the 2D deviator.
"""
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
from jax import grad, jit

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.effective_stress import J2_effective_stress
from cmad.models.elastic_constants import ElasticConstants
from cmad.models.flow_stress import make_yield_function
from cmad.models.global_fields import (
    GlobalFieldsAtPoint,
    StepTime,
    temperature_at_point,
)
from cmad.models.kinematics import det_3x3, gather_F, inv_3x3, off_axis_idx
from cmad.models.mechanics_model import MechanicsModel, require_def_type
from cmad.models.paths import compute_yield_threshold, cond_residual
from cmad.models.radial_return import (
    plastic_multiplier_increment,
    resolve_initial_guess,
)
from cmad.models.temperature_dependent_parameters import (
    DEFAULT_REFERENCE_TEMPERATURE,
)
from cmad.models.var_types import (
    VarType,
    get_dev_sym_tensor_from_vector,
    get_num_eqs,
    get_scalar,
    get_vector_from_dev_sym_tensor,
)
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray, Scalar, StateBlock, StateList

_NUM_RETURN_SWEEPS = 2
_NUM_IE_STEPS = 2


def zeta_ndims(def_type: int) -> int:
    return 3 if def_type == DefType.FULL_3D else 2


def relative_be_bar(
        zeta_prev: StateBlock, Ie_prev: StateBlock,
        F: JaxArray, F_prev: JaxArray, def_type: int,
) -> JaxArray:
    """Trial ``be_bar`` advanced from the previous step.

    ``be_bar_prev = zeta_prev + Ie_prev * I`` is pushed forward by the
    isochoric part of the relative deformation gradient
    ``rF_bar = rF / det(rF)^(1/3)``, ``rF = F @ F_prev^{-1}``.
    """
    eye = jnp.eye(3)
    be_bar_prev = get_dev_sym_tensor_from_vector(
        zeta_prev, zeta_ndims(def_type)) + Ie_prev * eye
    rF = F @ inv_3x3(F_prev)
    rF_bar = rF / jnp.cbrt(det_3x3(rF))
    return rF_bar @ be_bar_prev @ rF_bar.T


def elastic_predictor(
        xi: StateList, xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
        def_type: int, oop_stretch_idx: int, uniaxial_stress_idx: int,
) -> StateList:
    """Elastic predictor state ``[dev(be_bar_trial), tr(be_bar_trial)/3,
    alpha_prev]``, plus the stretch unknowns under plane or uniaxial
    stress.

    The closed form root of the elastic branch: plastic flow frozen, the
    elastic ``be_bar`` advanced by the relative deformation.

    ``xi`` supplies only the current stretch unknowns, the ``F_33``
    (plane stress) or the two off-axis stretches (uniaxial stress) that
    :func:`gather_F` embeds, so the trial is a function of local unknowns
    there. Everything advected comes from ``xi_prev``: the previous
    ``be_bar``, the hardening, and the previous deformation. In FULL_3D
    and plane strain ``xi`` never reaches the deformation gradient and
    the two coincide.
    """
    F = gather_F(xi, U, def_type, oop_stretch_idx, uniaxial_stress_idx)
    F_prev = gather_F(
        xi_prev, U_prev, def_type, oop_stretch_idx, uniaxial_stress_idx)
    be_bar_trial = relative_be_bar(
        xi_prev[0], xi_prev[1], F, F_prev, def_type)
    dev_be_bar_trial = be_bar_trial - jnp.trace(be_bar_trial) / 3. * jnp.eye(3)
    trial = [
        get_vector_from_dev_sym_tensor(dev_be_bar_trial, zeta_ndims(def_type)),
        jnp.atleast_1d(jnp.trace(be_bar_trial) / 3.),
        xi_prev[2],
    ]
    if def_type in (DefType.PLANE_STRESS, DefType.UNIAXIAL_STRESS):
        trial.append(xi[oop_stretch_idx])
    return trial


def start_from_elastic_predictor(
        xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
        step_time: StepTime,
        def_type: int, oop_stretch_idx: int, uniaxial_stress_idx: int,
) -> StateList:
    """Starting state for the local Newton: the elastic predictor taken at
    the previous stretch unknowns.

    The current stretches do not exist yet, so under plane or uniaxial
    stress this sits off the root of the stress free equations and costs
    iterations.
    What it does hold is ``delta_gamma`` at zero with the deviator on the
    frozen flow manifold, which is what keeps the return map away from its
    other root, the one on the yield surface with a negative plastic
    increment.

    ``step_time`` is unused; this model has no rate dependence. It is in the
    signature because ``make_newton_solve`` calls the initial guess with the
    residual's trailing arguments.
    """
    return elastic_predictor(
        xi_prev, xi_prev, params, U, U_prev, def_type, oop_stretch_idx,
        uniaxial_stress_idx)


def compute_cauchy(
        xi: StateList, params: dict[str, Any], U: GlobalFieldsAtPoint,
        def_type: int, oop_stretch_idx: int, uniaxial_stress_idx: int,
        thermal_stretch: Scalar,
) -> JaxArray:
    """The Cauchy stress on the elastic volume ratio ``J_e = J /
    thermal_stretch^3``; the deviator ``zeta`` is isochoric and carries no
    thermal part."""
    elastic = ElasticConstants.from_params(params["elastic"])
    I = jnp.eye(3)
    F = gather_F(xi, U, def_type, oop_stretch_idx, uniaxial_stress_idx)
    J_e = det_3x3(F) / thermal_stretch ** 3
    zeta = get_dev_sym_tensor_from_vector(xi[0], zeta_ndims(def_type))
    dev_cauchy = elastic.mu * zeta / J_e
    hydro_cauchy = 0.5 * elastic.kappa * (J_e - 1. / J_e)
    return dev_cauchy + hydro_cauchy * I


def compute_yield_fun(
        zeta: JaxArray, alpha: StateBlock, alpha_prev: StateBlock,
        params: dict[str, Any], U: GlobalFieldsAtPoint, step_time: StepTime,
        yield_function: Callable[..., JaxArray], shear_scale_factor: float,
        reference_temperature: float,
) -> JaxArray:
    """Von Mises yield function on the Kirchhoff stress.

    ``zeta`` is the deviator of ``be_bar`` as a tensor; the deviatoric
    Kirchhoff stress is ``s = mu * zeta``, and the J2 effective stress is
    evaluated on it.
    """
    plastic_params = params["plastic"]
    mu = ElasticConstants.from_params(params["elastic"]).mu

    s = mu * zeta
    phi = J2_effective_stress(s, None)
    alpha_dot = (alpha - alpha_prev) / step_time.dt
    yield_fun = yield_function(
        phi, alpha, alpha_dot, temperature_at_point(U, reference_temperature),
        plastic_params["flow stress"])

    return yield_fun / shear_scale_factor


def compute_yield_fun_and_normal(
        zeta: JaxArray, alpha: StateBlock, alpha_prev: StateBlock,
        params: dict[str, Any], U: GlobalFieldsAtPoint, step_time: StepTime,
        yield_function: Callable[..., JaxArray], shear_scale_factor: float,
        is_complex: bool, reference_temperature: float,
) -> tuple[JaxArray, JaxArray]:
    """Yield function and flow normal, the gradient of the J2 effective
    stress at the deviatoric Kirchhoff stress.
    """
    mu = ElasticConstants.from_params(params["elastic"]).mu
    s = mu * zeta
    yield_normal = grad(J2_effective_stress, holomorphic=is_complex)(s, None)

    return compute_yield_fun(
        zeta, alpha, alpha_prev, params, U, step_time, yield_function,
        shear_scale_factor, reference_temperature,
    ), yield_normal


def start_from_radial_return(
        xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
        step_time: StepTime,
        def_type: int, oop_stretch_idx: int, uniaxial_stress_idx: int,
        yield_function: Callable[..., JaxArray],
        shear_scale_factor: float, yield_threshold: float,
        resolve_parameters: Callable[..., dict[str, Any]],
        reference_temperature: float,
) -> StateList:
    """Starting state for the local Newton: the elastic predictor, its
    deviator returned to the yield surface along its own normal when it
    lies outside, with ``Ie`` corrected from ``det(be_bar) = 1``. The
    return is swept again so that it uses a corrected ``Ie``."""
    params = resolve_parameters(params, U)
    trial = elastic_predictor(
        xi_prev, xi_prev, params, U, U_prev, def_type, oop_stretch_idx,
        uniaxial_stress_idx)
    ndims = zeta_ndims(def_type)
    zeta_trial = get_dev_sym_tensor_from_vector(trial[0], ndims)
    alpha_prev = get_scalar(xi_prev[2])[0]
    mu = ElasticConstants.from_params(params["elastic"]).mu
    normal_trial = grad(J2_effective_stress)(mu * zeta_trial, None)
    eye = jnp.eye(3)

    def returned_zeta(delta_gamma: JaxArray, Ie: JaxArray) -> JaxArray:
        return zeta_trial - 2. * delta_gamma * Ie * normal_trial

    def g(delta_gamma: JaxArray, Ie: JaxArray) -> JaxArray:
        return compute_yield_fun(
            returned_zeta(delta_gamma, Ie), alpha_prev + delta_gamma,
            alpha_prev, params, U, step_time, yield_function,
            shear_scale_factor, reference_temperature)

    def det_equation(Ie: JaxArray, zeta: JaxArray) -> JaxArray:
        return det_3x3(zeta + Ie * eye) - 1.

    Ie = get_scalar(trial[1])[0]
    is_plastic = g(jnp.zeros(()), Ie) > yield_threshold

    for _ in range(_NUM_RETURN_SWEEPS):
        delta_gamma = plastic_multiplier_increment(
            partial(g, Ie=Ie), yield_threshold)
        zeta = returned_zeta(delta_gamma, Ie)
        for _ in range(_NUM_IE_STEPS):
            Ie = Ie - det_equation(Ie, zeta) / grad(det_equation)(Ie, zeta)

    returned = [
        get_vector_from_dev_sym_tensor(zeta, ndims),
        jnp.array([Ie]),
        xi_prev[2] + delta_gamma,
    ]

    return [
        jnp.where(is_plastic, returned_block, trial_block)
        for returned_block, trial_block in zip(returned, trial, strict=True)
    ]


class BeBarElasticPlastic(MechanicsModel):
    """Finite deformation elastic-plastic model via the be_bar return map.

    Elastic: neohookean. Plastic: J2 yield on the Kirchhoff stress +
    modular hardening. State ``[zeta (deviatoric be_bar), Ie (hydrostatic
    be_bar), alpha]``, plus the stretches the deformation leaves free under
    plane stress or uniaxial stress.
    """

    supports_mixed = True
    is_finite_deformation = True

    _def_type: int
    _ndims: int
    _uniaxial_stress_idx: int

    def __init__(
            self, parameters: Parameters,
            def_type: int = DefType.FULL_3D,
            hardening_funs: dict | None = None,
            yield_tol: float = 1e-12,
            uniaxial_stress_idx: int = 0,
            is_complex: bool = False,
            initial_guess: str | None = None,
            reference_temperature: float = DEFAULT_REFERENCE_TEMPERATURE,
    ) -> None:

        if def_type not in (
                DefType.FULL_3D, DefType.PLANE_STRAIN, DefType.PLANE_STRESS,
                DefType.UNIAXIAL_STRESS):
            raise NotImplementedError(
                "be_bar_elastic_plastic supports FULL_3D, PLANE_STRAIN, "
                "PLANE_STRESS and UNIAXIAL_STRESS",
            )
        initial_guess = resolve_initial_guess(
            initial_guess, def_type, "be_bar_elastic_plastic")

        self._is_complex = is_complex
        self.dtype = complex if is_complex else float

        self._def_type = def_type
        self._ndims = def_type_ndims(def_type)

        if def_type == DefType.FULL_3D or def_type == DefType.PLANE_STRAIN:
            num_residuals = 3
        else:
            num_residuals = 4

        self._init_residuals(num_residuals)

        # deviatoric part of the elastic left Cauchy-Green be_bar
        self.var_names[0] = "zeta"
        self.resid_names[0] = "be_bar deviator"
        self._var_types[0] = VarType.DEV_SYM_TENSOR
        self._num_eqs[0] = get_num_eqs(
            VarType.DEV_SYM_TENSOR, zeta_ndims(def_type))

        # hydrostatic part such that be_bar = zeta + Ie * I, det(be_bar) = 1
        self.var_names[1] = "Ie"
        self.resid_names[1] = "be_bar hydrostatic"
        self._var_types[1] = VarType.SCALAR
        self._num_eqs[1] = get_num_eqs(VarType.SCALAR, 3)

        # isotropic hardening variable
        self.var_names[2] = "alpha"
        self.resid_names[2] = "yield surface"
        self._var_types[2] = VarType.SCALAR
        self._num_eqs[2] = get_num_eqs(VarType.SCALAR, 3)

        self._init_xi = [
            np.zeros(self._num_eqs[0]),
            np.ones(self._num_eqs[1]),
            np.zeros(self._num_eqs[2]),
        ]

        if def_type == DefType.PLANE_STRESS:
            # Out of plane stretch, the F_33 that gather_F embeds. Unlike
            # plane strain, where F_33 = 1 embeds directly, plane stress
            # fixes it by the plane stress condition sigma_33 = 0, so it
            # is an unknown.
            self.var_names[3] = "out of plane stretch"
            self.resid_names[3] = "cauchy_33"
            self._var_types[3] = VarType.SCALAR
            self._num_eqs[3] = get_num_eqs(VarType.SCALAR, 3)
            self._oop_stretch_idx = 3

            self._init_xi += [np.ones(self._num_eqs[3])]

        elif def_type == DefType.UNIAXIAL_STRESS:
            # The two stretches off the loading axis, which gather_F
            # embeds beside the prescribed one, fixed by the two off-axis
            # normal stresses vanishing.
            self.var_names[3] = "off-axis stretches"
            self.resid_names[3] = "off-axis normal stress"
            self._var_types[3] = VarType.VECTOR
            self._num_eqs[3] = get_num_eqs(VarType.VECTOR, 2)
            self._oop_stretch_idx = 3
            self._uniaxial_stress_idx = uniaxial_stress_idx

            self._init_xi += [np.ones(self._num_eqs[3])]

        self._init_state_variables()
        self.set_xi_to_init_vals()

        self._init_parameters(parameters, reference_temperature)

        plastic_subtree = cast(dict[str, Any], parameters.values["plastic"])
        yield_function = make_yield_function(
            plastic_subtree["flow stress"], hardening_funs)
        yield_threshold = compute_yield_threshold(
            yield_tol, self.reference_parameters, yield_function,
            self.shear_scale_factor, reference_temperature)

        residual = partial(
            self._residual_fn,
            def_type=def_type, oop_stretch_idx=self._oop_stretch_idx,
            uniaxial_stress_idx=uniaxial_stress_idx,
            yield_function=yield_function,
            shear_scale_factor=self.shear_scale_factor,
            yield_threshold=yield_threshold, is_complex=is_complex,
            resolve_parameters=self.resolve_parameters,
            compute_thermal_stretch=self.compute_thermal_stretch,
            reference_temperature=reference_temperature)

        cauchy = partial(
            self._cauchy_fn,
            def_type=def_type, oop_stretch_idx=self._oop_stretch_idx,
            uniaxial_stress_idx=uniaxial_stress_idx,
            resolve_parameters=self.resolve_parameters,
            compute_thermal_stretch=self.compute_thermal_stretch)

        if "taylor-quinney" in plastic_subtree:
            self.dissipation = partial(
                self._dissipation_fn,
                def_type=def_type, oop_stretch_idx=self._oop_stretch_idx,
                uniaxial_stress_idx=uniaxial_stress_idx,
                resolve_parameters=self.resolve_parameters,
                compute_thermal_stretch=self.compute_thermal_stretch)

        if initial_guess == "radial return":
            self.initial_guess_fn = jit(partial(
                start_from_radial_return,
                def_type=def_type, oop_stretch_idx=self._oop_stretch_idx,
                uniaxial_stress_idx=uniaxial_stress_idx,
                yield_function=yield_function,
                shear_scale_factor=self.shear_scale_factor,
                yield_threshold=yield_threshold,
                resolve_parameters=self.resolve_parameters,
                reference_temperature=reference_temperature))
        else:
            self.initial_guess_fn = jit(partial(
                start_from_elastic_predictor,
                def_type=def_type, oop_stretch_idx=self._oop_stretch_idx,
                uniaxial_stress_idx=uniaxial_stress_idx))

        super().__init__(residual, cauchy)

    @classmethod
    def from_deck(
            cls,
            model_section: dict[str, Any],
            parameters: Parameters,
            def_type: int | None,
    ) -> "BeBarElasticPlastic":
        return cls(
            parameters=parameters,
            def_type=require_def_type(def_type, cls.__name__),
            uniaxial_stress_idx=model_section.get("uniaxial_stress_idx", 0),
            initial_guess=model_section.get("initial guess"),
            reference_temperature=model_section.get(
                "reference temperature", DEFAULT_REFERENCE_TEMPERATURE),
        )

    def derived_output_field_names(self) -> list[str]:
        return ["cauchy"]

    @staticmethod
    def _residual_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            step_time: StepTime,
            def_type: int, oop_stretch_idx: int, uniaxial_stress_idx: int,
            yield_function: Callable[..., JaxArray],
            shear_scale_factor: float, yield_threshold: float,
            is_complex: bool,
            resolve_parameters: Callable[..., dict[str, Any]],
            compute_thermal_stretch: Callable[..., Scalar],
            reference_temperature: float,
    ) -> JaxArray:

        thermal_stretch = compute_thermal_stretch(params, U)
        params = resolve_parameters(params, U)
        ndims = zeta_ndims(def_type)
        zeta = get_dev_sym_tensor_from_vector(xi[0], ndims)
        Ie = get_scalar(xi[1])
        alpha = get_scalar(xi[2])
        alpha_prev = get_scalar(xi_prev[2])

        eye = jnp.eye(3)
        xi_elastic = elastic_predictor(
            xi, xi_prev, params, U, U_prev, def_type, oop_stretch_idx,
            uniaxial_stress_idx)
        dev_be_bar_trial = get_dev_sym_tensor_from_vector(
            xi_elastic[0], ndims)

        yield_fun, yield_normal = compute_yield_fun_and_normal(
            zeta, alpha, alpha_prev, params, U, step_time, yield_function,
            shear_scale_factor, is_complex, reference_temperature)
        trial_yield_fun = compute_yield_fun(
            dev_be_bar_trial, alpha_prev, alpha_prev, params, U, step_time,
            yield_function, shear_scale_factor, reference_temperature)
        delta_gamma = alpha - alpha_prev

        C_elastic = jnp.concatenate(
            [xi[i] - xi_elastic[i] for i in range(3)])

        # plastic return map
        C_zeta_plastic = get_vector_from_dev_sym_tensor(
            zeta - dev_be_bar_trial + 2. * delta_gamma * Ie * yield_normal,
            ndims)
        C_Ie_plastic = det_3x3(zeta + Ie * eye) - 1.
        C_plastic = jnp.r_[C_zeta_plastic, C_Ie_plastic, yield_fun]

        if def_type in (DefType.PLANE_STRESS, DefType.UNIAXIAL_STRESS):
            # The stretch unknowns are fixed by the undriven normal
            # stresses vanishing, which holds whether or not the step
            # yields, so those equations close both branches.
            cauchy = compute_cauchy(
                xi, params, U, def_type, oop_stretch_idx,
                uniaxial_stress_idx, thermal_stretch)
            if def_type == DefType.PLANE_STRESS:
                C_stretch = jnp.atleast_1d(cauchy[2, 2] / shear_scale_factor)
            else:
                off_axis_stress_idx = off_axis_idx(uniaxial_stress_idx)
                first_idx = off_axis_stress_idx[0]
                second_idx = off_axis_stress_idx[1]
                C_stretch = jnp.r_[cauchy[first_idx, first_idx],
                                   cauchy[second_idx, second_idx]] \
                    / shear_scale_factor
            C_elastic = jnp.r_[C_elastic, C_stretch]
            C_plastic = jnp.r_[C_plastic, C_stretch]

        return cond_residual(
            trial_yield_fun, C_elastic, C_plastic, yield_threshold)

    @staticmethod
    def _cauchy_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            def_type: int, oop_stretch_idx: int, uniaxial_stress_idx: int,
            resolve_parameters: Callable[..., dict[str, Any]],
            compute_thermal_stretch: Callable[..., Scalar],
    ) -> JaxArray:
        thermal_stretch = compute_thermal_stretch(params, U)
        params = resolve_parameters(params, U)
        return compute_cauchy(
            xi, params, U, def_type, oop_stretch_idx, uniaxial_stress_idx,
            thermal_stretch)

    @staticmethod
    def _dissipation_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            step_time: StepTime,
            def_type: int, oop_stretch_idx: int, uniaxial_stress_idx: int,
            resolve_parameters: Callable[..., dict[str, Any]],
            compute_thermal_stretch: Callable[..., Scalar],
    ) -> Scalar:
        """The plastic work rate that becomes heat per unit reference
        volume, ``beta J sigma : d eps_p / dt``."""
        thermal_stretch = compute_thermal_stretch(params, U)
        params = resolve_parameters(params, U)
        cauchy = compute_cauchy(
            xi, params, U, def_type, oop_stretch_idx, uniaxial_stress_idx,
            thermal_stretch)
        J = det_3x3(gather_F(
            xi, U, def_type, oop_stretch_idx, uniaxial_stress_idx))
        ndims = zeta_ndims(def_type)
        zeta = get_dev_sym_tensor_from_vector(xi[0], ndims)
        Ie = get_scalar(xi[1])[0]
        trial = elastic_predictor(
            xi, xi_prev, params, U, U_prev, def_type, oop_stretch_idx,
            uniaxial_stress_idx)
        dev_be_bar_trial = get_dev_sym_tensor_from_vector(trial[0], ndims)
        plastic_increment = (dev_be_bar_trial - zeta) / (2. * Ie)
        beta = params["plastic"]["taylor-quinney"]
        return beta * J * jnp.sum(cauchy * plastic_increment) / step_time.dt
