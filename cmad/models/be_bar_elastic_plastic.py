"""Finite deformation elastic-plastic model (be_bar return mapping).

Multiplicative split with a neohookean elastic response. The elastic left
Cauchy-Green ``be_bar`` is carried (split into its deviator ``zeta`` and a
hydrostatic part ``Ie``), advanced by the relative deformation gradient,
and returned to the yield surface. The yield is von Mises (J2) on the
deviatoric Kirchhoff stress, which is what the be_bar formulation
supports; the hardening is modular. Runs in FULL_3D or in either 2D form.

For plane strain the relative deformation gradient embeds ``F_33 = 1``,
so the 3D return map carries the out of plane ``be_bar`` with no extra
local unknown. Plane stress instead solves for ``F_33`` as a fourth
local unknown, fixed by ``sigma_33 = 0``, which the return map cannot
supply on its own.
"""
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar

import jax.numpy as jnp
import numpy as np
from jax import grad, jit

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.effective_stress import J2_effective_stress
from cmad.models.elastic_constants import ElasticConstants
from cmad.models.elastic_stress import two_mu_scale_factor
from cmad.models.global_fields import GlobalFieldsAtPoint, StepTime
from cmad.models.hardening import combined_hardening_fun, get_hardening_funs
from cmad.models.kinematics import det_3x3, gather_F, inv_3x3
from cmad.models.mechanics_model import MechanicsModel, require_def_type
from cmad.models.paths import cond_residual, yield_threshold
from cmad.models.var_types import (
    VarType,
    get_num_eqs,
    get_scalar,
    get_sym_tensor_from_vector,
    get_vector_from_sym_tensor,
)
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray, Scalar, StateBlock, StateList


def relative_be_bar(
        zeta_prev: StateBlock, Ie_prev: StateBlock,
        F: JaxArray, F_prev: JaxArray,
) -> JaxArray:
    """Trial ``be_bar`` advanced from the previous step.

    ``be_bar_prev = zeta_prev + Ie_prev * I`` is pushed forward by the
    isochoric part of the relative deformation gradient
    ``rF_bar = rF / det(rF)^(1/3)``, ``rF = F @ F_prev^{-1}``.
    """
    eye = jnp.eye(3)
    be_bar_prev = get_sym_tensor_from_vector(zeta_prev, 3) + Ie_prev * eye
    rF = F @ inv_3x3(F_prev)
    rF_bar = rF / jnp.cbrt(det_3x3(rF))
    return rF_bar @ be_bar_prev @ rF_bar.T


def elastic_predictor(
        xi: StateList, xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
        def_type: int, oop_stretch_idx: int,
) -> StateList:
    """Elastic predictor state ``[dev(be_bar_trial), tr(be_bar_trial)/3,
    alpha_prev]``, plus the out of plane stretch under plane stress.

    The closed form root of the elastic branch: plastic flow frozen, the
    elastic ``be_bar`` advanced by the relative deformation.

    ``xi`` supplies only the current out of plane stretch, which under
    plane stress is the ``F_33`` that :func:`gather_F` embeds, so the
    trial is a function of a local unknown there. Everything advected
    comes from ``xi_prev``: the previous ``be_bar``, the hardening, and
    the previous deformation. In FULL_3D and plane strain ``xi`` never
    reaches the deformation gradient and the two coincide.
    """
    F = gather_F(xi, U, def_type, oop_stretch_idx)
    F_prev = gather_F(xi_prev, U_prev, def_type, oop_stretch_idx)
    be_bar_trial = relative_be_bar(xi_prev[0], xi_prev[1], F, F_prev)
    dev_be_bar_trial = be_bar_trial - jnp.trace(be_bar_trial) / 3. * jnp.eye(3)
    trial = [
        get_vector_from_sym_tensor(dev_be_bar_trial, 3),
        jnp.atleast_1d(jnp.trace(be_bar_trial) / 3.),
        xi_prev[2],
    ]
    if def_type == DefType.PLANE_STRESS:
        trial.append(xi[oop_stretch_idx])
    return trial


def initial_guess(
        xi_prev: StateList, params: dict[str, Any],
        U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
        step_time: StepTime,
        def_type: int, oop_stretch_idx: int,
) -> StateList:
    """Starting state for the local Newton: the elastic predictor taken at
    the previous out of plane stretch.

    The current stretch does not exist yet, so under plane stress this
    sits off the root of the out of plane equation and costs iterations.
    What it does hold is ``delta_gamma`` at zero with the deviator on the
    frozen flow manifold, which is what keeps the return map away from its
    other root, the one on the yield surface with a negative plastic
    increment.

    ``step_time`` is unused; this model has no rate dependence. It is in the
    signature because ``make_newton_solve`` calls the initial guess with the
    residual's trailing arguments.
    """
    return elastic_predictor(
        xi_prev, xi_prev, params, U, U_prev, def_type, oop_stretch_idx)


def compute_yield_fun(
        zeta: StateBlock, alpha: StateBlock, params: dict[str, Any],
        hardening: Callable[..., JaxArray],
) -> JaxArray:
    """Von Mises yield function on the Kirchhoff stress.

    The deviatoric Kirchhoff stress is ``s = mu * zeta``; the J2 effective
    stress is evaluated on it, and the flow stress is the modular
    hardening.
    """
    plastic_params = params["plastic"]
    Y = plastic_params["flow stress"]["initial yield"]["Y"]
    hardening_params = plastic_params["flow stress"]["hardening"]
    mu = ElasticConstants.from_params(params["elastic"]).mu

    s = mu * get_sym_tensor_from_vector(zeta, 3)
    phi = J2_effective_stress(s, None)
    sigma_flow = Y + hardening(alpha, hardening_params)

    return (phi - sigma_flow) / two_mu_scale_factor(params)


def compute_yield_fun_and_normal(
        zeta: StateBlock, alpha: StateBlock, params: dict[str, Any],
        hardening: Callable[..., JaxArray], is_complex: bool,
) -> tuple[JaxArray, JaxArray]:
    """Yield function and flow normal, the gradient of the J2 effective
    stress at the deviatoric Kirchhoff stress.
    """
    mu = ElasticConstants.from_params(params["elastic"]).mu
    s = mu * get_sym_tensor_from_vector(zeta, 3)
    yield_normal = grad(J2_effective_stress, holomorphic=is_complex)(s, None)

    return compute_yield_fun(zeta, alpha, params, hardening), yield_normal


class BeBarElasticPlastic(MechanicsModel):
    """Finite deformation elastic-plastic model via the be_bar return map.

    Elastic: neohookean. Plastic: J2 yield on the Kirchhoff stress +
    modular hardening. State ``[zeta (deviatoric be_bar), Ie (hydrostatic
    be_bar), alpha]``.
    """

    supports_mixed: ClassVar[bool] = True
    is_finite_deformation = True

    _def_type: int
    _ndims: int

    def __init__(
            self, parameters: Parameters,
            def_type: int = DefType.FULL_3D,
            hardening_funs: dict | None = None,
            yield_tol: float = 1e-12,
            is_complex: bool = False,
    ) -> None:

        if def_type not in (
                DefType.FULL_3D, DefType.PLANE_STRAIN, DefType.PLANE_STRESS):
            raise NotImplementedError(
                "be_bar_elastic_plastic supports FULL_3D, PLANE_STRAIN and "
                "PLANE_STRESS",
            )
        if hardening_funs is None:
            hardening_funs = get_hardening_funs()

        self._is_complex = is_complex
        self.dtype = complex if is_complex else float

        self._def_type = def_type
        self._ndims = def_type_ndims(def_type)

        plane_stress = def_type == DefType.PLANE_STRESS
        self._init_residuals(4 if plane_stress else 3)

        # deviatoric part of the elastic left Cauchy-Green be_bar
        self.var_names[0] = "zeta"
        self.resid_names[0] = "be_bar deviator"
        self._var_types[0] = VarType.SYM_TENSOR
        self._num_eqs[0] = get_num_eqs(VarType.SYM_TENSOR, 3)

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

        if plane_stress:
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

        self._init_state_variables()
        self.set_xi_to_init_vals()

        self.parameters = parameters

        residual = partial(
            self._residual_fn,
            def_type=def_type, oop_stretch_idx=self._oop_stretch_idx,
            hardening=partial(
                combined_hardening_fun, hardening_funs=hardening_funs),
            yield_tol=yield_tol, is_complex=is_complex)

        cauchy = partial(
            self._cauchy_fn,
            def_type=def_type, oop_stretch_idx=self._oop_stretch_idx)

        self.initial_guess_fn = jit(partial(
            initial_guess,
            def_type=def_type, oop_stretch_idx=self._oop_stretch_idx))

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
        )

    def derived_output_field_names(self) -> list[str]:
        return ["cauchy"]

    @staticmethod
    def _residual_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            step_time: StepTime,
            def_type: int, oop_stretch_idx: int,
            hardening: Callable[..., JaxArray],
            yield_tol: float, is_complex: bool,
    ) -> JaxArray:

        zeta = get_sym_tensor_from_vector(xi[0], 3)
        Ie = get_scalar(xi[1])
        alpha = get_scalar(xi[2])
        alpha_prev = get_scalar(xi_prev[2])

        eye = jnp.eye(3)
        xi_elastic = elastic_predictor(
            xi, xi_prev, params, U, U_prev, def_type, oop_stretch_idx)
        dev_be_bar_trial = get_sym_tensor_from_vector(xi_elastic[0], 3)

        yield_fun, yield_normal = compute_yield_fun_and_normal(
            xi[0], alpha, params, hardening, is_complex)
        trial_yield_fun = compute_yield_fun(
            xi_elastic[0], alpha_prev, params, hardening)
        delta_gamma = alpha - alpha_prev

        C_elastic = jnp.concatenate(
            [xi[i] - xi_elastic[i] for i in range(3)])

        # plastic return map
        C_zeta_plastic = get_vector_from_sym_tensor(
            zeta - dev_be_bar_trial + 2. * delta_gamma * Ie * yield_normal, 3)
        C_Ie_plastic = det_3x3(zeta + Ie * eye) - 1.
        C_plastic = jnp.r_[C_zeta_plastic, C_Ie_plastic, yield_fun]

        if def_type == DefType.PLANE_STRESS:
            # The out of plane stretch is fixed by sigma_33 = 0, which holds
            # whether or not the step yields, so it closes both branches.
            cauchy = BeBarElasticPlastic._cauchy_fn(
                xi, xi_prev, params, U, U_prev, def_type, oop_stretch_idx)
            C_oop = jnp.atleast_1d(
                cauchy[2, 2] / two_mu_scale_factor(params))
            C_elastic = jnp.r_[C_elastic, C_oop]
            C_plastic = jnp.r_[C_plastic, C_oop]

        return cond_residual(trial_yield_fun, C_elastic, C_plastic,
                             yield_threshold(yield_tol, params))

    @staticmethod
    def _cauchy_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            def_type: int, oop_stretch_idx: int,
    ) -> JaxArray:
        elastic = ElasticConstants.from_params(params["elastic"])
        eye = jnp.eye(3)
        F = gather_F(xi, U, def_type, oop_stretch_idx)
        J = det_3x3(F)
        zeta = get_sym_tensor_from_vector(xi[0], 3)
        dev_cauchy = elastic.mu * zeta / J
        hydro_cauchy = 0.5 * elastic.kappa * (J - 1. / J)
        return dev_cauchy + hydro_cauchy * eye

    @staticmethod
    def pressure_scale_factor(params: dict[str, Any]) -> Scalar:
        return ElasticConstants.from_params(params["elastic"]).kappa

    @staticmethod
    def shear_scale_factor(params: dict[str, Any]) -> Scalar:
        return ElasticConstants.from_params(params["elastic"]).mu
