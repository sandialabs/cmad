"""Finite deformation elastic-plastic model (be_bar return mapping).

Multiplicative split with a neohookean elastic response. The elastic left
Cauchy-Green ``be_bar`` is carried (split into its deviator ``zeta`` and a
hydrostatic part ``Ie``), advanced by the relative deformation gradient,
and returned to the yield surface. The yield is von Mises (J2) on the
deviatoric Kirchhoff stress, which is what the be_bar formulation
supports; the hardening is modular. Runs in FULL_3D or 2D plane strain;
for plane strain the relative deformation gradient embeds ``F_33 = 1``,
so the 3D return map carries the out of plane ``be_bar`` with no extra
local unknown.
"""
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar

import jax.numpy as jnp
import numpy as np
from jax import grad

from cmad.io.registry import register_model
from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.effective_stress import J2_effective_stress
from cmad.models.elastic_constants import ElasticConstants
from cmad.models.elastic_stress import two_mu_scale_factor
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.models.hardening import combined_hardening_fun, get_hardening_funs
from cmad.models.kinematics import gather_F
from cmad.models.mechanics_model import MechanicsModel
from cmad.models.paths import cond_residual
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
    rF = F @ jnp.linalg.inv(F_prev)
    rF_bar = rF / jnp.cbrt(jnp.linalg.det(rF))
    return rF_bar @ be_bar_prev @ rF_bar.T


def compute_yield_fun_and_normal(
        zeta: StateBlock, alpha: StateBlock, params: dict[str, Any],
        hardening: Callable[..., JaxArray], is_complex: bool,
) -> tuple[JaxArray, JaxArray]:
    """Von Mises yield function and flow normal on the Kirchhoff stress.

    The deviatoric Kirchhoff stress is ``s = mu * zeta``; the J2 effective
    stress and its gradient (the flow normal) are evaluated on it, and the
    flow stress is the modular hardening.
    """
    plastic_params = params["plastic"]
    Y = plastic_params["flow stress"]["initial yield"]["Y"]
    hardening_params = plastic_params["flow stress"]["hardening"]
    mu = ElasticConstants.from_params(params["elastic"]).mu

    s = mu * get_sym_tensor_from_vector(zeta, 3)
    phi = J2_effective_stress(s, None)
    sigma_flow = Y + hardening(alpha, hardening_params)

    yield_fun = (phi - sigma_flow) / two_mu_scale_factor(params)
    yield_normal = grad(J2_effective_stress, holomorphic=is_complex)(s, None)

    return yield_fun, yield_normal


@register_model("be_bar_elastic_plastic")
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
            yield_tol: float = 1e-14,
            is_complex: bool = False,
    ) -> None:

        if def_type not in (DefType.FULL_3D, DefType.PLANE_STRAIN):
            raise NotImplementedError(
                "be_bar_elastic_plastic supports FULL_3D and PLANE_STRAIN",
            )
        if hardening_funs is None:
            hardening_funs = get_hardening_funs()

        self._is_complex = is_complex
        self.dtype = complex if is_complex else float

        self._def_type = def_type
        self._ndims = def_type_ndims(def_type)

        self._init_residuals(3)

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

        self._init_state_variables()
        self.set_xi_to_init_vals()

        self.parameters = parameters

        residual = partial(
            self._residual_fn,
            def_type=def_type,
            hardening=partial(
                combined_hardening_fun, hardening_funs=hardening_funs),
            yield_tol=yield_tol, is_complex=is_complex)

        cauchy = partial(self._cauchy_fn, def_type=def_type)

        super().__init__(residual, cauchy)

    @classmethod
    def from_deck(
            cls,
            model_section: dict[str, Any],
            parameters: Parameters,
            def_type: int,
    ) -> "BeBarElasticPlastic":
        return cls(parameters=parameters, def_type=def_type)

    def derived_output_field_names(self) -> list[str]:
        return ["cauchy"]

    @staticmethod
    def _residual_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            def_type: int,
            hardening: Callable[..., JaxArray],
            yield_tol: float, is_complex: bool,
    ) -> JaxArray:

        zeta = get_sym_tensor_from_vector(xi[0], 3)
        Ie = get_scalar(xi[1])
        alpha = get_scalar(xi[2])
        alpha_prev = get_scalar(xi_prev[2])

        eye = jnp.eye(3)
        F = gather_F(xi, U, def_type, local_var_idx=0)
        F_prev = gather_F(xi_prev, U_prev, def_type, local_var_idx=0)
        be_bar_trial = relative_be_bar(xi_prev[0], xi_prev[1], F, F_prev)
        dev_be_bar_trial = be_bar_trial - jnp.trace(be_bar_trial) / 3. * eye

        yield_fun, yield_normal = compute_yield_fun_and_normal(
            xi[0], alpha, params, hardening, is_complex)
        delta_gamma = alpha - alpha_prev

        # elastic trial state
        C_zeta_elastic = get_vector_from_sym_tensor(zeta - dev_be_bar_trial, 3)
        C_Ie_elastic = Ie - jnp.trace(be_bar_trial) / 3.
        C_elastic = jnp.r_[C_zeta_elastic, C_Ie_elastic, delta_gamma]

        # plastic return map
        C_zeta_plastic = get_vector_from_sym_tensor(
            zeta - dev_be_bar_trial + 2. * delta_gamma * Ie * yield_normal, 3)
        C_Ie_plastic = jnp.linalg.det(zeta + Ie * eye) - 1.
        C_plastic = jnp.r_[C_zeta_plastic, C_Ie_plastic, yield_fun]

        return cond_residual(yield_fun, C_elastic, C_plastic, yield_tol)

    @staticmethod
    def _cauchy_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            def_type: int,
    ) -> JaxArray:
        elastic = ElasticConstants.from_params(params["elastic"])
        eye = jnp.eye(3)
        F = gather_F(xi, U, def_type, local_var_idx=0)
        J = jnp.linalg.det(F)
        zeta = get_sym_tensor_from_vector(xi[0], 3)
        dev_cauchy = elastic.mu * zeta / J
        hydro_cauchy = 0.5 * elastic.kappa * (J - 1. / J)
        return dev_cauchy + hydro_cauchy * eye

    def dev_cauchy(
            self,
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> JaxArray:
        mu = ElasticConstants.from_params(params["elastic"]).mu
        F = gather_F(xi, U, self._def_type, local_var_idx=0)
        J = jnp.linalg.det(F)
        zeta = get_sym_tensor_from_vector(xi[0], 3)
        return mu * zeta / J

    def hydro_cauchy(
            self,
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> Scalar:
        kappa = ElasticConstants.from_params(params["elastic"]).kappa
        F = gather_F(xi, U, self._def_type, local_var_idx=0)
        J = jnp.linalg.det(F)
        return 0.5 * kappa * (J - 1. / J)

    @staticmethod
    def pressure_scale_factor(params: dict[str, Any]) -> Scalar:
        return ElasticConstants.from_params(params["elastic"]).kappa

    @staticmethod
    def shear_scale_factor(params: dict[str, Any]) -> Scalar:
        return ElasticConstants.from_params(params["elastic"]).mu
