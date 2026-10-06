"""A mechanics model and a thermal model on one element block."""
from typing import Any

from jax import numpy as jnp

from cmad.models.global_fields import GlobalFieldsAtPoint, StepTime
from cmad.models.mechanics_model import MechanicsModel
from cmad.models.thermal_model import ThermalModel
from cmad.models.var_types import VarType
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray, Params, Scalar, StateList


class ThermomechanicsModel(MechanicsModel, ThermalModel):
    """A mechanics model and a thermal model on one element block, sharing
    one ``Parameters``.

    The local state stacks the mechanics blocks before the thermal blocks,
    and the residual stacks the same way. The Cauchy stress and the
    deformation gradient come from the mechanics model, the heat flux and
    the heat capacity rate from the thermal model. The input file builder
    composes it for the thermomechanics global residual; it is not an
    input file type.
    """

    def __init__(
            self, mechanics: MechanicsModel, thermal: ThermalModel,
    ) -> None:
        if mechanics.parameters is not thermal.parameters:
            raise ValueError(
                "ThermomechanicsModel: the mechanics model and the thermal "
                "model must share one Parameters")
        self.mechanics = mechanics
        self.thermal = thermal
        self._num_mechanics_blocks = mechanics.num_residuals

        self._is_complex = mechanics._is_complex
        self.dtype = mechanics.dtype
        self._ndims = mechanics.ndims
        self.is_finite_deformation = mechanics.is_finite_deformation
        self._def_type = mechanics._def_type
        self._oop_stretch_idx = mechanics._oop_stretch_idx
        self.supports_closed_form = (
            mechanics.supports_closed_form and thermal.supports_closed_form)
        self.supports_mixed = mechanics.supports_mixed

        self._init_residuals(mechanics.num_residuals + thermal.num_residuals)
        for offset, model in (
                (0, mechanics), (mechanics.num_residuals, thermal)):
            for r in range(model.num_residuals):
                self._num_eqs[offset + r] = model._num_eqs[r]
                self._var_types[offset + r] = model._var_types[r]
                self.resid_names[offset + r] = model.resid_names[r]
                self.var_names[offset + r] = model.var_names[r]
        self._init_xi = [*mechanics._init_xi, *thermal._init_xi]
        self._init_state_variables()
        self.set_xi_to_init_vals()

        self.parameters = mechanics.parameters
        self.resolve_parameters = mechanics.resolve_parameters
        self.reference_parameters = mechanics.reference_parameters
        self.bulk_scale_factor = mechanics.bulk_scale_factor
        self.shear_scale_factor = mechanics.shear_scale_factor
        self.compute_thermal_stretch = mechanics.compute_thermal_stretch
        if thermal.supports_closed_form:
            self.heat_flux_closed_form = thermal.heat_flux_closed_form
        if mechanics.initial_guess_fn is not None:
            self.initial_guess_fn = self._initial_guess_fn

        super().__init__(
            self._residual_fn, self._cauchy_fn,
            cauchy_closed_form_fun=mechanics.cauchy_closed_form)

    @classmethod
    def from_deck(
            cls,
            model_section: dict[str, Any],
            parameters: Parameters,
            def_type: int | None,
    ) -> "ThermomechanicsModel":
        raise ValueError(
            "residuals.local residual.type: thermomechanics_model is not an "
            "input file type; name the mechanics model and use the "
            "thermomechanics global residual")

    def _residual_fn(
            self, xi: StateList, xi_prev: StateList, params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            step_time: StepTime,
    ) -> JaxArray:
        n = self._num_mechanics_blocks
        return jnp.concatenate([
            self.mechanics._residual(
                xi[:n], xi_prev[:n], params, U, U_prev, step_time),
            self.thermal._residual(
                xi[n:], xi_prev[n:], params, U, U_prev, step_time),
        ])

    def _cauchy_fn(
            self, xi: StateList, xi_prev: StateList, params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> JaxArray:
        n = self._num_mechanics_blocks
        return self.mechanics.cauchy(xi[:n], xi_prev[:n], params, U, U_prev)

    def _initial_guess_fn(
            self, xi_prev: StateList, params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            step_time: StepTime,
    ) -> StateList:
        n = self._num_mechanics_blocks
        assert self.mechanics.initial_guess_fn is not None
        return [
            *self.mechanics.initial_guess_fn(
                xi_prev[:n], params, U, U_prev, step_time),
            *xi_prev[n:],
        ]

    def deformation_gradient(
            self, xi: StateList, U: GlobalFieldsAtPoint,
    ) -> JaxArray:
        return self.mechanics.deformation_gradient(
            xi[:self._num_mechanics_blocks], U)

    def heat_flux(
            self,
            xi: StateList, xi_prev: StateList, params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> JaxArray:
        n = self._num_mechanics_blocks
        return self.thermal.heat_flux(xi[n:], xi_prev[n:], params, U, U_prev)

    def heat_capacity_rate(
            self,
            params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            step_time: StepTime,
    ) -> Scalar:
        return self.thermal.heat_capacity_rate(params, U, U_prev, step_time)

    def state_output_fields(self) -> list[tuple[str, VarType]]:
        return [
            *self.mechanics.state_output_fields(),
            *self.thermal.state_output_fields(),
        ]

    def derived_output_field_names(self) -> list[str]:
        return [
            *self.mechanics.derived_output_field_names(),
            *self.thermal.derived_output_field_names(),
        ]
