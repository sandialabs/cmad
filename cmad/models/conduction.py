"""Heat conduction: Fourier's flux and the heat capacity."""
from typing import Any, ClassVar

from jax import numpy as jnp

from cmad.models.global_fields import GlobalFieldsAtPoint, StepTime
from cmad.models.thermal_model import ThermalModel
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray, Scalar, StateList


def isotropic_heat_flux(grad_T: JaxArray, params: dict[str, Any]) -> JaxArray:
    """Fourier's flux of an isotropic conductor, ``-k grad T``."""
    return -params["thermal"]["conductivity"] * grad_T


class Conduction(ThermalModel):
    """Heat conduction from Fourier's law, with the heat capacity ``rho c``
    when the material names a density and a specific heat.

    No local state, so the flux is closed-form; without the capacity the
    rate term is zero and the problem is steady.
    """

    supports_closed_form: ClassVar[bool] = True

    def __init__(self, parameters: Parameters) -> None:
        self._is_complex = False
        self.dtype = float
        self._ndims = 3

        thermal = parameters.values["thermal"]
        assert isinstance(thermal, dict)
        has_density = "density" in thermal
        has_specific_heat = "specific heat" in thermal
        if has_density != has_specific_heat:
            raise ValueError(
                "thermal: density and specific heat are given together or "
                "not at all",
            )
        self._has_capacity = has_density

        self._init_residuals(0)
        self._init_xi = []
        self._init_state_variables()
        self.set_xi_to_init_vals()
        self.parameters = parameters

        super().__init__(self._residual_fn)

    @classmethod
    def from_deck(
            cls,
            model_section: dict[str, Any],
            parameters: Parameters,
            def_type: int | None,
    ) -> "Conduction":
        return cls(parameters=parameters)

    def derived_output_field_names(self) -> list[str]:
        return ["heat flux"]

    @staticmethod
    def _residual_fn(
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            step_time: StepTime,
    ) -> JaxArray:
        return jnp.zeros(0)

    @staticmethod
    def heat_flux_closed_form(
            params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> JaxArray:
        return isotropic_heat_flux(U.grad_fields["T"][0], params)

    def heat_flux(
            self,
            xi: StateList, xi_prev: StateList, params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> JaxArray:
        return isotropic_heat_flux(U.grad_fields["T"][0], params)

    def heat_capacity_rate(
            self,
            params: dict[str, Any],
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            step_time: StepTime,
    ) -> Scalar:
        if not self._has_capacity:
            return 0.0
        thermal = params["thermal"]
        rho_c = thermal["density"] * thermal["specific heat"]
        return rho_c * (U.fields["T"][0] - U_prev.fields["T"][0]) / step_time.dt
