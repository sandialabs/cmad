"""Base class for constitutive models whose flux is the heat flux.

Separates the thermal contract the heat transfer global residual relies
on from the general :class:`cmad.models.model.Model`, the way
:class:`cmad.models.mechanics_model.MechanicsModel` does for stress.
"""
from cmad.models.global_fields import GlobalFieldsAtPoint, StepTime
from cmad.models.model import Model
from cmad.typing import JaxArray, Params, Scalar, StateList


class ThermalModel(Model):
    """Constitutive model whose flux is the heat flux.

    The base the heat transfer global residual binds to: the model gives
    the heat flux for the energy balance and the heat capacity rate for
    its rate term. A model with no local state provides the closed-form
    flux and has ``supports_closed_form`` True; a model with local state
    provides ``heat_flux`` from it.
    """

    @staticmethod
    def heat_flux_closed_form(
            params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> JaxArray:
        raise NotImplementedError

    def heat_flux(
            self,
            xi: StateList, xi_prev: StateList, params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> JaxArray:
        raise NotImplementedError

    def heat_capacity_rate(
            self,
            params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
            step_time: StepTime,
    ) -> Scalar:
        raise NotImplementedError
