"""Base class for constitutive models whose flux is the heat flux.

Separates the thermal contract the heat transfer global residual relies
on from the general :class:`cmad.models.model.Model`, the way
:class:`cmad.models.mechanics_model.MechanicsModel` does for stress.
"""
from collections.abc import Callable

from cmad.models.global_fields import GlobalFieldsAtPoint, StepTime
from cmad.models.model import Model
from cmad.typing import JaxArray, Params, Scalar, StateList


class ThermalModel(Model):
    """Constitutive model whose flux is the heat flux.

    The base the heat transfer global residual binds to: the model gives
    the heat flux for the energy balance and the heat capacity rate for
    its rate term. A model with no local state sets ``heat_flux_closed_form``
    at construction and has ``supports_closed_form`` True; a model with
    local state provides ``heat_flux`` from it. The temperature gradient
    in ``U`` is with respect to the current position; under finite
    deformation the energy balance converts it before the call. ``face_flux``
    is the heat
    flux out of one face of a plate modeled in its plane, per unit face
    area, set at construction by a model that has one. ``heat_generation``
    is the rate of heat generation per unit reference volume, set by a
    model that has one.
    """

    heat_flux_closed_form: Callable[
        [Params, GlobalFieldsAtPoint, GlobalFieldsAtPoint], JaxArray]
    face_flux: Callable[
        [Params, GlobalFieldsAtPoint, GlobalFieldsAtPoint], Scalar] | None = None
    heat_generation: Callable[
        [StateList, StateList, Params, GlobalFieldsAtPoint,
         GlobalFieldsAtPoint, StepTime], Scalar] | None = None

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
