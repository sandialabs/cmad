"""Thermal expansion.

``thermal expansion`` in a material has two entries: ``alpha``, the mean
coefficient of linear thermal expansion, a plain number or a function of
temperature (:mod:`cmad.models.temperature_dependent_parameters`), and
``reference temperature``, the temperature the mean is measured from
(handbook tables give means from room temperature). A free piece has the
length ``L(T) = L(T0) (1 + alpha(T) (T - T0))``. The models take the body
as unstrained at the simulation's reference temperature, so the thermal
stretch they use is ``L(T) / L(T_ref)``.
"""
from collections.abc import Callable
from typing import Any

import numpy as np
from jax import numpy as jnp

from cmad.models.global_fields import GlobalFieldsAtPoint, temperature_at_point
from cmad.models.temperature_dependent_parameters import evaluate_parameters
from cmad.typing import JaxArray, Scalar

THERMAL_EXPANSION_KEYS = frozenset({"alpha", "reference temperature"})


def check_thermal_expansion(values: dict[str, Any]) -> None:
    """Raise a ``ValueError`` unless ``thermal expansion``, when present,
    holds exactly ``alpha`` and one number ``reference temperature``."""
    if "thermal expansion" not in values:
        return
    expansion = values["thermal expansion"]
    if not isinstance(expansion, dict) \
            or set(expansion) != THERMAL_EXPANSION_KEYS:
        raise ValueError(
            "thermal expansion: needs 'alpha', the mean coefficient of linear "
            "thermal expansion, and 'reference temperature', the temperature "
            "the mean is measured from")
    T0 = expansion["reference temperature"]
    if isinstance(T0, dict) or np.ndim(T0) != 0:
        raise ValueError(
            "thermal expansion.reference temperature: must be one number")


def make_thermal_stretch_function(
        values: dict[str, Any], reference_temperature: float,
) -> Callable[[dict[str, Any], GlobalFieldsAtPoint], Scalar]:
    """Build ``compute_thermal_stretch(params, U)``: ``L(T) / L(T_ref)`` at
    the point's temperature, or one when the material has no ``thermal
    expansion``. It reads ``params`` as stored, ``alpha`` unresolved,
    because it evaluates ``alpha`` at ``T`` and at ``T_ref``."""
    check_thermal_expansion(values)
    if "thermal expansion" not in values:
        return lambda params, U: 1.0

    def length(expansion: dict[str, Any], T: Scalar) -> Scalar:
        alpha = evaluate_parameters(expansion, T)["alpha"]
        return 1.0 + alpha * (T - expansion["reference temperature"])

    def compute_thermal_stretch(
            params: dict[str, Any], U: GlobalFieldsAtPoint,
    ) -> Scalar:
        expansion = params["thermal expansion"]
        T = temperature_at_point(U, reference_temperature)
        return length(expansion, T) / length(expansion, reference_temperature)

    return compute_thermal_stretch


def elastic_deformation_gradient(
        F: JaxArray, thermal_stretch: Scalar, finite_deformation: bool,
) -> JaxArray:
    """``F`` with its thermal part removed: ``F / thermal_stretch`` in
    finite deformation, ``F - (thermal_stretch - 1) I`` in small strain."""
    if finite_deformation:
        return F / thermal_stretch
    return F - (thermal_stretch - 1.0) * jnp.eye(3)


def thermal_strain_increment(
        thermal_stretch: Scalar, thermal_stretch_prev: Scalar,
        finite_deformation: bool,
) -> JaxArray:
    """The thermal strain of one step: ``log(thermal_stretch /
    thermal_stretch_prev) I`` in finite deformation, ``(thermal_stretch -
    thermal_stretch_prev) I`` in small strain."""
    I = jnp.eye(3)
    if finite_deformation:
        return jnp.log(thermal_stretch / thermal_stretch_prev) * I
    return (thermal_stretch - thermal_stretch_prev) * I
