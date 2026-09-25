"""Material parameters written as functions of temperature.

A parameter is a plain number, ``{polynomial: {coefficients: [c0, c1, ...]}}``
for ``c0 + c1 T + c2 T^2 + ...``, or ``{table: {T: [...], values: [...]}}``,
interpolated linearly and held at its end values outside the knots.
Temperatures are in Kelvin.
"""
from collections.abc import Callable
from typing import Any

import numpy as np
from jax import numpy as jnp

from cmad.models.global_fields import GlobalFieldsAtPoint, temperature_at_point
from cmad.typing import JaxArray, Scalar

DEFAULT_REFERENCE_TEMPERATURE = 293.15


def evaluate_polynomial(node: dict[str, Any], T: Scalar) -> JaxArray:
    return jnp.polyval(node["coefficients"][::-1], T)


def evaluate_table(node: dict[str, Any], T: Scalar) -> JaxArray:
    return jnp.interp(T, node["T"], node["values"])


FORMS: dict[str, Callable[[dict[str, Any], Scalar], JaxArray]] = {
    "polynomial": evaluate_polynomial,
    "table": evaluate_table,
}


def parameter_form(node: Any) -> str | None:
    """The form's name when ``node`` is a parameter written as a function of
    temperature, else ``None``."""
    if isinstance(node, dict) and len(node) == 1:
        (name,) = node
        if name in FORMS:
            return str(name)
    return None


def has_parameter_forms(values: Any) -> bool:
    if parameter_form(values) is not None:
        return True
    if isinstance(values, dict):
        return any(has_parameter_forms(value) for value in values.values())
    return False


def check_parameter_forms(values: Any, path: str = "") -> None:
    """Raise a ``ValueError`` naming the parameter whose polynomial or table
    is malformed."""
    if not isinstance(values, dict):
        return

    named = [key for key in values if key in FORMS]
    if named and len(values) > 1:
        raise ValueError(
            f"{path}: '{named[0]}' must be the only key for its parameter")

    name = parameter_form(values)
    if name == "polynomial":
        node = values[name]
        if not isinstance(node, dict) or set(node) != {"coefficients"}:
            raise ValueError(
                f"{path}.polynomial: needs the one key 'coefficients'")
        coefficients = np.asarray(node["coefficients"])
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError(
                f"{path}.polynomial.coefficients: needs a list with one or "
                f"more numbers")
    elif name == "table":
        node = values[name]
        if not isinstance(node, dict) or set(node) != {"T", "values"}:
            raise ValueError(f"{path}.table: needs the keys 'T' and 'values'")
        knots = np.asarray(node["T"])
        entries = np.asarray(node["values"])
        if knots.ndim != 1 or knots.shape != entries.shape or knots.size < 2:
            raise ValueError(
                f"{path}.table: 'T' and 'values' need the same length, two "
                f"or more")
        if not np.all(np.diff(knots) > 0.0):
            raise ValueError(f"{path}.table.T: must be strictly increasing")
    else:
        for key, value in values.items():
            check_parameter_forms(value, f"{path}.{key}" if path else key)


def evaluate_parameters(params: dict[str, Any], T: Scalar) -> dict[str, Any]:
    """``params`` with every parameter written as a function of temperature
    replaced by its value at ``T``."""
    evaluated: dict[str, Any] = {}
    for key, value in params.items():
        name = parameter_form(value)
        if name is not None:
            evaluated[key] = FORMS[name](value[name], T)
        elif isinstance(value, dict):
            evaluated[key] = evaluate_parameters(value, T)
        else:
            evaluated[key] = value
    return evaluated


def make_parameter_resolver(
        values: dict[str, Any], reference_temperature: float,
) -> Callable[[dict[str, Any], GlobalFieldsAtPoint], dict[str, Any]]:
    """The function a model calls as ``resolve_parameters(params, U)``: the
    parameters at the point's temperature, or at ``reference_temperature``
    when the point carries no ``T`` field. The identity when no parameter is
    written as a function of temperature, so nothing enters the trace."""
    check_parameter_forms(values)
    if not has_parameter_forms(values):
        return lambda params, U: params

    def resolve_parameters(
            params: dict[str, Any], U: GlobalFieldsAtPoint,
    ) -> dict[str, Any]:
        T = temperature_at_point(U)
        if T is None:
            T = reference_temperature
        return evaluate_parameters(params, T)

    return resolve_parameters
