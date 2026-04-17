"""Build a :class:`Parameters` instance from a deck's ``parameters:`` tree.

The deck's parameter leaves are either bare scalars/lists (implicit
``active=False``, no transform) or dict-form entries of shape
``{value, active?, transform?}``. The builder walks the tree and splits
each leaf into three parallel pytrees (values / active flags / transforms)
in the shape :class:`Parameters` expects.

Leaf-value coercions happen at the boundary so the rest of the framework
sees the types it expects: YAML lists become ``np.ndarray`` (so matrices
like ``rotation matrix`` are real arrays), and integer scalars become
floats (``get_size`` in :mod:`cmad.parameters.parameters` only accepts
float / ndarray leaves).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from cmad.parameters.parameters import Parameters
from cmad.typing import ActiveFlags, Params, Transforms


def build_parameters(parameters_section: dict[str, Any]) -> Parameters:
    """Split ``{value, active, transform}`` leaves into three parallel pytrees."""
    values, active_flags, transforms = _split(parameters_section)
    return Parameters(
        values=cast(Params, values),
        active_flags=cast(ActiveFlags, active_flags),
        transforms=cast(Transforms, transforms),
    )


def _split(node: Any) -> tuple[Any, Any, Any]:
    if isinstance(node, dict) and "value" in node:
        return (
            _coerce_value(node["value"]),
            bool(node.get("active", False)),
            _parse_transform(node.get("transform")),
        )
    if isinstance(node, dict):
        vals: dict[str, Any] = {}
        acts: dict[str, Any] = {}
        trs: dict[str, Any] = {}
        for k, v in node.items():
            vals[k], acts[k], trs[k] = _split(v)
        return vals, acts, trs
    return _coerce_value(node), False, None


def _coerce_value(value: Any) -> Any:
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return float(value)
    return value


def _parse_transform(spec: Any) -> NDArray[np.float64] | None:
    # Transforms must be ndarrays (not Python lists), because
    # ``jax.tree_util.tree_flatten`` recurses into lists. Existing test
    # fixtures use ``np.array([lo, hi])`` / ``np.array([ref])``.
    if spec is None:
        return None
    if isinstance(spec, dict) and "bounds" in spec:
        return np.asarray(spec["bounds"], dtype=np.float64)
    if isinstance(spec, dict) and "log" in spec:
        return np.asarray([spec["log"]], dtype=np.float64)
    raise ValueError(f"unknown transform spec: {spec!r}")
