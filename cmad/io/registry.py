"""Model registry for the CMAD deck driver.

Models register themselves at import time via ``@register_model("name")``.
Drivers resolve a deck's ``model.name`` to the registered class via
``resolve_model``. See the driver registry contract for details.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from cmad.models.model import Model


ModelT = TypeVar("ModelT", bound="Model")

_REGISTRY: dict[str, type[Model]] = {}


def register_model(name: str) -> Callable[[type[ModelT]], type[ModelT]]:
    """Register a :class:`Model` subclass under a deck-facing name."""
    if not name or not name.strip():
        raise ValueError("register_model: name must be a non-empty string")

    def decorator(cls: type[ModelT]) -> type[ModelT]:
        if name in _REGISTRY:
            existing = _REGISTRY[name].__name__
            raise ValueError(
                f"register_model: '{name}' is already registered "
                f"(by {existing}); each name must be unique",
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def resolve_model(name: str) -> type[Model]:
    """Look up a registered class by its deck name."""
    if name not in _REGISTRY:
        known = registered_models()
        listing = ", ".join(known) if known else "(none)"
        raise ValueError(
            f"model.name: '{name}' is not registered. "
            f"Registered model names: {listing}",
        )
    return _REGISTRY[name]


def registered_models() -> list[str]:
    """Return the sorted list of registered model names."""
    return sorted(_REGISTRY)
