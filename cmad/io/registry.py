"""Model registry for the CMAD deck driver.

Models register themselves at import time via ``@register_model("name")``.
Drivers resolve a deck's ``model.name`` to a registered class via
:func:`resolve_model`. Resolution is lazy: the concrete model module is
imported on first lookup so the CLI pays only for the models a given
deck actually uses, which keeps startup cost flat as the model library
grows.

Discoverability for :func:`registered_models` (used in the "unknown
model" error message) comes from the schema-fragment directory
(``cmad/io/schemas/models/<name>.yaml``) rather than the runtime
registry, so listing available names triggers no model imports.

See the driver registry contract for details.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from cmad.models.model import Model


ModelT = TypeVar("ModelT", bound="Model")

_REGISTRY: dict[str, type[Model]] = {}
_SCHEMAS_MODELS_DIR = Path(__file__).parent / "schemas" / "models"


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
    """Look up a registered class by deck name, importing its module on demand.

    The module path convention is ``cmad.models.<name>``. If the schema
    fragment for ``name`` exists but the module path has no registration,
    the "not registered" error still fires after the import attempt.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]

    known = registered_models()
    if name in known:
        import_module(f"cmad.models.{name}")

    if name not in _REGISTRY:
        listing = ", ".join(known) if known else "(none)"
        raise ValueError(
            f"model.name: '{name}' is not registered. "
            f"Registered model names: {listing}",
        )
    return _REGISTRY[name]


def registered_models() -> list[str]:
    """Return the sorted list of discoverable model names.

    Discoverability is by schema-fragment presence under
    ``cmad/io/schemas/models/<name>.yaml``; no models are imported.
    """
    if not _SCHEMAS_MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in _SCHEMAS_MODELS_DIR.glob("*.yaml"))
