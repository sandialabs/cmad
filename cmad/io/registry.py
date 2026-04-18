"""Model and QoI registries for the CMAD deck driver.

Models and QoIs register themselves at import time via
``@register_model("name")`` / ``@register_qoi("name")``. Drivers resolve
a deck's ``model.name`` / ``qoi.name`` to a registered class via
:func:`resolve_model` / :func:`resolve_qoi`. Resolution is lazy: the
concrete module is imported on first lookup so the CLI pays only for the
classes a given deck actually uses, which keeps startup cost flat as the
library grows.

Discoverability for :func:`registered_models` / :func:`registered_qois`
(used in the "unknown ..." error messages) comes from the schema-fragment
directories (``cmad/io/schemas/models/<name>.yaml`` and
``cmad/io/schemas/qois/<name>.yaml``) rather than the runtime registries,
so listing available names triggers no user-code imports.

Convention in both registries: ``name`` equals the module filename
(``cmad.models.<name>`` or ``cmad.qois.<name>``), so the lazy import in
the resolve function can find the module by name alone.

See the driver registry contract for details.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from cmad.models.model import Model
    from cmad.qois.qoi import QoI


ModelT = TypeVar("ModelT", bound="Model")
QoIT = TypeVar("QoIT", bound="QoI")

_REGISTRY: dict[str, type[Model]] = {}
_QOI_REGISTRY: dict[str, type[QoI]] = {}
_SCHEMAS_MODELS_DIR = Path(__file__).parent / "schemas" / "models"
_SCHEMAS_QOIS_DIR = Path(__file__).parent / "schemas" / "qois"


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


def register_qoi(name: str) -> Callable[[type[QoIT]], type[QoIT]]:
    """Register a :class:`QoI` subclass under a deck-facing name."""
    if not name or not name.strip():
        raise ValueError("register_qoi: name must be a non-empty string")

    def decorator(cls: type[QoIT]) -> type[QoIT]:
        if name in _QOI_REGISTRY:
            existing = _QOI_REGISTRY[name].__name__
            raise ValueError(
                f"register_qoi: '{name}' is already registered "
                f"(by {existing}); each name must be unique",
            )
        _QOI_REGISTRY[name] = cls
        return cls

    return decorator


def resolve_qoi(name: str) -> type[QoI]:
    """Look up a registered QoI class by deck name, importing on demand.

    The module path convention is ``cmad.qois.<name>``. If the schema
    fragment for ``name`` exists but the module path has no
    registration, the "not registered" error still fires after the
    import attempt.
    """
    if name in _QOI_REGISTRY:
        return _QOI_REGISTRY[name]

    known = registered_qois()
    if name in known:
        import_module(f"cmad.qois.{name}")

    if name not in _QOI_REGISTRY:
        listing = ", ".join(known) if known else "(none)"
        raise ValueError(
            f"qoi.name: '{name}' is not registered. "
            f"Registered qoi names: {listing}",
        )
    return _QOI_REGISTRY[name]


def registered_qois() -> list[str]:
    """Return the sorted list of discoverable qoi names.

    Discoverability is by schema-fragment presence under
    ``cmad/io/schemas/qois/<name>.yaml``; no qoi modules are imported.
    """
    if not _SCHEMAS_QOIS_DIR.exists():
        return []
    return sorted(p.stem for p in _SCHEMAS_QOIS_DIR.glob("*.yaml"))
