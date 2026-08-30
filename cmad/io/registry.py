"""Model, QoI, and global residual resolution for the CMAD deck driver.

A deck name is a module name: :func:`resolve_model` imports
``cmad.models.<name>`` and returns the one :class:`Model` subclass that
module defines; :func:`resolve_qoi` and :func:`resolve_global_residual`
do the same under ``cmad.qois`` and ``cmad.global_residuals``. A name
is available exactly when its module exists; nothing registers itself
as an import side effect. Resolution is lazy, importing only the module
a deck names, which keeps startup cost flat as the library grows.
Unknown names raise with a listing of the modules actually present.
"""

from __future__ import annotations

import inspect
import pkgutil
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from cmad.global_residuals.global_residual import GlobalResidual
    from cmad.models.model import Model
    from cmad.qois.qoi_base import QoIBase

T = TypeVar("T")


def resolve_model(name: str, where: str = "model.name") -> type[Model]:
    """The Model subclass defined by ``cmad.models.<name>``."""
    from cmad.models.model import Model
    return _resolve_class("cmad.models", name, Model, where, "model")


def resolve_qoi(name: str, where: str = "qoi.name") -> type[QoIBase]:
    """The QoI class defined by ``cmad.qois.<name>``.

    Returns ``type[QoIBase]`` since one namespace serves the MP
    (:class:`cmad.qois.qoi.QoI`) and FE (:class:`cmad.qois.fe_qoi.FEQoI`)
    hierarchies. Callers (``build_mp_problem`` /
    ``build_fe_problem_from_deck``) check ``cls.problem_type`` against
    the deck's ``problem.type``, so the driver enforces the pairing
    rather than the resolver.
    """
    from cmad.qois.qoi_base import QoIBase
    return _resolve_class("cmad.qois", name, QoIBase, where, "qoi")


def resolve_global_residual(
        name: str, where: str = "residuals.global residual.type",
) -> type[GlobalResidual]:
    """The GlobalResidual subclass defined by ``cmad.global_residuals.<name>``."""
    from cmad.global_residuals.global_residual import GlobalResidual
    return _resolve_class(
        "cmad.global_residuals", name, GlobalResidual, where,
        "global residual",
    )


def _resolve_class(
        package: str, name: str, base: type[T], where: str, kind: str,
) -> type[T]:
    """Import ``<package>.<name>`` and return its one ``base`` subclass.

    Only the class defined in the module itself counts: classes the
    module merely imports, and abstract classes, do not. A missing
    import inside a real module propagates; only the named module's own
    absence becomes the availability error.
    """
    module_path = f"{package}.{name}"
    module: ModuleType | None
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as err:
        if err.name != module_path:
            raise
        module = None
    if module is None:
        listing = ", ".join(_available_names(package, base)) or "(none)"
        raise ValueError(
            f"{where}: '{name}' is not available. "
            f"Available {kind} names: {listing}",
        )
    found = _defined_classes(module, base)
    if len(found) != 1:
        raise ValueError(
            f"{where}: module '{module_path}' defines {len(found)} "
            f"{base.__name__} subclasses; expected exactly one",
        )
    return found[0]


def _defined_classes(module: ModuleType, base: type[T]) -> list[type[T]]:
    """The ``base`` subclasses defined in ``module`` itself.

    Classes the module imports from elsewhere do not count, and neither
    do abstract classes.
    """
    return [
        obj for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, base)
        and obj.__module__ == module.__name__
        and not inspect.isabstract(obj)
    ]


def _available_names(package: str, base: type[T]) -> list[str]:
    """The module names under ``package`` defining a ``base`` subclass.

    A class that another found class inherits from is dropped, so base
    class modules are not offered as names. Imports every module in the
    package; this only runs while building the error message for an
    unknown name.
    """
    class_by_name = {}
    for found in pkgutil.iter_modules(import_module(package).__path__):
        module = import_module(f"{package}.{found.name}")
        classes = _defined_classes(module, base)
        if classes:
            class_by_name[found.name] = classes[0]
    names = []
    for name, cls in class_by_name.items():
        is_base_of_another = any(
            issubclass(other, cls) and other is not cls
            for other in class_by_name.values()
        )
        if not is_base_of_another:
            names.append(name)
    return sorted(names)
