"""Type aliases, protocols, and result types for the CMAD framework.

This module is the single source of truth for cross-module type names.
It imports from typing, collections.abc, jax, and numpy only.
"""
from collections.abc import Callable
from typing import NamedTuple, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
from jax import Array as JaxArray


# ----- Pytree types -----

PyTreeLeaf: TypeAlias = JaxArray | NDArray[np.floating] | float | int | bool | None
"""A leaf in a pytree: numeric scalar, array, bool, or None."""

Transform: TypeAlias = list[float] | None
"""None (identity), [lo, hi] (bounds), or [ref] (log)."""

PyTree: TypeAlias = (
    PyTreeLeaf
    | dict[str, "PyTree"]
    | list["PyTree"]
    | tuple["PyTree", ...]
)
"""A nested mapping/list/tuple of PyTreeLeafs (the loose, fully-general
pytree)."""

PyTreeDict: TypeAlias = dict[str, "PyTreeDict | PyTreeLeaf | Transform"]
"""A pytree restricted to nested dicts of leaves; used for parameter,
active-flag, and transform trees in CMAD, which are always dict-shaped."""

Params: TypeAlias = PyTreeDict
"""Constitutive-model parameter values."""

ActiveFlags: TypeAlias = PyTreeDict
"""Pytree parallel to a Params tree; each leaf is a bool (covered by
PyTreeLeaf)."""

Transforms: TypeAlias = PyTreeDict
"""Pytree parallel to a Params tree; each leaf is a Transform."""


# ----- Material-point state -----

StateBlock: TypeAlias = NDArray[np.floating] | JaxArray
"""One residual block's state vector (xi or xi_prev). Either numpy or
jax-typed; the framework navigates between them freely."""

StateList: TypeAlias = list[StateBlock]
"""All residual blocks' state vectors, indexed by residual index."""

GlobalField: TypeAlias = NDArray[np.floating] | JaxArray
"""One block of prescribed global data (u or u_prev) at this material
point. Same numpy-or-jax convention as StateBlock."""

GlobalList: TypeAlias = list[GlobalField]
"""All global-field blocks, indexed by residual index."""


# ----- Function signatures -----

ResidualFn: TypeAlias = Callable[
    [StateList, StateList, Params, GlobalList, GlobalList], JaxArray
]
"""Signature of the per-block residual function passed to Model.__init__."""

CauchyFn: TypeAlias = Callable[
    [StateList, StateList, Params, GlobalList, GlobalList], JaxArray
]
"""Signature of the Cauchy stress function passed to Model.__init__."""

QoIFn: TypeAlias = Callable[
    [StateList, StateList, Params, GlobalList, GlobalList, JaxArray, JaxArray],
    JaxArray,
]
"""Signature of the QoI function passed to QoI.__init__."""


# ----- Indices -----

Step: TypeAlias = int
"""A time-step index (0-based)."""


# ----- Objective.evaluate() result types -----

class GradientResult(NamedTuple):
    """Result of a gradient-only sensitivity evaluation."""
    J: float
    grad: NDArray[np.floating]


class HessianResult(NamedTuple):
    """Result of a Hessian-providing sensitivity evaluation."""
    J: float
    grad: NDArray[np.floating]
    hessian: NDArray[np.floating]


# ----- Protocols -----

class SupportsNewton(Protocol):
    """Minimum surface area newton_solve needs from a model.

    Decoupled from Model so the same solver works against future
    GlobalResidual instances without inheritance plumbing.
    """
    def evaluate(self) -> None: ...
    def C(self) -> NDArray[np.floating]: ...
    def Jac(self) -> NDArray[np.floating]: ...
    def add_to_xi(self, delta: NDArray[np.floating]) -> None: ...
    def seed_none(self) -> None: ...
    def seed_xi(self) -> None: ...
