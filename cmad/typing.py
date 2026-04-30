"""Type aliases, protocols, and result types for the CMAD framework.

This module is the single source of truth for cross-module type names.
Imports are kept minimal: stdlib, jax, numpy, and (under TYPE_CHECKING)
`cmad.models.global_fields` for the `GlobalFieldsAtPoint` ctx type
referenced in function-signature aliases.
"""
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeAlias, TypedDict

import numpy as np
from jax import Array as JaxArray
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cmad.fem.shapes import ShapeFunctionsAtIP
    from cmad.models.global_fields import GlobalFieldsAtPoint
    from cmad.models.model import Model

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
"""Constitutive-model parameter values.

Used as a strict alias for storage on Parameters and at the public API
boundary. Helper functions that traverse `params["X"]["Y"]`-style chains
internally (effective_stress, hardening, elastic_stress, etc.) instead
take `dict[str, Any]` because mypy cannot narrow the recursive PyTreeDict
union at each chain step. A future commit could replace the looseness
with TypedDicts per parameter shape (J2EffectiveStressParams,
HillCoeffs, VoceParams, etc.) for full strict typing.
"""

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

# ----- Function signatures -----

ResidualFn: TypeAlias = Callable[
    [StateList, StateList, Params,
     "GlobalFieldsAtPoint", "GlobalFieldsAtPoint"], JaxArray,
]
"""Signature of the per-block residual function passed to Model.__init__."""

CauchyFn: TypeAlias = Callable[
    [StateList, StateList, Params,
     "GlobalFieldsAtPoint", "GlobalFieldsAtPoint"], JaxArray,
]
"""Signature of the Cauchy stress function passed to Model.__init__."""

QoIFn: TypeAlias = Callable[
    [StateList, StateList, Params,
     "GlobalFieldsAtPoint", "GlobalFieldsAtPoint",
     JaxArray, JaxArray], JaxArray,
]
"""Signature of the QoI function passed to QoI.__init__."""

ResidualFnGR: TypeAlias = Callable[
    [StateList, StateList, Params,
     Sequence[JaxArray], Sequence[JaxArray],
     "Model",
     Sequence["ShapeFunctionsAtIP"],
     float, float,
     int],
    Sequence[JaxArray],
]
"""Signature of the per-element-IP residual function passed to
GlobalResidual.__init__. Returns a per-residual-block sequence:
entry ``r`` has shape ``(n_basis_fns_r, n_eqs_r)``, allowing
different blocks to have different shapes (Taylor-Hood u/p, etc.).
U/U_prev and shapes_ip are also per-residual-block sequences. ``w``
and ``dv`` are the quadrature weight and reference-volume factor at
the IP. The trailing int is the integration-point-set index
dispatched by the assembly layer; single-ip_set GRs ignore it,
multi-ip_set GRs (e.g. mixed u-p with two quadrature orders for
divergence + pressure-mass terms) read it to dispatch term-specific
residual contributions via lax.switch or per-ip_set branches."""

REvaluator: TypeAlias = Callable[
    [Params,
     Sequence[JaxArray], Sequence[JaxArray],
     Sequence["ShapeFunctionsAtIP"],
     float | JaxArray, float | JaxArray,
     int],
    Sequence[JaxArray],
]
"""Signature of the model-closed R-only evaluator returned by
GlobalResidual.for_model in CLOSED_FORM mode. The closure binds the
``model`` argument away and the U-only call shape — xi / xi_prev
are not arguments because CLOSED_FORM evaluates stress through
``model.cauchy_closed_form(params, U_ip, U_ip_prev)`` and never
consults state. Returns the per-residual-block ``R_blocks`` list.
``w`` and ``dv`` accept Python floats (test-side scalars) or 0-d
JaxArrays (assembly-side quadrature weights and Jacobian
determinants)."""

RAndDRDUEvaluator: TypeAlias = Callable[
    [Params,
     Sequence[JaxArray], Sequence[JaxArray],
     Sequence["ShapeFunctionsAtIP"],
     float | JaxArray, float | JaxArray,
     int],
    tuple[Sequence[JaxArray], Sequence[Sequence[JaxArray]]],
]
"""Signature of the fused R + dR/dU evaluator returned by
GlobalResidual.for_model in CLOSED_FORM mode. Returns
``(R_blocks, dR_dU_blocks)`` where ``R_blocks`` matches the
REvaluator return and ``dR_dU_blocks`` is list-of-lists keyed by
``(residual_block_r, U_block_s)``. U-only at the closure boundary
for the same reason as REvaluator."""


class GREvaluators(TypedDict, total=False):
    """Per-(GR, model, mode) evaluator dict from GlobalResidual.for_model.

    Keys present depend on the mode passed to ``for_model``. CLOSED_FORM
    populates both ``R`` and ``R_and_dR_dU``: the R-only evaluator
    covers cheap-residual paths (linesearch trial points, finite-
    difference probes), the fused evaluator is the Newton-step path
    (XLA CSEs interpolation / kinematics / Cauchy-stress between R and
    its tangent). Future modes (COUPLED with custom_vjp around per-IP
    local Newton) will add their own keys here.
    """
    R: REvaluator
    R_and_dR_dU: RAndDRDUEvaluator


# ----- Indices -----

Step: TypeAlias = int
"""A time-step index (0-based)."""


# ----- MPObjective.evaluate() result types -----

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


class SupportsPrimalLoop(SupportsNewton, Protocol):
    """Minimum surface area for the MP forward-solve loop.

    Extends SupportsNewton with the state-advancing and stress-extraction
    operations that the ``cmad primal`` driver invokes around each
    Newton solve.
    """
    def set_xi_to_init_vals(self) -> None: ...
    def gather_global(
        self,
        U: "GlobalFieldsAtPoint",
        U_prev: "GlobalFieldsAtPoint",
    ) -> None: ...
    def advance_xi(self) -> None: ...
    def evaluate_cauchy(self) -> None: ...
    def xi(self) -> StateList: ...
    def Sigma(self) -> NDArray[np.floating]: ...
