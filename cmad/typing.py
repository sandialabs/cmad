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
    from cmad.global_residuals.modes import GlobalResidualMode
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
     "Model", "GlobalResidualMode",
     Sequence["ShapeFunctionsAtIP"],
     float, float,
     int],
    Sequence[JaxArray],
]
"""Signature of the per-element-IP residual function passed to
GlobalResidual.__init__. Returns a per-residual-block sequence:
entry ``r`` has shape ``(n_basis_fns_r, n_eqs_r)``, allowing
different blocks to have different shapes (Taylor-Hood u/p, etc.).
U/U_prev and shapes_ip are also per-residual-block sequences.
``mode`` is the operational mode (CLOSED_FORM or COUPLED); the
body branches on it to dispatch the per-physics flux call.
``GlobalResidual.for_model`` captures one mode per closure, so
independent bindings on the same GR don't share state. ``w``
and ``dv`` are the quadrature weight and reference-volume
factor at the IP. The trailing int is the integration-point-set
index dispatched by the assembly layer; single-ip_set GRs ignore
it, multi-ip_set GRs (e.g. mixed u-p with two quadrature orders
for divergence + pressure-mass terms) read it to dispatch term-
specific residual contributions via lax.switch or per-ip_set
branches."""

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
determinants). In COUPLED mode the ``R`` evaluator is a raw
partial at supplied ξ with the different 9-arg call shape
``(params, U, U_prev, xi, xi_prev, shapes_ip, w, dv, ip_set)``;
that signature is not captured by this alias."""

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
for the same reason as REvaluator. COUPLED mode does not provide
this fused evaluator; use ``RAndDRDUAndXiEvaluator`` instead, which
also returns the converged xi as a free side-product of the local
Newton solve."""

DRDUEvaluator: TypeAlias = Callable[
    [Params,
     Sequence[JaxArray], Sequence[JaxArray],
     Sequence["ShapeFunctionsAtIP"],
     float | JaxArray, float | JaxArray,
     int],
    Sequence[Sequence[JaxArray]],
]
"""Signature of the standalone ``dR/dU`` evaluator returned by
GlobalResidual.for_model. ``dR_dU`` is the *total* derivative the
consumer plugs into K and K^T. In CLOSED_FORM mode the closure
boundary matches REvaluator (7-arg, U-only); ``dR_dU`` is a direct
``jacfwd`` over the U-only closure (xi(U) baked into
``cauchy_closed_form``). In COUPLED mode it's the IFT-corrected
total via ``custom_jvp`` around the per-IP local Newton, with the
8-arg call shape ``(params, U, U_prev, xi_prev, shapes_ip, w, dv,
ip_set)`` (xi internally solved from xi_prev via
``make_newton_solve``). Same return shape, same consumer-facing
role; mode determines implementation and input sig."""

DRDUPrevEvaluator: TypeAlias = Callable[
    [Params,
     Sequence[JaxArray], Sequence[JaxArray],
     Sequence["ShapeFunctionsAtIP"],
     float | JaxArray, float | JaxArray,
     int],
    Sequence[Sequence[JaxArray]],
]
"""Signature of the standalone ``dR/dU_prev`` evaluator returned by
GlobalResidual.for_model. In CLOSED_FORM mode the closure boundary
matches REvaluator (7-arg, U-only); zero-valued for GRs whose
``residual_fn`` doesn't depend on U_prev (the currently-implemented
mechanical residuals), and non-trivial for GRs that read previous-
step global fields (e.g., a thermal residual with a transient term
reading T_prev). In COUPLED mode it's a raw partial at supplied ξ
— not an IFT-corrected total — with the 9-arg call shape
``(params, U, U_prev, xi, xi_prev, shapes_ip, w, dv, ip_set)``."""

DRDPEvaluator: TypeAlias = Callable[
    [Params,
     Sequence[JaxArray], Sequence[JaxArray],
     Sequence["ShapeFunctionsAtIP"],
     float | JaxArray, float | JaxArray,
     int],
    Sequence[PyTree],
]
"""Signature of the standalone ``dR/dparams`` evaluator returned by
GlobalResidual.for_model. In CLOSED_FORM mode the closure boundary
matches REvaluator (7-arg, U-only); returns a per-residual-block
pytree parallel to ``params``. In COUPLED mode it's a raw partial
at supplied ξ — not an IFT-corrected total — with the 9-arg call
shape ``(params, U, U_prev, xi, xi_prev, shapes_ip, w, dv,
ip_set)``."""

DRDXIEvaluator: TypeAlias = Callable[
    [Params,
     Sequence[JaxArray], Sequence[JaxArray],
     StateList, StateList,
     Sequence["ShapeFunctionsAtIP"],
     float | JaxArray, float | JaxArray,
     int],
    Sequence[Sequence[JaxArray]],
]
"""Signature of the standalone ``dR/dxi`` evaluator returned by
GlobalResidual.for_model in COUPLED mode (no CLOSED_FORM
counterpart — xi is bound to zeros in CLOSED_FORM). Raw partial
at supplied ξ via ``jacfwd`` over the residual closure with
respect to xi. 9-arg call shape ``(params, U, U_prev, xi, xi_prev,
shapes_ip, w, dv, ip_set)``; returns a list-of-lists indexed by
``(residual_block_r, xi_block_s)``."""

DRDXIPrevEvaluator: TypeAlias = Callable[
    [Params,
     Sequence[JaxArray], Sequence[JaxArray],
     StateList, StateList,
     Sequence["ShapeFunctionsAtIP"],
     float | JaxArray, float | JaxArray,
     int],
    Sequence[Sequence[JaxArray]],
]
"""Signature of the standalone ``dR/dxi_prev`` evaluator returned by
GlobalResidual.for_model in COUPLED mode. Mirror of DRDXIEvaluator
for the previous-step xi (no CLOSED_FORM counterpart)."""

RAndDRDUAndXiEvaluator: TypeAlias = Callable[
    [Params,
     Sequence[JaxArray], Sequence[JaxArray],
     StateList,
     Sequence["ShapeFunctionsAtIP"],
     float | JaxArray, float | JaxArray,
     int],
    tuple[
        Sequence[JaxArray],
        Sequence[Sequence[JaxArray]],
        StateList,
    ],
]
"""Signature of the bundled R + dR/dU + xi evaluator returned by
GlobalResidual.for_model in COUPLED mode. The FE-Newton hot path:
runs the per-IP local Newton once and returns
``(R_blocks, dR_dU_blocks, xi_converged)``; ``R_blocks`` and
``dR_dU_blocks`` match RAndDRDUEvaluator's return shape, plus the
converged xi exposed as a free side-product so state-history
storage at FE-Newton convergence is the bundled call's third
element (no extra solve). 8-arg call shape ``(params, U, U_prev,
xi_prev, shapes_ip, w, dv, ip_set)``; xi is internally solved from
xi_prev via ``make_newton_solve``, with JAX-AD seeing the IFT-
corrected total via the solver's ``custom_jvp`` rule."""


class GREvaluators(TypedDict, total=False):
    """Per-(GR, model, mode) evaluator dict from GlobalResidual.for_model.

    Keys present depend on the mode passed to ``for_model``.

    CLOSED_FORM populates 5 keys, all 7-arg sig (U-only):
    ``R`` (residual only — linesearch trial points and finite-
    difference probes), ``R_and_dR_dU`` (fused for the Newton step,
    XLA CSEs interpolation / kinematics / Cauchy-stress between R
    and its tangent), and the standalone first-derivative evaluators
    ``dR_dU`` / ``dR_dU_prev`` / ``dR_dp``.

    COUPLED populates 7 keys with mixed sigs. Raw partials at
    supplied ξ (9-arg sig including ``xi`` and ``xi_prev``):
    ``R``, ``dR_dU_prev``, ``dR_dp``, ``dR_dxi``, ``dR_dxi_prev``.
    Newton-running totals (8-arg sig — xi internally solved from
    ``xi_prev``): ``dR_dU`` (IFT-corrected total via ``custom_jvp``
    around the per-IP local Newton — the global tangent K, transpose
    K^T for the adjoint) and the bundled ``R_and_dR_dU_and_xi``
    (FE-Newton hot path; returns ``(R, dR_dU, xi)``). The dict
    shape differs per mode — there is no cross-mode key symmetry,
    and ``dR_dxi`` / ``dR_dxi_prev`` / ``R_and_dR_dU_and_xi``
    don't exist in CLOSED_FORM.
    """
    R: REvaluator
    R_and_dR_dU: RAndDRDUEvaluator
    dR_dU: DRDUEvaluator
    dR_dU_prev: DRDUPrevEvaluator
    dR_dp: DRDPEvaluator
    dR_dxi: DRDXIEvaluator
    dR_dxi_prev: DRDXIPrevEvaluator
    R_and_dR_dU_and_xi: RAndDRDUAndXiEvaluator


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
