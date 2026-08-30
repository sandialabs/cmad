"""FE-side QoI ABC: closure-factory lifecycle keyed by per-block params."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.qois.qoi_base import QoIBase
from cmad.typing import JaxArray, Params, Scalar

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem, FEState
    from cmad.fem.kernel_arrays import FEKernelArrays
    from cmad.models.global_fields import StepTime


StepContribution: TypeAlias = Callable[
    [
        JaxArray,
        JaxArray,
        Mapping[str, JaxArray],
        Mapping[str, JaxArray],
        "StepTime",
    ],
    JaxArray,
]
"""Per-step QoI increment.

Signature ``(U, U_prev, xi, xi_prev, step_time) -> J_n`` where
``J_n`` is the scalar increment whose sum over the time loop is the
full QoI value.

- ``U``, ``U_prev``: global flat basis-coefficient vectors of shape
  ``(num_total_dofs,)`` — the whole FE state at the current and
  previous steps. The closure interpolates them to integration
  points internally if it needs IP-level field values.
- ``xi``, ``xi_prev``: per-block per-element per-IP local-state
  dicts of the form
  ``{block_name: array of shape (n_elems, n_ips, total_xi_dofs)}``
  (empty for CLOSED_FORM-only problems, since CLOSED_FORM blocks
  carry no time-varying state).
- ``step_time``: the current and previous step times as a
  :class:`~cmad.models.global_fields.StepTime`; ``step_time.dt`` is
  the increment.

Time-varying state only — params do not appear on this interface;
they are captured by the factory :meth:`FEQoI.step_contribution`
when the QoI needs them.
"""


@dataclass(frozen=True)
class MatchTimes:
    """The times a matching QoI scores and the weight of each.

    The weight of match time ``n`` is ``t_n - t_(n-1)``, zero for the
    first, so summing weight times mismatch over the solve steps
    integrates the mismatch in time. A solve step whose time is not a
    match time, within ``tol``, contributes nothing. When the match
    times are the time schedule itself, every step's weight is its
    ``dt``.
    """

    times: NDArray[np.float64]
    weights: NDArray[np.float64]
    tol: float

    @classmethod
    def from_times(
            cls,
            times: Sequence[float] | NDArray[np.float64],
            rtol: float = 1.0e-8,
    ) -> MatchTimes:
        arr = np.asarray(times, dtype=np.float64).ravel()
        if arr.size == 0 or not np.all(np.diff(arr) > 0.0):
            raise ValueError(
                "match times must be nonempty and strictly increasing"
            )
        weights = np.concatenate([[0.0], np.diff(arr)])
        return cls(arr, weights, rtol * float(arr[-1] - arr[0]))

    @property
    def span(self) -> float:
        return float(self.times[-1] - self.times[0])

    def index_and_weight(self, t: Scalar) -> tuple[JaxArray, JaxArray]:
        """Index of the match time nearest ``t`` and its weight, zero when
        ``t`` is not a match time; traceable."""
        times = jnp.asarray(self.times)
        k = jnp.argmin(jnp.abs(times - t))
        weight = jnp.where(
            jnp.abs(times[k] - t) <= self.tol,
            jnp.asarray(self.weights)[k], 0.0,
        )
        return k, weight


class FEQoI(QoIBase, ABC):
    """ABC for FE-shaped QoIs accumulated over a quasi-static time loop.

    FE QoIs build a per-step closure via a factory call
    :meth:`step_contribution` that takes ``params_by_block`` and the
    mesh kernel arrays, and returns a state-only callable. The factory
    exists to let
    parameter-dependent QoIs (future FEMU / regularization terms)
    capture ``params_by_block`` by Python closure — AD traces through
    the capture when ``params_flat`` is the tracer in
    ``J(params_flat)``. QoIs whose value depends on the parameters
    only through the solved state (e.g.
    :class:`cmad.qois.fe_displacement_l2.FEDisplacementL2`) ignore
    ``params_by_block`` and return a state-only closure unchanged.

    Structural data — ``fe_problem``, total volume, time horizon — are
    captured at ``__init__`` (cached once per instance), not per
    factory call. Future QoIs that have their own parameters separate
    from the model parameters (FEMU regularization weights, etc.)
    capture them at ``__init__`` from the deck via :meth:`from_deck`.
    """

    problem_type: ClassVar[str] = "fe"

    @abstractmethod
    def step_contribution(
            self,
            params_by_block: Mapping[str, Params],
            fe_arrays: FEKernelArrays,
    ) -> StepContribution:
        """Build the per-step closure that accumulates into J.

        ``params_by_block`` enters here so QoIs that depend on params
        — e.g. those that call ``assemble_global`` to read reaction
        forces — can capture it via Python closure. ``fe_arrays`` is
        the mesh-derived kernel-array carrier; QoIs that interpolate
        or integrate fields over the mesh capture it the same way and
        read their geometry and index arrays from it. The returned
        closure has the time-varying-state-only signature
        :data:`StepContribution`; the driver calls it once per step
        with ``(U, U_prev, xi, xi_prev, step_time)`` and accumulates
        the returned scalar into ``J``.
        """
        ...

    @classmethod
    @abstractmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
    ) -> FEQoI:
        """Build a QoI instance from a parsed deck's ``qoi:`` section.

        ``qoi_section`` carries QoI-specific fields (sideset names,
        regularization weights, etc.). ``fe_problem`` is the built FE
        problem the QoI evaluates against. ``t_schedule`` is the
        deck's time schedule; QoIs that integrate over the time loop
        capture the horizon ``T = t_schedule[-1] - t_schedule[0]`` at
        construction.
        """
        ...

    def produces_primal_output(self) -> bool:
        """Whether this QoI writes an output from the solved trajectory
        rather than being an objective accumulated over the time loop.

        Default ``False``. A QoI that returns ``True`` is asked by ``cmad
        primal`` to emit its file via :meth:`write_primal_outputs` after the
        solve, and is not evaluated as an objective.
        """
        return False

    def write_primal_outputs(
            self, fe_problem: FEProblem, fe_state: FEState,
    ) -> None:
        """Write an output from the solved trajectory ``fe_state``.

        Called by ``cmad primal`` only when :meth:`produces_primal_output`
        is ``True``; the default raises, since an objective QoI has nothing
        to write.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not produce a primal output"
        )
