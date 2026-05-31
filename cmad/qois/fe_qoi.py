"""FE-side QoI ABC: closure-factory lifecycle keyed by per-block params."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

from cmad.qois.qoi_base import QoIBase
from cmad.typing import JaxArray, Params

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem, FEState
    from cmad.fem.kernel_arrays import FEKernelArrays


StepContribution: TypeAlias = Callable[
    [
        JaxArray,
        JaxArray,
        Mapping[str, JaxArray],
        Mapping[str, JaxArray],
        JaxArray,
        JaxArray,
    ],
    JaxArray,
]
"""Per-step QoI increment.

Signature ``(U, U_prev, xi, xi_prev, t, t_prev) -> J_n`` where
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
- ``t``, ``t_prev``: scalar times of the current and previous
  steps.

Time-varying state only — params do not appear on this interface;
they are captured by the factory :meth:`FEQoI.step_contribution`
when the QoI needs them.
"""


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
        with ``(U, U_prev, xi, xi_prev, t, t_prev)`` and accumulates
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
