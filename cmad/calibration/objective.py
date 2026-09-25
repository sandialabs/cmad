"""Calibration objectives over FE problems.

:class:`Objective` is the weighted sum over one or more :class:`Specimen`
problems in the form an optimizer consumes. ``evaluate`` returns
``(J, grad)``, refining a specimen's solve schedule on a stall and guarding
a failed evaluation instead of raising.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import hessian as jax_hessian
from jax import jit, value_and_grad
from jax.tree_util import tree_flatten_with_path
from numpy.typing import NDArray

from cmad.cli.common import FEProblemBundle, build_fe_trajectory_cost
from cmad.fem.time_refinement import TimeRefinement, refine_schedule
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray


@dataclass(frozen=True)
class Evaluation:
    """What one specimen returned at a trial point: its value and gradient
    on the schedule it ended with, the times inserted on the way, its
    accumulated QoIs by name, and the reason when it failed."""

    J: float
    grad: NDArray[np.float64]
    schedule: NDArray[np.float64]
    inserted: list[float]
    accumulated_qois: dict[str, float]
    failure: str | None = None


class Specimen:
    """One FE problem in an :class:`Objective`.

    ``evaluate`` splits a stalled step and tries again, up to
    ``refinement.max_depth``. It reports a failure and leaves the answer
    to the objective. ``snap_to`` gives the times refinement may insert.
    """

    def __init__(
            self,
            bundle: FEProblemBundle,
            *,
            refinement: TimeRefinement | None = None,
            snap_to: NDArray[np.float64] | None = None,
            weight: float = 1.0,
            print_global_convergence: bool = False,
    ) -> None:
        _params_flat, state_init, cost = build_fe_trajectory_cost(
            bundle, print_global_convergence,
        )
        assert bundle.qoi is not None
        self._accumulated_qoi_names = bundle.qoi.accumulated_qoi_names()
        self._fe_problem = bundle.fe_problem
        self._fe_arrays = bundle.fe_problem.kernel_arrays
        self._state_init = state_init
        self._value = jit(cost)
        self._value_and_grad = jit(value_and_grad(cost, argnums=0, has_aux=True))
        self._hessian = jit(jax_hessian(cost, argnums=0, has_aux=True))
        self._refinement = (
            refinement if refinement is not None else TimeRefinement()
        )
        self._snap_to = snap_to
        self.weight = float(weight)
        self.models: Mapping[str, Model] = bundle.fe_problem.models_by_block
        self._input_schedule: NDArray[np.float64] = np.asarray(
            bundle.t_schedule, dtype=np.float64,
        )
        self.schedule: NDArray[np.float64] = self._input_schedule

    def evaluate(
            self,
            x: NDArray[np.floating],
            *,
            have_accepted: bool,
            label: str,
    ) -> Evaluation:
        """The value and gradient at ``x``, refined as the class docstring
        describes. A non-finite result is not refined once the objective
        has an accepted evaluation. ``label`` opens the printed lines."""
        return self._refined(x, have_accepted, label, with_grad=True)

    def value(
            self,
            x: NDArray[np.floating],
            *,
            have_accepted: bool,
            label: str,
    ) -> Evaluation:
        """As :meth:`evaluate`, the value alone; the returned gradient is
        empty."""
        return self._refined(x, have_accepted, label, with_grad=False)

    def _refined(
            self,
            x: NDArray[np.floating],
            have_accepted: bool,
            label: str,
            with_grad: bool,
    ) -> Evaluation:
        schedule = self.schedule
        inserted: list[float] = []
        for depth in range(self._refinement.max_depth + 1):
            args = (x, self._state_init, self._fe_arrays, jnp.asarray(schedule))
            if with_grad:
                (J, (first_failed, _iters, accumulated_qois)), grad = (
                    self._value_and_grad(*args)
                )
                grad_np = np.asarray(grad, dtype=np.float64)
            else:
                J, (first_failed, _iters, accumulated_qois) = self._value(*args)
                grad_np = np.zeros(0, dtype=np.float64)
            failed = int(first_failed)
            value = float(J)
            finite = np.isfinite(value) and bool(
                np.all(np.isfinite(grad_np)),
            )
            if finite and failed < 0:
                return Evaluation(
                    value, grad_np, schedule, inserted,
                    self._by_name(accumulated_qois),
                )
            if not finite and (have_accepted or failed < 0):
                return Evaluation(
                    value, grad_np, schedule, inserted,
                    self._by_name(accumulated_qois),
                    "the value or gradient is not finite",
                )
            if depth == self._refinement.max_depth:
                break
            schedule, new_times = refine_schedule(
                schedule, failed, self._refinement.factor, self._snap_to,
            )
            for t in new_times:
                self._fe_problem.dof_map.evaluate_prescribed_values(
                    self._fe_arrays.dbc_arrays, float(t),
                )
            inserted.extend(new_times.tolist())
            print(
                f"{label}: step {failed + 1} failed; "
                f"inserted t = {', '.join(f'{t:g}' for t in new_times)}"
            )
        return Evaluation(
            value, grad_np, schedule, inserted,
            self._by_name(accumulated_qois),
            f"step {failed + 1} failed at refinement depth {depth}",
        )

    def _by_name(self, accumulated_qois: JaxArray) -> dict[str, float]:
        return dict(zip(
            self._accumulated_qoi_names,
            np.asarray(accumulated_qois, dtype=np.float64).ravel().tolist(),
            strict=True,
        ))

    def keep(self, evaluation: Evaluation) -> None:
        """Adopt the schedule an accepted evaluation ended with."""
        self.schedule = evaluation.schedule

    @property
    def inserted_times(self) -> NDArray[np.float64]:
        """The times refinement has added to the input file's schedule."""
        return self.schedule[~np.isin(self.schedule, self._input_schedule)]

    def hessian(self, x: NDArray[np.floating]) -> NDArray[np.float64]:
        """The Hessian at ``x`` on the current schedule; raises when a step
        fails or the result is not finite, since a Hessian has no guard."""
        H, (first_failed, _iters, _accumulated_qois) = self._hessian(
            x, self._state_init, self._fe_arrays, jnp.asarray(self.schedule),
        )
        H_np = np.asarray(H, dtype=np.float64)
        if int(first_failed) >= 0 or not bool(np.all(np.isfinite(H_np))):
            raise RuntimeError("the Hessian evaluation failed")
        return H_np

    def set_params(self, x: NDArray[np.floating]) -> None:
        """Store ``x`` into the models: each block's slice, canonical
        inverted to native."""
        _set_active_values(
            {block: m.parameters for block, m in self.models.items()}, x,
        )

    @property
    def param_paths(self) -> list[str]:
        return _param_paths(
            {block: m.parameters for block, m in self.models.items()},
        )


class Objective:
    """Value and gradient of the weighted sum over one or more specimens.

    ``parameters`` is the calibrated parameter tree by element block,
    built from the input file's shared ``materials`` section; the
    optimizer's starting point, bounds, and parameter names come from it,
    and every specimen's models must hold the same active parameters.

    An evaluation is accepted when every specimen succeeds, and the
    inserted times are then kept. When one fails, the rest are not solved,
    every inserted time is discarded, and ``evaluate`` returns the guard
    instead of raising: a value above the last accepted one with a
    gradient pointing back toward it. The first evaluation has no guard to
    fall back on and raises. ``log_params`` records native parameter
    values in the history entries.
    """

    def __init__(
            self,
            specimens: Mapping[str, Specimen],
            parameters: Mapping[str, Parameters],
            *,
            log_params: bool = False,
    ) -> None:
        if not specimens:
            raise ValueError("an objective needs at least one specimen")
        self._specimens = dict(specimens)
        self.parameters = dict(parameters)
        paths = self.param_paths
        for tag, specimen in self._specimens.items():
            if specimen.param_paths != paths:
                raise ValueError(
                    f"specimen '{tag}' has active parameters "
                    f"{specimen.param_paths}; the objective's are {paths}",
                )
        self._log_params = log_params
        self._accepted: dict[str, Any] = {}
        self._evaluations = 0
        self.history: list[dict[str, Any]] = []

    def evaluate(
            self, x: NDArray[np.floating],
    ) -> tuple[float, NDArray[np.float64]]:
        """``(J, grad)`` at ``x``, guarded as the class docstring
        describes."""
        self._evaluations += 1
        several = len(self._specimens) > 1
        done: dict[str, Evaluation] = {}
        for tag, specimen in self._specimens.items():
            label = f"evaluation {self._evaluations}"
            if several:
                label += f", {tag}"
            evaluation = specimen.evaluate(
                x, have_accepted=bool(self._accepted), label=label,
            )
            if evaluation.failure is not None:
                reason = (
                    f"{tag}: {evaluation.failure}" if several
                    else evaluation.failure
                )
                return self._guard(x, reason, {**done, tag: evaluation})
            done[tag] = evaluation
        for tag, evaluation in done.items():
            self._specimens[tag].keep(evaluation)
        value = 0.0
        grad = np.zeros_like(x, dtype=np.float64)
        for tag, evaluation in done.items():
            weight = self._specimens[tag].weight
            value += weight * evaluation.J
            grad += weight * evaluation.grad
        return self._accept(x, value, grad, done)

    def value(self, x: NDArray[np.floating]) -> float:
        """The weighted sum of the specimen values at ``x``, refined as
        ``evaluate`` is and recorded in the history without a gradient
        norm. A failure raises, naming the specimen: a value alone is asked
        for once and has nothing to fall back on."""
        self._evaluations += 1
        several = len(self._specimens) > 1
        done: dict[str, Evaluation] = {}
        for tag, specimen in self._specimens.items():
            label = f"evaluation {self._evaluations}"
            if several:
                label += f", {tag}"
            evaluation = specimen.value(x, have_accepted=False, label=label)
            if evaluation.failure is not None:
                reason = (
                    f"{tag}: {evaluation.failure}" if several
                    else evaluation.failure
                )
                raise RuntimeError(f"the evaluation failed: {reason}")
            done[tag] = evaluation
        for tag, evaluation in done.items():
            self._specimens[tag].keep(evaluation)
        value = sum(
            self._specimens[tag].weight * evaluation.J
            for tag, evaluation in done.items()
        )
        self._record_accepted(x, value, None, done)
        return value

    def hessian(self, x: NDArray[np.floating]) -> NDArray[np.float64]:
        """The weighted sum of the specimen Hessians at ``x``; raises,
        naming the specimen, when one fails."""
        several = len(self._specimens) > 1
        H = np.zeros((x.shape[0], x.shape[0]), dtype=np.float64)
        for tag, specimen in self._specimens.items():
            try:
                H += specimen.weight * specimen.hessian(x)
            except RuntimeError as exc:
                if not several:
                    raise
                raise RuntimeError(f"{tag}: {exc}") from exc
        return H

    def _accept(
            self,
            x: NDArray[np.floating],
            value: float,
            grad: NDArray[np.float64],
            done: Mapping[str, Evaluation],
    ) -> tuple[float, NDArray[np.float64]]:
        self._accepted.update(
            x=np.array(x, dtype=np.float64), J=value, grad=grad,
        )
        self._record_accepted(x, value, grad, done)
        return value, grad

    def _record_accepted(
            self,
            x: NDArray[np.floating],
            value: float,
            grad: NDArray[np.float64] | None,
            done: Mapping[str, Evaluation],
    ) -> None:
        entry: dict[str, Any] = {}
        if len(done) == 1:
            (evaluation,) = done.values()
            if evaluation.inserted:
                entry["refined"] = evaluation.inserted
            self._record(entry, x, value, grad)
            entry["accumulated_qois"] = evaluation.accumulated_qois
        else:
            self._record(entry, x, value, grad)
            specimens: dict[str, Any] = {}
            for tag, evaluation in done.items():
                specimen_entry: dict[str, Any] = {
                    "J": evaluation.J,
                    "accumulated_qois": evaluation.accumulated_qois,
                }
                if evaluation.inserted:
                    specimen_entry["refined"] = evaluation.inserted
                specimens[tag] = specimen_entry
            entry["specimens"] = specimens

    def _guard(
            self,
            x: NDArray[np.floating],
            reason: str,
            done: Mapping[str, Evaluation],
    ) -> tuple[float, NDArray[np.float64]]:
        if not self._accepted:
            raise RuntimeError(f"the first evaluation failed: {reason}")
        value = 2.0 * abs(self._accepted["J"]) + 1.0
        step = np.asarray(x, dtype=np.float64) - self._accepted["x"]
        length = float(np.linalg.norm(step))
        if length > 0.0:
            grad = np.linalg.norm(self._accepted["grad"]) * step / length
        else:
            grad = np.zeros_like(x, dtype=np.float64)
        several = len(self._specimens) > 1
        discarded = [
            (f"{tag}: " if several else "")
            + ", ".join(f"{t:g}" for t in evaluation.inserted)
            for tag, evaluation in done.items() if evaluation.inserted
        ]
        suffix = f"; discarded t = {'; '.join(discarded)}" if discarded else ""
        print(f"guarded evaluation: {reason}{suffix}")
        entry: dict[str, Any] = {"guarded": reason}
        self._record(entry, x, value, grad)
        return value, grad

    def _record(
            self,
            entry: dict[str, Any],
            x: NDArray[np.floating],
            value: float,
            grad: NDArray[np.float64] | None,
    ) -> None:
        entry["J"] = value
        if grad is not None:
            entry["grad_norm"] = float(np.linalg.norm(grad))
        if self._log_params:
            self.set_params(x)
            entry["params"] = self.param_values
        self.history.append(entry)

    def set_params(self, x: NDArray[np.floating]) -> None:
        """Store ``x`` into the objective's parameters and into every
        specimen's models."""
        _set_active_values(self.parameters, x)
        for specimen in self._specimens.values():
            specimen.set_params(x)

    @property
    def x0(self) -> NDArray[np.float64]:
        """The active parameters' canonical values, in block order."""
        return np.concatenate([
            np.asarray(p.flat_active_values(return_canonical=True),
                       dtype=np.float64)
            for p in self.parameters.values()
        ] or [np.zeros(0)])

    @property
    def schedules(self) -> dict[str, NDArray[np.float64]]:
        """Each specimen's current solve schedule, by tag."""
        return {tag: s.schedule for tag, s in self._specimens.items()}

    @property
    def inserted_times(self) -> dict[str, NDArray[np.float64]]:
        """The times refinement added, by tag, for the specimens whose
        schedule grew."""
        return {
            tag: s.inserted_times for tag, s in self._specimens.items()
            if s.inserted_times.size
        }

    @property
    def bounds(self) -> NDArray[np.floating] | None:
        """Per block ``opt_bounds`` concatenated in block order (``None``
        if no block has active parameters)."""
        blocks = [
            p.opt_bounds for p in self.parameters.values()
            if p.num_active_params > 0
        ]
        return np.concatenate(blocks) if blocks else None

    @property
    def param_values(self) -> list[float]:
        """Flat native values of the active parameters, in block order."""
        return [
            float(v) for p in self.parameters.values()
            for v in p.flat_active_values(return_canonical=False)
        ]

    @property
    def param_paths(self) -> list[str]:
        """Block qualified dotted labels for the active parameters, in
        block order (aligned with :attr:`param_values`)."""
        return _param_paths(self.parameters)


def _set_active_values(
        parameters: Mapping[str, Parameters], x: NDArray[np.floating],
) -> None:
    """Store ``x`` into each block's parameters: its slice, canonical
    inverted to native."""
    offset = 0
    for p in parameters.values():
        n = p.num_active_params
        p.set_active_values_from_flat(
            np.asarray(x[offset:offset + n]), are_canonical=True,
        )
        offset += n


def _param_paths(parameters: Mapping[str, Parameters]) -> list[str]:
    return [
        f"{block}.{path}" for block, p in parameters.items()
        for path in active_param_paths(p)
    ]


def active_param_paths(parameters: Parameters) -> list[str]:
    """Dotted-path labels for the active parameters in ``active_idx`` order.

    Path segments come from :func:`jax.tree_util.tree_flatten_with_path`;
    spaces inside segment keys (``"flow stress"``) are replaced with
    underscores so the resulting strings can be used as plain dotted
    identifiers in diagnostic output.
    """
    flat, _ = tree_flatten_with_path(parameters.values)
    all_paths = [_dotted(key_path) for key_path, _ in flat]
    return [all_paths[i] for i in parameters.active_idx]


def _dotted(key_path: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for k in key_path:
        s = str(k.key) if hasattr(k, "key") else str(k)
        parts.append(s.replace(" ", "_"))
    return ".".join(parts)
