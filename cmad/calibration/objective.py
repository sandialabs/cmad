"""Calibration objectives over FE problems.

:class:`Objective` wraps one FE calibration problem so an optimizer
can consume it: ``evaluate`` returns ``(J, grad)``, refining the solve
schedule on a stall and guarding a failed evaluation instead of
raising.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad
from jax.tree_util import tree_flatten_with_path
from numpy.typing import NDArray

from cmad.cli.common import FEProblemBundle, build_fe_trajectory_cost
from cmad.fem.time_refinement import TimeRefinement, refine_schedule
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters


class Objective:
    """Value and gradient of one FE calibration problem's objective.

    ``evaluate`` splits a stalled step and retries, up to
    ``refinement.max_depth``, keeping the inserted times only when the
    evaluation succeeds; a non finite solve skips refinement. A failed
    evaluation returns the guard instead of raising: a value above the
    last accepted one with a gradient pointing back toward it. The
    first evaluation has no guard to fall back on: it refines
    regardless and raises at exhausted depth.

    ``snap_to`` gives the times refinement may insert; ``log_params``
    records native parameter values in the history entries.
    """

    def __init__(
            self,
            bundle: FEProblemBundle,
            *,
            refinement: TimeRefinement | None = None,
            snap_to: NDArray[np.float64] | None = None,
            log_params: bool = False,
    ) -> None:
        params_flat, state_init, cost = build_fe_trajectory_cost(bundle)
        self._fe_problem = bundle.fe_problem
        self._models = bundle.fe_problem.models_by_block
        self._fe_arrays = bundle.fe_problem.kernel_arrays
        self._state_init = state_init
        self._value_and_grad = jit(value_and_grad(cost, argnums=0, has_aux=True))
        self._refinement = (
            refinement if refinement is not None else TimeRefinement()
        )
        self._snap_to = snap_to
        self._log_params = log_params
        self._accepted: dict[str, Any] = {}
        self._evaluations = 0
        self.x0: NDArray[np.float64] = np.asarray(
            params_flat, dtype=np.float64,
        )
        self.schedule: NDArray[np.float64] = np.asarray(
            bundle.t_schedule, dtype=np.float64,
        )
        self.history: list[dict[str, Any]] = []

    def evaluate(
            self, x: NDArray[np.floating],
    ) -> tuple[float, NDArray[np.float64]]:
        """``(J, grad)`` at ``x``, refined or guarded as the class
        docstring describes."""
        self._evaluations += 1
        schedule = self.schedule
        inserted: list[float] = []
        for depth in range(self._refinement.max_depth + 1):
            (J, (first_failed, _iters)), grad = self._value_and_grad(
                x, self._state_init, self._fe_arrays, jnp.asarray(schedule),
            )
            failed = int(first_failed)
            value = float(J)
            grad_np = np.asarray(grad, dtype=np.float64)
            finite = np.isfinite(value) and bool(
                np.all(np.isfinite(grad_np)),
            )
            if finite and failed < 0:
                self.schedule = schedule
                return self._accept(x, value, grad_np, inserted)
            if not finite and (self._accepted or failed < 0):
                return self._guard(
                    x, "the value or gradient is not finite", inserted,
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
                f"evaluation {self._evaluations}: step {failed + 1} failed; "
                f"inserted t = {', '.join(f'{t:g}' for t in new_times)}"
            )
        return self._guard(
            x, f"step {failed + 1} failed at refinement depth {depth}",
            inserted,
        )

    def _accept(
            self,
            x: NDArray[np.floating],
            value: float,
            grad: NDArray[np.float64],
            inserted: list[float],
    ) -> tuple[float, NDArray[np.float64]]:
        self._accepted.update(
            x=np.array(x, dtype=np.float64), J=value, grad=grad,
        )
        entry: dict[str, Any] = {}
        if inserted:
            entry["refined"] = inserted
        self._record(entry, x, value, grad)
        return value, grad

    def _guard(
            self,
            x: NDArray[np.floating],
            reason: str,
            inserted: list[float],
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
        discarded = (
            f"; discarded t = {', '.join(f'{t:g}' for t in inserted)}"
            if inserted else ""
        )
        print(f"guarded evaluation: {reason}{discarded}")
        entry: dict[str, Any] = {"guarded": reason}
        self._record(entry, x, value, grad)
        return value, grad

    def _record(
            self,
            entry: dict[str, Any],
            x: NDArray[np.floating],
            value: float,
            grad: NDArray[np.float64],
    ) -> None:
        entry.update(J=value, grad_norm=float(np.linalg.norm(grad)))
        if self._log_params:
            self.set_params(x)
            entry["params"] = self.param_values
        self.history.append(entry)

    def set_params(self, x: NDArray[np.floating]) -> None:
        """Store ``x`` into the models: each block's slice, canonical
        inverted to native."""
        offset = 0
        for model in self._models.values():
            n = model.parameters.num_active_params
            model.parameters.set_active_values_from_flat(
                np.asarray(x[offset:offset + n]), are_canonical=True,
            )
            offset += n

    @property
    def bounds(self) -> NDArray[np.floating] | None:
        """Per block ``opt_bounds`` concatenated in block order (``None``
        if no block has active parameters)."""
        blocks = [
            m.parameters.opt_bounds for m in self._models.values()
            if m.parameters.num_active_params > 0
        ]
        return np.concatenate(blocks) if blocks else None

    @property
    def param_values(self) -> list[float]:
        """Flat native values of the active parameters, in block order."""
        return [
            float(v) for m in self._models.values()
            for v in m.parameters.flat_active_values(return_canonical=False)
        ]

    @property
    def param_paths(self) -> list[str]:
        """Block qualified dotted labels for the active parameters, in
        block order (aligned with :attr:`param_values`)."""
        return [
            f"{block}.{path}" for block, m in self._models.items()
            for path in active_param_paths(m.parameters)
        ]

    @property
    def models(self) -> Mapping[str, Model]:
        return self._models


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
