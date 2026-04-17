"""Implementation of the ``cmad primal`` subcommand.

Wires the deck loader, schema validator, parameters builder, deformation
loader, registry, Newton solver, and output writers into a single
end-to-end forward-solve pipeline. No numerical logic lives here — it is
all delegated to the numerical core.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cmad.io.deck import load_deck
from cmad.io.deformation import load_history
from cmad.io.params_builder import build_parameters
from cmad.io.registry import resolve_model
from cmad.io.schema import validate_deck
from cmad.io.writers import (
    write_cauchy,
    write_resolved_deck,
    write_solver_log,
    write_xi,
)
from cmad.solver.nonlinear_solver import newton_solve
from cmad.typing import SupportsPrimalLoop

_SOLVER_DEFAULTS: dict[str, dict[str, Any]] = {
    "newton": {
        "max_iters": 10,
        "abs_tol": 1e-14,
        "rel_tol": 1e-14,
        "max_ls_evals": 0,
    },
}
_OUTPUT_DEFAULTS: dict[str, Any] = {"prefix": "", "format": "npy"}


def run_primal(deck_path: Path) -> int:
    """Execute the primal subcommand on ``deck_path``. Returns an exit code."""
    deck = load_deck(deck_path)
    # Defaults are applied before validation so a minimal deck (no
    # ``solver:`` section, no ``output.format`` field, etc.) fills in to
    # a valid shape before the required-keys check runs.
    resolved = _apply_defaults(deck)
    validate_deck(resolved, "primal")

    cls = resolve_model(resolved["model"]["name"])
    parameters = build_parameters(resolved["parameters"])
    model = cls.from_deck(resolved["model"], parameters)

    F = load_history(resolved["deformation"], deck_path.parent)
    num_steps = F.shape[2] - 1

    newton_kwargs = resolved["solver"]["newton"]
    cauchy, xi_trajectory, solver_log = _primal_loop(
        model, F, num_steps, newton_kwargs,
    )

    out_dir = Path(resolved["output"]["path"])
    if not out_dir.is_absolute():
        out_dir = deck_path.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = resolved["output"]["prefix"]
    fmt = resolved["output"]["format"]

    write_cauchy(out_dir, prefix, cauchy, fmt)
    write_xi(out_dir, prefix, xi_trajectory, fmt)
    write_solver_log(out_dir, prefix, solver_log)
    write_resolved_deck(out_dir, prefix, resolved)
    return 0


def _primal_loop(
        model: SupportsPrimalLoop,
        F: NDArray[np.floating],
        num_steps: int,
        newton_kwargs: dict[str, Any],
) -> tuple[
    NDArray[np.floating],
    list[list[NDArray[np.floating]]],
    list[dict[str, Any]],
]:
    cauchy = np.zeros((3, 3, num_steps + 1))
    model.set_xi_to_init_vals()
    xi_trajectory: list[list[NDArray[np.floating]]] = [
        [np.asarray(x).copy() for x in model.xi()],
    ]
    solver_log: list[dict[str, Any]] = []

    for step in range(1, num_steps + 1):
        model.gather_global([F[:, :, step]], [F[:, :, step - 1]])
        iters, final_res = newton_solve(model, **newton_kwargs)
        model.advance_xi()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
        xi_trajectory.append([np.asarray(x).copy() for x in model.xi()])
        solver_log.append(
            {"step": step, "iters": iters, "final_residual": final_res},
        )

    return cauchy, xi_trajectory, solver_log


def _apply_defaults(deck: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copy of ``deck`` with schema defaults merged in.

    jsonschema's ``default`` keyword is advisory; the driver applies
    Newton and output defaults manually so ``deck.resolved.yaml``
    reflects the values actually used.
    """
    resolved = copy.deepcopy(deck)
    newton_in = resolved.setdefault("solver", {}).setdefault("newton", {})
    for k, v in _SOLVER_DEFAULTS["newton"].items():
        newton_in.setdefault(k, v)
    output_in = resolved.setdefault("output", {})
    for k, v in _OUTPUT_DEFAULTS.items():
        output_in.setdefault(k, v)
    return resolved
