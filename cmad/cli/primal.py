"""Implementation of the ``cmad primal`` subcommand.

Wires the deck loader, schema validator, parameters builder, deformation
loader, registry, Newton solver, and output writers into a single
end-to-end forward-solve pipeline. No numerical logic lives here — it is
all delegated to the numerical core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cmad.cli.common import build_mp_problem, resolve_output
from cmad.io.writers import (
    write_cauchy,
    write_resolved_deck,
    write_solver_log,
    write_xi,
)
from cmad.models.global_fields import mp_U_from_F
from cmad.qois.qoi import QoI
from cmad.solver.nonlinear_solver import newton_solve
from cmad.typing import SupportsPrimalLoop


def run_primal(deck_path: Path) -> int:
    """Execute the primal subcommand on ``deck_path``. Returns an exit code."""
    graph = build_mp_problem(deck_path, "primal")
    num_steps = graph.F.shape[2] - 1

    newton_kwargs = graph.resolved["solver"]["newton"]
    cauchy, xi_trajectory, solver_log, _ = run_primal_pass(
        graph.model, graph.F, num_steps, newton_kwargs,
    )

    out_dir, prefix, fmt = resolve_output(graph.resolved, deck_path)
    write_cauchy(out_dir, prefix, cauchy, fmt)
    write_xi(out_dir, prefix, xi_trajectory, fmt)
    write_solver_log(out_dir, prefix, solver_log)
    write_resolved_deck(out_dir, prefix, graph.resolved)
    return 0


def run_primal_pass(
        model: SupportsPrimalLoop,
        F: NDArray[np.floating],
        num_steps: int,
        newton_kwargs: dict[str, Any],
        qoi: QoI | None = None,
) -> tuple[
    NDArray[np.floating],
    list[list[NDArray[np.floating]]],
    list[dict[str, Any]],
    float,
]:
    """Run a forward pass and return ``(cauchy, xi_trajectory, solver_log, J)``.

    One primal time-step loop with stress and state-variable recording,
    optionally accumulating the scalar QoI value ``J`` when ``qoi`` is
    supplied. Without a QoI the returned ``J`` is ``0.0``. Callable by
    any subcommand that needs primal outputs; the optional-QoI path is
    what ``cmad objective`` uses to get J alongside cauchy/xi/solver_log
    in a single forward pass.
    """
    cauchy = np.zeros((3, 3, num_steps + 1))
    model.set_xi_to_init_vals()
    xi_trajectory: list[list[NDArray[np.floating]]] = [
        [np.asarray(x).copy() for x in model.xi()],
    ]
    solver_log: list[dict[str, Any]] = []
    J = 0.0

    for step in range(1, num_steps + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step]),
            mp_U_from_F(F[:, :, step - 1]),
        )
        iters, final_res = newton_solve(model, **newton_kwargs)
        model.advance_xi()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
        xi_trajectory.append([np.asarray(x).copy() for x in model.xi()])
        solver_log.append(
            {"step": step, "iters": iters, "final_residual": final_res},
        )
        if qoi is not None:
            model.seed_none()
            qoi.evaluate(step)
            J += float(np.asarray(qoi.J()))

    return cauchy, xi_trajectory, solver_log, J
