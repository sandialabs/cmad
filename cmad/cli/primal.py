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

from cmad.cli.common import (
    build_fe_problem_from_deck,
    build_mp_problem,
    resolve_output,
)
from cmad.fem.driver import fe_quasistatic_drive
from cmad.io.deck import load_deck, unwrap_top_level
from cmad.io.writers import (
    write_cauchy,
    write_fe_exodus,
    write_resolved_deck,
    write_solver_log,
    write_xi,
)
from cmad.models.global_fields import mp_U_from_F
from cmad.qois.qoi import QoI
from cmad.solver.nonlinear_solver import newton_solve
from cmad.typing import SupportsPrimalLoop


def run_primal(deck_path: Path) -> int:
    """Execute the primal subcommand on ``deck_path``. Returns an exit code.

    Dispatches on ``problem.type``: ``material_point`` runs the MP
    forward solve and writes ``cauchy`` / ``xi`` arrays; ``fe`` runs
    the FE forward solve and writes Exodus II results. Both branches
    share the ``solver.json`` and ``deck.resolved.yaml`` writers.
    """
    deck = unwrap_top_level(load_deck(deck_path))
    problem_type = deck["problem"]["type"]
    if problem_type == "material_point":
        return _run_primal_mp(deck_path)
    if problem_type == "fe":
        return _run_primal_fe(deck_path)
    raise ValueError(
        f"unsupported problem.type {problem_type!r}; expected "
        f"'material_point' or 'fe'"
    )


def _run_primal_mp(deck_path: Path) -> int:
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


def _run_primal_fe(deck_path: Path) -> int:
    bundle = build_fe_problem_from_deck(deck_path, "primal")
    gr_section = bundle.resolved["residuals"]["global residual"]
    fe_state, solver_log = fe_quasistatic_drive(
        bundle.fe_problem,
        bundle.t_schedule.tolist(),
        max_iters=int(gr_section["nonlinear max iters"]),
        abs_tol=float(gr_section["nonlinear absolute tol"]),
        rel_tol=float(gr_section["nonlinear relative tol"]),
    )

    out_dir, prefix, fmt = resolve_output(bundle.resolved, deck_path)
    if fmt != "exodus":
        raise ValueError(
            f"output.format must be 'exodus' for FE primal; got {fmt!r}"
        )
    write_fe_exodus(
        out_dir, prefix, bundle.fe_problem, fe_state,
        bundle.resolved["output"],
    )
    write_solver_log(out_dir, prefix, solver_log)
    write_resolved_deck(out_dir, prefix, bundle.resolved)
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
            {"iters": iters, "final_residual": final_res},
        )
        if qoi is not None:
            model.seed_none()
            qoi.evaluate(step)
            J += float(np.asarray(qoi.J()))

    return cauchy, xi_trajectory, solver_log, J
