"""Implementation of the ``cmad objective`` subcommand.

Dispatches on ``problem.type``. The MP branch builds the MP problem
plus the registered QoI, runs a single forward pass through
:func:`cmad.cli.primal.run_primal_pass` with the QoI supplied so J
is accumulated alongside cauchy, xi, and solver log in one loop;
writes the primal output set plus ``J.json``. The FE branch builds
the :class:`cmad.calibration.Objective` from the input file, evaluates
its value at the file's parameter point, and writes ``J.json`` +
``deck.resolved.yaml``; with a ``specimens`` section ``J.json`` also
holds each specimen's own value. No sensitivities are computed in
either branch; FE state-trajectory output is reserved to ``cmad
primal``.
"""

from __future__ import annotations

from pathlib import Path

from cmad.calibration import build_objective
from cmad.cli.common import (
    build_mp_problem,
    load_fe_input,
    resolve_output,
)
from cmad.cli.primal import run_primal_pass
from cmad.io.deck import load_deck, unwrap_top_level
from cmad.io.writers import (
    write_cauchy,
    write_J,
    write_resolved_deck,
    write_solver_log,
    write_xi,
)


def run_objective(deck_path: Path) -> int:
    """Execute the objective subcommand on ``deck_path``. Returns an exit code."""
    deck = unwrap_top_level(load_deck(deck_path))
    problem_type = deck["problem"]["type"]
    if problem_type == "material_point":
        return _run_objective_mp(deck_path)
    if problem_type == "fe":
        return _run_objective_fe(deck_path)
    raise ValueError(
        f"unsupported problem.type {problem_type!r}; expected "
        f"'material_point' or 'fe'"
    )


def _run_objective_mp(deck_path: Path) -> int:
    graph = build_mp_problem(deck_path, "objective")
    qoi = graph.qoi
    assert qoi is not None
    num_steps = graph.F.shape[2] - 1

    newton_kwargs = graph.resolved["solver"]["newton"]
    cauchy, xi_trajectory, solver_log, J = run_primal_pass(
        graph.model, graph.F, num_steps, newton_kwargs, qoi=qoi,
        times=graph.times,
    )

    out_dir, prefix, fmt = resolve_output(graph.resolved)
    write_cauchy(out_dir, prefix, cauchy, fmt)
    write_xi(out_dir, prefix, xi_trajectory, fmt)
    write_solver_log(out_dir, prefix, solver_log)
    write_resolved_deck(out_dir, prefix, graph.resolved)
    write_J(out_dir, prefix, J)
    return 0


def _run_objective_fe(deck_path: Path) -> int:
    resolved = load_fe_input(deck_path, "objective")
    objective = build_objective(resolved)
    J = objective.value(objective.x0)
    specimens = objective.history[-1].get("specimens")

    out_dir, prefix, _fmt = resolve_output(resolved)
    write_resolved_deck(out_dir, prefix, resolved)
    write_J(
        out_dir, prefix, J,
        None if specimens is None
        else {tag: entry["J"] for tag, entry in specimens.items()},
    )
    return 0
