"""Implementation of the ``cmad objective`` subcommand.

Builds the same object graph as ``cmad primal`` plus the registered
QoI, runs a single forward pass through :func:`cmad.cli.primal.run_primal_pass`
with the QoI supplied so J is accumulated alongside cauchy, xi, and
solver log in one loop. Writes the primal output set plus ``J.json``.
No sensitivities are computed.
"""

from __future__ import annotations

from pathlib import Path

from cmad.cli.common import build_mp_object_graph, resolve_output
from cmad.cli.primal import run_primal_pass
from cmad.io.writers import (
    write_cauchy,
    write_J,
    write_resolved_deck,
    write_solver_log,
    write_xi,
)


def run_objective(deck_path: Path) -> int:
    """Execute the objective subcommand on ``deck_path``. Returns an exit code."""
    graph = build_mp_object_graph(deck_path, "objective")
    qoi = graph.qoi
    assert qoi is not None
    num_steps = graph.F.shape[2] - 1

    newton_kwargs = graph.resolved["solver"]["newton"]
    cauchy, xi_trajectory, solver_log, J = run_primal_pass(
        graph.model, graph.F, num_steps, newton_kwargs, qoi=qoi,
    )

    out_dir, prefix, fmt = resolve_output(graph.resolved, deck_path)
    write_cauchy(out_dir, prefix, cauchy, fmt)
    write_xi(out_dir, prefix, xi_trajectory, fmt)
    write_solver_log(out_dir, prefix, solver_log)
    write_resolved_deck(out_dir, prefix, graph.resolved)
    write_J(out_dir, prefix, J)
    return 0
