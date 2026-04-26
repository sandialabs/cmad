"""Implementation of the ``cmad hessian`` subcommand.

Mirrors ``cmad gradient`` but evaluates ``(J, grad, hess)`` via the
sensitivity driver. ``sensitivity.type`` must be ``direct_adjoint`` or
``jvp`` — the two strategies that actually produce a Hessian; the
factory in :mod:`cmad.cli.sensitivity` enforces this. Writes
``J.json``, ``grad.{npy,csv}``, ``hess.{npy,csv}``, and
``deck.resolved.yaml``. No primal trajectories; run ``cmad primal`` on
the same deck for those.
"""

from __future__ import annotations

from pathlib import Path

from cmad.cli.common import build_mp_problem, resolve_output
from cmad.cli.sensitivity import build_sensitivity_driver
from cmad.io.writers import (
    write_grad,
    write_hessian,
    write_J,
    write_resolved_deck,
)


def run_hessian(deck_path: Path) -> int:
    """Execute the hessian subcommand on ``deck_path``. Returns an exit code."""
    graph = build_mp_problem(deck_path, "hessian")
    qoi = graph.qoi
    assert qoi is not None

    newton_kwargs = graph.resolved["solver"]["newton"]
    driver = build_sensitivity_driver(
        graph.resolved["sensitivity"], qoi, graph.F, newton_kwargs,
        subcommand="hessian",
    )
    x = graph.parameters.flat_active_values(return_canonical=True)
    result = driver.evaluate_hess(x)

    out_dir, prefix, fmt = resolve_output(graph.resolved, deck_path)
    write_resolved_deck(out_dir, prefix, graph.resolved)
    write_J(out_dir, prefix, result.J)
    write_grad(out_dir, prefix, result.grad, fmt)
    write_hessian(out_dir, prefix, result.hessian, fmt)
    return 0
