"""Implementation of the ``cmad gradient`` subcommand.

Builds the problem (model / parameters / QoI), constructs the
sensitivity driver dictated by ``sensitivity.type``, and evaluates
``(J, grad)`` at the deck's parameter point (canonical coordinates).
Writes ``J.json``, ``grad.{npy,csv}``, and ``deck.resolved.yaml``; no
primal trajectories — the sensitivity evaluator's internal forward
pass is the one Newton solve per step, so recording cauchy / xi /
solver_log would require a second pass. Run ``cmad primal`` separately
on the same deck for those outputs.
"""

from __future__ import annotations

from pathlib import Path

from cmad.cli.common import build_mp_problem, resolve_output
from cmad.cli.sensitivity import build_sensitivity_driver
from cmad.io.deck import load_deck, unwrap_top_level
from cmad.io.writers import write_grad, write_J, write_resolved_deck


def run_gradient(deck_path: Path) -> int:
    """Execute the gradient subcommand on ``deck_path``. Returns an exit code."""
    deck = unwrap_top_level(load_deck(deck_path))
    problem_type = deck["problem"]["type"]
    if problem_type == "fe":
        raise NotImplementedError(
            "cmad gradient with problem.type='fe' is not yet supported"
        )
    if problem_type != "material_point":
        raise ValueError(
            f"unsupported problem.type {problem_type!r}; expected "
            f"'material_point' or 'fe'"
        )

    graph = build_mp_problem(deck_path, "gradient")
    qoi = graph.qoi
    assert qoi is not None

    newton_kwargs = graph.resolved["solver"]["newton"]
    driver = build_sensitivity_driver(
        graph.resolved["sensitivity"], qoi, graph.F, newton_kwargs,
        subcommand="gradient",
    )
    x = graph.parameters.flat_active_values(return_canonical=True)
    result = driver.evaluate_grad(x)

    out_dir, prefix, fmt = resolve_output(graph.resolved, deck_path)
    write_resolved_deck(out_dir, prefix, graph.resolved)
    write_J(out_dir, prefix, result.J)
    write_grad(out_dir, prefix, result.grad, fmt)
    return 0
