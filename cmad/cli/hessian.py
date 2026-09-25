"""Implementation of the ``cmad hessian`` subcommand.

Dispatches on ``problem.type``. The MP branch evaluates
``(J, grad, hess)`` via the sensitivity driver
(``sensitivity.type`` must be ``direct_adjoint`` or ``jvp`` — the
two strategies that actually produce a Hessian; the factory in
:mod:`cmad.cli.sensitivity` enforces this) and writes ``J.json``,
``grad.{npy,csv}``, ``hess.{npy,csv}``, and ``deck.resolved.yaml``.
The FE branch builds the :class:`cmad.calibration.Objective` from the
input file, evaluates its Hessian at the file's parameter point (no
separate adjoint driver), and writes ``hess.{npy,csv}`` and
``deck.resolved.yaml`` only (run ``cmad objective`` and ``cmad
gradient`` separately on the same deck for ``J`` and grad output).
Neither branch writes primal trajectories — run ``cmad primal``
separately on the same deck for those.
"""

from __future__ import annotations

from pathlib import Path

from cmad.calibration import build_objective
from cmad.cli.common import (
    build_mp_problem,
    load_fe_input,
    resolve_output,
)
from cmad.cli.sensitivity import build_sensitivity_driver
from cmad.io.deck import load_deck, unwrap_top_level
from cmad.io.writers import (
    write_grad,
    write_hessian,
    write_J,
    write_resolved_deck,
)


def run_hessian(deck_path: Path) -> int:
    """Execute the hessian subcommand on ``deck_path``. Returns an exit code."""
    deck = unwrap_top_level(load_deck(deck_path))
    problem_type = deck["problem"]["type"]
    if problem_type == "material_point":
        return _run_hessian_mp(deck_path)
    if problem_type == "fe":
        return _run_hessian_fe(deck_path)
    raise ValueError(
        f"unsupported problem.type {problem_type!r}; expected "
        f"'material_point' or 'fe'"
    )


def _run_hessian_mp(deck_path: Path) -> int:
    graph = build_mp_problem(deck_path, "hessian")
    qoi = graph.qoi
    assert qoi is not None

    newton_kwargs = graph.resolved["solver"]["newton"]
    driver = build_sensitivity_driver(
        graph.resolved["sensitivity"], qoi, graph.F, newton_kwargs,
        subcommand="hessian", times=graph.times,
    )
    x = graph.parameters.flat_active_values(return_canonical=True)
    result = driver.evaluate_hess(x)

    out_dir, prefix, fmt = resolve_output(graph.resolved)
    write_resolved_deck(out_dir, prefix, graph.resolved)
    write_J(out_dir, prefix, result.J)
    write_grad(out_dir, prefix, result.grad, fmt)
    write_hessian(out_dir, prefix, result.hessian, fmt)
    return 0


def _run_hessian_fe(deck_path: Path) -> int:
    resolved = load_fe_input(deck_path, "hessian")
    objective = build_objective(resolved)
    hess = objective.hessian(objective.x0)

    out_dir, prefix, fmt = resolve_output(resolved)
    write_resolved_deck(out_dir, prefix, resolved)
    write_hessian(out_dir, prefix, hess, fmt)
    return 0
