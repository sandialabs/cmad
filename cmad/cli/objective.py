"""Implementation of the ``cmad objective`` subcommand.

Builds the same object graph as ``cmad primal`` plus the registered
QoI, runs a single forward pass through :func:`cmad.cli.primal.run_primal_pass`
with the QoI supplied so J is accumulated alongside cauchy, xi, and
solver log in one loop. Writes the primal output set plus ``J.json``.
No sensitivities are computed.
"""

from __future__ import annotations

from pathlib import Path

from cmad.cli.primal import run_primal_pass
from cmad.io.deck import apply_deck_defaults, load_deck
from cmad.io.deformation import load_history
from cmad.io.params_builder import build_parameters
from cmad.io.qoi_data import load_qoi_data
from cmad.io.registry import resolve_model, resolve_qoi
from cmad.io.schema import validate_deck
from cmad.io.writers import (
    write_cauchy,
    write_J,
    write_resolved_deck,
    write_solver_log,
    write_xi,
)


def run_objective(deck_path: Path) -> int:
    """Execute the objective subcommand on ``deck_path``. Returns an exit code."""
    deck = load_deck(deck_path)
    resolved = apply_deck_defaults(deck)
    validate_deck(resolved, "objective")

    model_cls = resolve_model(resolved["model"]["name"])
    qoi_cls = resolve_qoi(resolved["qoi"]["name"])

    parameters = build_parameters(resolved["parameters"])
    model = model_cls.from_deck(resolved["model"], parameters)

    F = load_history(
        resolved["deformation"], deck_path.parent,
        expected_ndims=model._ndims,
    )
    num_steps = F.shape[2] - 1

    data, weight = load_qoi_data(resolved["qoi"], deck_path.parent)
    qoi = qoi_cls.from_deck(resolved["qoi"], model, F, data, weight)

    newton_kwargs = resolved["solver"]["newton"]
    cauchy, xi_trajectory, solver_log, J = run_primal_pass(
        model, F, num_steps, newton_kwargs, qoi=qoi,
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
    write_J(out_dir, prefix, J)
    return 0
