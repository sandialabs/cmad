"""Implementation of the ``cmad gradient`` subcommand.

Builds the object graph (model / parameters / QoI), constructs the
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

from cmad.cli.sensitivity import build_sensitivity_driver
from cmad.io.deck import apply_deck_defaults, load_deck
from cmad.io.deformation import load_history
from cmad.io.params_builder import build_parameters
from cmad.io.qoi_data import load_qoi_data
from cmad.io.registry import resolve_model, resolve_qoi
from cmad.io.schema import validate_deck
from cmad.io.writers import write_grad, write_J, write_resolved_deck


def run_gradient(deck_path: Path) -> int:
    """Execute the gradient subcommand on ``deck_path``. Returns an exit code."""
    deck = load_deck(deck_path)
    resolved = apply_deck_defaults(deck)
    validate_deck(resolved, "gradient")

    model_cls = resolve_model(resolved["model"]["name"])
    qoi_cls = resolve_qoi(resolved["qoi"]["name"])

    parameters = build_parameters(resolved["parameters"])
    model = model_cls.from_deck(resolved["model"], parameters)

    F = load_history(
        resolved["deformation"], deck_path.parent,
        expected_ndims=model.ndims,
    )
    data, weight = load_qoi_data(resolved["qoi"], deck_path.parent)
    qoi = qoi_cls.from_deck(resolved["qoi"], model, data, weight)

    newton_kwargs = resolved["solver"]["newton"]
    driver = build_sensitivity_driver(
        resolved["sensitivity"], qoi, F, newton_kwargs,
        subcommand="gradient",
    )
    x = parameters.flat_active_values(return_canonical=True)
    result = driver.evaluate_grad(x)

    out_dir = Path(resolved["output"]["path"])
    if not out_dir.is_absolute():
        out_dir = deck_path.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = resolved["output"]["prefix"]
    fmt = resolved["output"]["format"]

    write_resolved_deck(out_dir, prefix, resolved)
    write_J(out_dir, prefix, result.J)
    write_grad(out_dir, prefix, result.grad, fmt)
    return 0
