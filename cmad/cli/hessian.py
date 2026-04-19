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

from cmad.cli.sensitivity import build_sensitivity_driver
from cmad.io.deck import apply_deck_defaults, load_deck
from cmad.io.deformation import load_history
from cmad.io.params_builder import build_parameters
from cmad.io.qoi_data import load_qoi_data
from cmad.io.registry import resolve_model, resolve_qoi
from cmad.io.schema import validate_deck
from cmad.io.writers import (
    write_grad,
    write_hessian,
    write_J,
    write_resolved_deck,
)


def run_hessian(deck_path: Path) -> int:
    """Execute the hessian subcommand on ``deck_path``. Returns an exit code."""
    deck = load_deck(deck_path)
    resolved = apply_deck_defaults(deck)
    validate_deck(resolved, "hessian")

    model_cls = resolve_model(resolved["model"]["name"])
    qoi_cls = resolve_qoi(resolved["qoi"]["name"])

    parameters = build_parameters(resolved["parameters"])
    model = model_cls.from_deck(resolved["model"], parameters)

    F = load_history(
        resolved["deformation"], deck_path.parent,
        expected_ndims=model._ndims,
    )
    data, weight = load_qoi_data(resolved["qoi"], deck_path.parent)
    qoi = qoi_cls.from_deck(resolved["qoi"], model, data, weight)

    newton_kwargs = resolved["solver"]["newton"]
    driver = build_sensitivity_driver(
        resolved["sensitivity"], qoi, F, newton_kwargs,
        subcommand="hessian",
    )
    x = parameters.flat_active_values(return_canonical=True)
    result = driver.evaluate_hess(x)

    out_dir = Path(resolved["output"]["path"])
    if not out_dir.is_absolute():
        out_dir = deck_path.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = resolved["output"]["prefix"]
    fmt = resolved["output"]["format"]

    write_resolved_deck(out_dir, prefix, resolved)
    write_J(out_dir, prefix, result.J)
    write_grad(out_dir, prefix, result.grad, fmt)
    write_hessian(out_dir, prefix, result.hessian, fmt)
    return 0
