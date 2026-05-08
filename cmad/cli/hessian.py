"""Implementation of the ``cmad hessian`` subcommand.

Dispatches on ``problem.type``. The MP branch evaluates
``(J, grad, hess)`` via the sensitivity driver
(``sensitivity.type`` must be ``direct_adjoint`` or ``jvp`` — the
two strategies that actually produce a Hessian; the factory in
:mod:`cmad.cli.sensitivity` enforces this) and writes
``J.json``, ``grad.{npy,csv}``, and ``hess.{npy,csv}``. The FE
branch builds the FE problem with the QoI attached, constructs
the ``J(params_flat)`` cost-function closure via
:func:`cmad.cli.common.build_fe_J_of_params_flat`, evaluates
``J`` and ``jax.hessian(J)(params_flat)`` directly — no separate
adjoint driver — and writes ``J.json`` and ``hess.{npy,csv}``.
Both branches write ``deck.resolved.yaml``; neither writes
primal trajectories (run ``cmad primal`` separately on the same
deck for those). Run ``cmad gradient`` separately for grad
output.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cmad.cli.common import (
    build_fe_J_of_params_flat,
    build_fe_problem_from_deck,
    build_mp_problem,
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


def _run_hessian_fe(deck_path: Path) -> int:
    bundle = build_fe_problem_from_deck(deck_path, "hessian")
    params_flat_init, J_of_params_flat = build_fe_J_of_params_flat(bundle)
    params_flat = jnp.asarray(params_flat_init, dtype=jnp.float64)

    J = float(J_of_params_flat(params_flat))
    hess = np.asarray(jax.hessian(J_of_params_flat)(params_flat))

    out_dir, prefix, fmt = resolve_output(bundle.resolved, deck_path)
    write_resolved_deck(out_dir, prefix, bundle.resolved)
    write_J(out_dir, prefix, J)
    write_hessian(out_dir, prefix, hess, fmt)
    return 0
