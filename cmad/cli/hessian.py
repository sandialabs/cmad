"""Implementation of the ``cmad hessian`` subcommand.

Dispatches on ``problem.type``. The MP branch evaluates
``(J, grad, hess)`` via the sensitivity driver
(``sensitivity.type`` must be ``direct_adjoint`` or ``jvp`` — the
two strategies that actually produce a Hessian; the factory in
:mod:`cmad.cli.sensitivity` enforces this) and writes ``J.json``,
``grad.{npy,csv}``, ``hess.{npy,csv}``, and ``deck.resolved.yaml``.
The FE branch builds the FE problem with the QoI attached,
constructs the ``J(params_flat)`` cost-function closure via
:func:`cmad.cli.common.build_fe_J_of_params_flat`, evaluates
``jax.hessian(J)(params_flat)`` directly — no separate adjoint
driver — and writes ``hess.{npy,csv}`` and ``deck.resolved.yaml``
only (``jax.hessian`` doesn't surface ``J`` or the gradient as a
side effect; run ``cmad objective`` and ``cmad gradient``
separately on the same deck for ``J`` and grad output). Neither
branch writes primal trajectories — run ``cmad primal``
separately on the same deck for those.
"""

from __future__ import annotations

from pathlib import Path

import jax
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

    out_dir, prefix, fmt = resolve_output(graph.resolved)
    write_resolved_deck(out_dir, prefix, graph.resolved)
    write_J(out_dir, prefix, result.J)
    write_grad(out_dir, prefix, result.grad, fmt)
    write_hessian(out_dir, prefix, result.hessian, fmt)
    return 0


def _run_hessian_fe(deck_path: Path) -> int:
    bundle = build_fe_problem_from_deck(deck_path, "hessian")
    params_flat, state_init, J_of_params_flat = build_fe_J_of_params_flat(
        bundle,
    )
    fe_arrays = bundle.fe_problem.kernel_arrays

    hess = np.asarray(
        jax.jit(jax.hessian(J_of_params_flat, argnums=0))(
            params_flat, state_init, fe_arrays,
        ),
    )

    out_dir, prefix, fmt = resolve_output(bundle.resolved)
    write_resolved_deck(out_dir, prefix, bundle.resolved)
    write_hessian(out_dir, prefix, hess, fmt)
    return 0
