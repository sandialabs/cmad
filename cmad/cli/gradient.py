"""Implementation of the ``cmad gradient`` subcommand.

Dispatches on ``problem.type``. The MP branch builds the problem
(model / parameters / QoI), constructs the sensitivity driver
dictated by ``sensitivity.type``, evaluates ``(J, grad)`` at the
deck's parameter point (canonical coordinates), and writes
``J.json``, ``grad.{npy,csv}``, and ``deck.resolved.yaml``. The FE
branch builds the FE problem with the QoI attached, constructs the
``J(params_flat)`` cost-function closure via
:func:`cmad.cli.common.build_fe_J_of_params_flat`, evaluates
``jax.grad(J)(params_flat)`` directly — no separate adjoint driver,
since the FE Newton solve's ``custom_jvp`` plumbing already handles
the implicit-step gradient — and writes ``grad.{npy,csv}`` and
``deck.resolved.yaml`` only (``jax.grad`` doesn't surface ``J`` as
a side effect; run ``cmad objective`` separately on the same deck
for ``J``). Neither branch writes primal trajectories: the internal
forward pass is one Newton solve per step, so recording
cauchy / xi / solver_log on the side would require a second pass —
run ``cmad primal`` separately on the same deck for those outputs.
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
from cmad.io.writers import write_grad, write_J, write_resolved_deck


def run_gradient(deck_path: Path) -> int:
    """Execute the gradient subcommand on ``deck_path``. Returns an exit code."""
    deck = unwrap_top_level(load_deck(deck_path))
    problem_type = deck["problem"]["type"]
    if problem_type == "material_point":
        return _run_gradient_mp(deck_path)
    if problem_type == "fe":
        return _run_gradient_fe(deck_path)
    raise ValueError(
        f"unsupported problem.type {problem_type!r}; expected "
        f"'material_point' or 'fe'"
    )


def _run_gradient_mp(deck_path: Path) -> int:
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

    out_dir, prefix, fmt = resolve_output(graph.resolved)
    write_resolved_deck(out_dir, prefix, graph.resolved)
    write_J(out_dir, prefix, result.J)
    write_grad(out_dir, prefix, result.grad, fmt)
    return 0


def _run_gradient_fe(deck_path: Path) -> int:
    bundle = build_fe_problem_from_deck(deck_path, "gradient")
    params_flat, state_init, J_of_params_flat = build_fe_J_of_params_flat(
        bundle,
    )
    fe_arrays = bundle.fe_problem.kernel_arrays

    grad = np.asarray(
        jax.jit(jax.grad(J_of_params_flat, argnums=0))(
            params_flat, state_init, fe_arrays,
        ),
    )

    out_dir, prefix, fmt = resolve_output(bundle.resolved)
    write_resolved_deck(out_dir, prefix, bundle.resolved)
    write_grad(out_dir, prefix, grad, fmt)
    return 0
