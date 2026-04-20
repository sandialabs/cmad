"""Implementation of the ``cmad calibrate`` subcommand.

Wraps the deck-resolved sensitivity driver in :func:`scipy.optimize.minimize`
and writes the three structured outputs (``opt_history.json``,
``opt_params.yaml``, ``opt_status.json``) plus the resolved deck.

Only first-order scipy methods are wired: ``fun`` returns ``(J, grad)``
with ``jac=True``. The sensitivity dispatcher rejects
``sensitivity.type=direct_adjoint`` here at factory time (Hessian-only
strategy, nothing to consume it). ``log_params`` in the ``optimizer:``
section controls whether per-fun-call raw parameter values are included
in the history trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from jax.tree_util import tree_flatten_with_path
from numpy.typing import NDArray
from scipy.optimize import minimize

from cmad.cli.common import build_mp_object_graph, resolve_output
from cmad.cli.sensitivity import build_sensitivity_driver
from cmad.io.writers import (
    write_opt_history,
    write_opt_params,
    write_opt_status,
    write_resolved_deck,
)
from cmad.parameters.parameters import Parameters


def run_calibrate(deck_path: Path) -> int:
    """Execute the calibrate subcommand on ``deck_path``. Returns an exit code."""
    graph = build_mp_object_graph(deck_path, "calibrate")
    qoi = graph.qoi
    assert qoi is not None
    parameters = graph.parameters

    newton_kwargs = graph.resolved["solver"]["newton"]
    driver = build_sensitivity_driver(
        graph.resolved["sensitivity"], qoi, graph.F, newton_kwargs,
        subcommand="calibrate",
    )

    optimizer_section = graph.resolved["optimizer"]
    x0 = _resolve_initial_guess(optimizer_section["initial_guess"], parameters)
    bounds = parameters.opt_bounds
    log_params = optimizer_section["log_params"]

    history: list[dict[str, Any]] = []
    active_param_paths = (
        _active_param_paths(parameters) if log_params else None
    )

    def fun(x: NDArray[np.floating]) -> tuple[float, NDArray[np.floating]]:
        r = driver.evaluate_grad(x)
        entry: dict[str, Any] = {
            "J": float(r.J),
            "grad_norm": float(np.linalg.norm(r.grad)),
        }
        if log_params:
            entry["params"] = parameters.flat_active_values(
                return_canonical=False,
            ).tolist()
        history.append(entry)
        return r.J, r.grad

    result = minimize(
        fun, x0, jac=True,
        method=optimizer_section["algorithm"],
        bounds=bounds,
        options=optimizer_section["options"],
    )

    parameters.set_active_values_from_flat(result.x, are_canonical=True)

    out_dir, prefix, _ = resolve_output(graph.resolved, deck_path)
    write_resolved_deck(out_dir, prefix, graph.resolved)
    write_opt_history(out_dir, prefix, history, active_param_paths)
    write_opt_params(
        out_dir, prefix, graph.resolved["parameters"], parameters.values,
    )
    write_opt_status(out_dir, prefix, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "fun": float(result.fun),
        "nfev": int(result.nfev),
        "njev": int(result.njev),
        "nit": int(result.nit),
    })
    return 0


def _resolve_initial_guess(
        spec: Any, parameters: Parameters,
) -> NDArray[np.floating]:
    """Canonical-coordinate ``x0`` for ``scipy.optimize.minimize``.

    ``from_deck`` takes the deck's parameter values through the inverse
    transforms; an explicit list is used verbatim.
    """
    if spec == "from_deck":
        return parameters.flat_active_values(return_canonical=True)
    return np.asarray(spec, dtype=np.float64)


def _active_param_paths(parameters: Parameters) -> list[str]:
    """Dotted-path labels for the active parameters in ``active_idx`` order.

    Path segments come from :func:`jax.tree_util.tree_flatten_with_path`;
    spaces inside segment keys (``"flow stress"``) are replaced with
    underscores so the resulting strings can be used as plain dotted
    identifiers in diagnostic output.
    """
    flat, _ = tree_flatten_with_path(parameters.values)
    all_paths = [_dotted(key_path) for key_path, _ in flat]
    return [all_paths[i] for i in parameters.active_idx]


def _dotted(key_path: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for k in key_path:
        s = str(k.key) if hasattr(k, "key") else str(k)
        parts.append(s.replace(" ", "_"))
    return ".".join(parts)
