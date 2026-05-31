"""Implementation of the ``cmad calibrate`` subcommand.

Dispatches on ``problem.type``. Both branches wrap a deck-resolved cost
function in :func:`scipy.optimize.minimize` (first-order: ``fun`` returns
``(J, grad)`` with ``jac=True``) and write ``opt_history.json`` /
``opt_status.json`` plus the resolved deck.

The MP branch drives the sensitivity driver dictated by ``sensitivity.type``
(the dispatcher rejects the Hessian-only ``direct_adjoint`` strategy) and
writes ``opt_params.yaml`` -- the deck ``parameters:`` subtree with optimized
native values.

The FE branch minimizes ``J(params_flat)`` from
:func:`cmad.cli.common.build_fe_J_of_params_flat` via ``jax.value_and_grad``
(FE has no ``sensitivity`` section; the Newton solve's ``custom_jvp`` supplies
the implicit-step gradient). It writes two parameter artifacts:
``opt_params.yaml`` (re-loadable per-block ``materials:`` subtree, all params)
and ``active_params.json`` (a flat ``"<block>.<path>" -> native value`` table
of just the calibrated parameters).

``log_params`` in the ``optimizer:`` section controls whether per-fun-call
native parameter values are recorded in the history trace.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from jax import jit, value_and_grad
from jax.tree_util import tree_flatten_with_path
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from cmad.cli.common import (
    build_fe_J_of_params_flat,
    build_fe_problem_from_deck,
    build_mp_problem,
    resolve_output,
)
from cmad.cli.sensitivity import build_sensitivity_driver
from cmad.io.deck import load_deck, unwrap_top_level
from cmad.io.writers import (
    write_fe_active_params,
    write_fe_opt_params,
    write_opt_history,
    write_opt_params,
    write_opt_status,
    write_resolved_deck,
)
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters


def run_calibrate(deck_path: Path) -> int:
    """Execute the calibrate subcommand on ``deck_path``. Returns an exit code."""
    deck = unwrap_top_level(load_deck(deck_path))
    problem_type = deck["problem"]["type"]
    if problem_type == "material_point":
        return _run_calibrate_mp(deck_path)
    if problem_type == "fe":
        return _run_calibrate_fe(deck_path)
    raise ValueError(
        f"unsupported problem.type {problem_type!r}; expected "
        f"'material_point' or 'fe'"
    )


def _run_calibrate_mp(deck_path: Path) -> int:
    graph = build_mp_problem(deck_path, "calibrate")
    qoi = graph.qoi
    assert qoi is not None
    parameters = graph.parameters

    newton_kwargs = graph.resolved["solver"]["newton"]
    driver = build_sensitivity_driver(
        graph.resolved["sensitivity"], qoi, graph.F, newton_kwargs,
        subcommand="calibrate",
    )

    optimizer_section = graph.resolved["optimizer"]
    x0 = _resolve_initial_guess(
        optimizer_section["initial_guess"],
        parameters.flat_active_values(return_canonical=True),
    )
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

    out_dir, prefix, _ = resolve_output(graph.resolved)
    write_resolved_deck(out_dir, prefix, graph.resolved)
    write_opt_history(out_dir, prefix, history, active_param_paths)
    write_opt_params(
        out_dir, prefix, graph.resolved["parameters"], parameters.values,
    )
    write_opt_status(out_dir, prefix, _optimize_status(result))
    return 0


def _run_calibrate_fe(deck_path: Path) -> int:
    bundle = build_fe_problem_from_deck(deck_path, "calibrate")
    params_flat, state_init, cost = build_fe_J_of_params_flat(bundle)
    models = bundle.fe_problem.models_by_block
    fe_arrays = bundle.fe_problem.kernel_arrays
    optimizer_section = bundle.resolved["optimizer"]
    log_params = optimizer_section["log_params"]

    def set_block_params(x: NDArray[np.floating]) -> None:
        # Split x by block; each block's Parameters stores its slice,
        # inverting canonical -> native.
        offset = 0
        for model in models.values():
            n = model.parameters.num_active_params
            model.parameters.set_active_values_from_flat(
                x[offset:offset + n], are_canonical=True,
            )
            offset += n

    value_and_grad_fn = jit(value_and_grad(cost, argnums=0))
    history: list[dict[str, Any]] = []

    def fun(x: NDArray[np.floating]) -> tuple[float, NDArray[np.floating]]:
        J, grad = value_and_grad_fn(x, state_init, fe_arrays)
        grad_np = np.asarray(grad, dtype=np.float64)
        entry: dict[str, Any] = {
            "J": float(J),
            "grad_norm": float(np.linalg.norm(grad_np)),
        }
        if log_params:
            set_block_params(x)
            entry["params"] = _fe_active_values(models)
        history.append(entry)
        return float(J), grad_np

    result = minimize(
        fun,
        _resolve_initial_guess(
            optimizer_section["initial_guess"],
            np.asarray(params_flat, dtype=np.float64),
        ),
        jac=True,
        method=optimizer_section["algorithm"],
        bounds=_fe_opt_bounds(models),
        options=optimizer_section["options"],
    )
    set_block_params(result.x)

    out_dir, prefix, _ = resolve_output(bundle.resolved)
    materials = bundle.resolved["residuals"]["local residual"]["materials"]
    write_resolved_deck(out_dir, prefix, bundle.resolved)
    write_opt_history(
        out_dir, prefix, history,
        _fe_active_param_paths(models) if log_params else None,
    )
    write_fe_opt_params(
        out_dir, prefix, materials,
        {block: model.parameters.values for block, model in models.items()},
    )
    write_fe_active_params(out_dir, prefix, dict(zip(
        _fe_active_param_paths(models), _fe_active_values(models),
        strict=True,
    )))
    write_opt_status(out_dir, prefix, _optimize_status(result))
    return 0


def _resolve_initial_guess(
        spec: Any, init_from_deck: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Canonical-coordinate ``x0`` for ``scipy.optimize.minimize``.

    ``"from_deck"`` uses ``init_from_deck`` (the deck's active values already
    taken through the inverse transforms by the caller); an explicit list is
    used verbatim.
    """
    if spec == "from_deck":
        return init_from_deck
    return np.asarray(spec, dtype=np.float64)


def _optimize_status(result: OptimizeResult) -> dict[str, Any]:
    """Status fields general over any ``scipy.optimize.minimize`` result.

    Always emits ``success`` / ``status`` / ``message`` / ``fun``; emits each
    of ``nfev`` / ``njev`` / ``nhev`` / ``nit`` only when the method reports
    it (derivative-free methods omit ``njev``; second-order ones add
    ``nhev``). The canonical optimum ``x`` is omitted -- native values live in
    the parameter outputs.
    """
    status: dict[str, Any] = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "fun": float(result.fun),
    }
    for name in ("nfev", "njev", "nhev", "nit"):
        value = getattr(result, name, None)
        if value is not None:
            status[name] = int(value)
    return status


def _fe_opt_bounds(
        models: Mapping[str, Model],
) -> NDArray[np.floating] | None:
    """Per-block ``opt_bounds`` concatenated in block order (``None`` if no
    block has active parameters)."""
    blocks = [
        m.parameters.opt_bounds for m in models.values()
        if m.parameters.num_active_params > 0
    ]
    return np.concatenate(blocks) if blocks else None


def _fe_active_values(models: Mapping[str, Model]) -> list[float]:
    """Flat native values of the active parameters, in block order."""
    return [
        float(v) for m in models.values()
        for v in m.parameters.flat_active_values(return_canonical=False)
    ]


def _fe_active_param_paths(models: Mapping[str, Model]) -> list[str]:
    """Block-qualified dotted labels for the active parameters, in block order
    (aligned with :func:`_fe_active_values`)."""
    return [
        f"{block}.{path}" for block, m in models.items()
        for path in _active_param_paths(m.parameters)
    ]


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
