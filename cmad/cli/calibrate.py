"""Implementation of the ``cmad calibrate`` subcommand.

Dispatches on ``problem.type``. Both branches wrap a deck-resolved cost
function in ``scipy.optimize.minimize`` (first-order: ``fun`` returns
``(J, grad)`` with ``jac=True``) and write ``opt_history.json`` /
``opt_status.json`` plus the resolved deck.

The MP branch drives the sensitivity driver dictated by ``sensitivity.type``
(the dispatcher rejects the Hessian-only ``direct_adjoint`` strategy) and
writes ``opt_params.yaml`` -- the deck ``parameters:`` subtree with optimized
native values.

The FE branch builds a :class:`cmad.calibration.Objective` from the
problem bundle and minimizes it through
:func:`cmad.calibration.minimize_objective`. It writes two parameter
artifacts: ``opt_params.yaml`` (reloadable per-block ``materials:``
subtree, all params) and ``active_params.json`` (a flat
``"<block>.<path>" -> native value`` table of just the calibrated
parameters).

``log_params`` in the ``optimizer:`` section controls whether per-fun-call
native parameter values are recorded in the history trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from cmad.calibration import (
    Objective,
    active_param_paths,
    minimize_objective,
    optimize_status,
)
from cmad.cli.common import (
    build_fe_problem_from_deck,
    build_mp_problem,
    calibration_data_times,
    resolve_output,
)
from cmad.cli.sensitivity import build_sensitivity_driver
from cmad.fem.time_refinement import TimeRefinement
from cmad.io.deck import load_deck, unwrap_top_level
from cmad.io.writers import (
    write_fe_active_params,
    write_fe_opt_params,
    write_opt_history,
    write_opt_params,
    write_opt_status,
    write_resolved_deck,
)


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
    param_paths = (
        active_param_paths(parameters) if log_params else None
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
    write_opt_history(out_dir, prefix, history, param_paths)
    write_opt_params(
        out_dir, prefix, graph.resolved["parameters"], parameters.values,
    )
    write_opt_status(out_dir, prefix, optimize_status(result))
    return 0


def _run_calibrate_fe(deck_path: Path) -> int:
    bundle = build_fe_problem_from_deck(deck_path, "calibrate")
    optimizer_section = bundle.resolved["optimizer"]
    log_params = optimizer_section["log_params"]
    gr_section = bundle.resolved["residuals"]["global residual"]

    objective = Objective(
        bundle,
        refinement=TimeRefinement.from_deck(
            gr_section.get("time refinement"),
        ),
        snap_to=calibration_data_times(bundle.resolved),
        log_params=log_params,
    )
    result = minimize_objective(
        objective,
        algorithm=optimizer_section["algorithm"],
        options=optimizer_section["options"],
        x0=_resolve_initial_guess(
            optimizer_section["initial_guess"], objective.x0,
        ),
    )
    objective.set_params(result.x)
    models = objective.models

    out_dir, prefix, _ = resolve_output(bundle.resolved)
    if objective.schedule.size > bundle.t_schedule.size:
        refined_path = out_dir / f"{prefix}refined_times.txt"
        np.savetxt(refined_path, objective.schedule)
        print(
            f"wrote {refined_path} "
            f"({objective.schedule.size - bundle.t_schedule.size} inserted)"
        )
    materials = bundle.resolved["residuals"]["local residual"]["materials"]
    write_resolved_deck(out_dir, prefix, bundle.resolved)
    write_opt_history(
        out_dir, prefix, objective.history,
        objective.param_paths if log_params else None,
    )
    write_fe_opt_params(
        out_dir, prefix, materials,
        {block: model.parameters.values for block, model in models.items()},
    )
    write_fe_active_params(out_dir, prefix, dict(zip(
        objective.param_paths, objective.param_values, strict=True,
    )))
    write_opt_status(out_dir, prefix, optimize_status(result))
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
