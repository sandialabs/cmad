"""Optimizer drivers for calibration objectives."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from cmad.calibration.objective import Objective


def minimize_objective(
        objective: Objective,
        *,
        algorithm: str,
        options: dict[str, Any],
        x0: NDArray[np.floating] | None = None,
) -> OptimizeResult:
    """``scipy.optimize.minimize`` over ``objective.evaluate`` with
    ``jac=True`` and the objective's bounds; ``x0`` defaults to the
    objective's."""
    return minimize(
        objective.evaluate,
        objective.x0 if x0 is None else np.asarray(x0, dtype=np.float64),
        jac=True,
        method=algorithm,
        bounds=objective.bounds,
        options=options,
    )


def optimize_status(result: OptimizeResult) -> dict[str, Any]:
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
