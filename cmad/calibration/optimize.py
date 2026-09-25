"""Optimizer drivers for calibration objectives."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from cmad.calibration.objective import Objective

# scipy reads the method name case insensitively.
_HESSIAN_METHODS = frozenset({
    "NEWTON-CG", "DOGLEG", "TRUST-NCG", "TRUST-KRYLOV", "TRUST-EXACT",
    "TRUST-CONSTR",
})
_BOUNDED_METHODS = frozenset({
    "L-BFGS-B", "TNC", "SLSQP", "POWELL", "NELDER-MEAD", "COBYLA",
    "TRUST-CONSTR",
})


def minimize_objective(
        objective: Objective,
        *,
        algorithm: str,
        options: dict[str, Any],
        x0: NDArray[np.floating] | None = None,
) -> OptimizeResult:
    """``scipy.optimize.minimize`` over ``objective.evaluate`` with
    ``jac=True``, the objective's Hessian for the methods that take one,
    and its bounds for the methods that accept them; ``x0`` defaults to
    the objective's."""
    method = algorithm.upper()
    return minimize(
        objective.evaluate,
        objective.x0 if x0 is None else np.asarray(x0, dtype=np.float64),
        jac=True,
        hess=objective.hessian if method in _HESSIAN_METHODS else None,
        method=algorithm,
        bounds=objective.bounds if method in _BOUNDED_METHODS else None,
        options=options,
    )


def resolve_initial_guess(
        spec: Any, init_from_deck: NDArray[np.floating],
) -> NDArray[np.floating]:
    """``x0`` in canonical coordinates for ``scipy.optimize.minimize``.

    ``"from_deck"`` uses ``init_from_deck`` (the deck's active values already
    taken through the inverse transforms by the caller); an explicit list is
    used verbatim.
    """
    if spec == "from_deck":
        return init_from_deck
    return np.asarray(spec, dtype=np.float64)


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
