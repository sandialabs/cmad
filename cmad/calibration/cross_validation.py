"""Leave-one-out cross validation over the specimens of an input file.

Each fold calibrates on every specimen but one, from the input file's
parameter values, and scores the held out specimen at that optimum: its
accumulated QoIs, unweighted, in their own units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult

from cmad.calibration.build import build_specimens, shared_parameters
from cmad.calibration.objective import Objective
from cmad.calibration.optimize import (
    minimize_objective,
    optimize_status,
    resolve_initial_guess,
)


@dataclass(frozen=True)
class Fold:
    """One fold: the objective calibrated on ``trained_on``, its result,
    each trained specimen's accumulated QoIs at the optimum, and the held
    out specimen's there, ``None`` with the reason when its solve failed."""

    held_out: str
    trained_on: list[str]
    objective: Objective
    result: OptimizeResult
    training: dict[str, dict[str, float]]
    held_out_qois: dict[str, float] | None
    failure: str | None = None


def cross_validate(
        resolved: dict[str, Any], *, log_params: bool = False,
) -> list[Fold]:
    """The folds of a validated multispecimen input file, one per held out
    tag of its ``cross validation`` section, every specimen in turn
    without one. Every specimen starts each fold on its input file
    schedule."""
    specimens = build_specimens(resolved)
    optimizer_section = resolved["optimizer"]
    hold_out = (
        resolved.get("cross validation", {}).get("hold out")
        or list(specimens)
    )
    folds: list[Fold] = []
    for tag in hold_out:
        for specimen in specimens.values():
            specimen.reset_schedule()
        trained_on = [t for t in specimens if t != tag]
        objective = Objective(
            {t: specimens[t] for t in trained_on},
            shared_parameters(resolved),
            log_params=log_params,
        )
        print(f"held out {tag}: calibrating on {', '.join(trained_on)}")
        result = minimize_objective(
            objective,
            algorithm=optimizer_section["algorithm"],
            options=optimizer_section["options"],
            x0=resolve_initial_guess(
                optimizer_section["initial_guess"], objective.x0,
            ),
        )
        objective.set_params(result.x)
        training = objective.accumulated_qois_at(result.x)
        evaluation = specimens[tag].value(
            result.x, have_accepted=False, label=f"held out {tag}",
        )
        folds.append(Fold(
            held_out=tag,
            trained_on=trained_on,
            objective=objective,
            result=result,
            training=training,
            held_out_qois=(
                None if evaluation.failure is not None
                else evaluation.accumulated_qois
            ),
            failure=evaluation.failure,
        ))
    return folds


def cv_score(folds: list[Fold]) -> dict[str, Any]:
    """The mean over the held out specimens of each accumulated QoI, with
    the count of folds that gave values and the tags of those that did
    not."""
    scored = [fold for fold in folds if fold.held_out_qois is not None]
    names: list[str] = []
    for fold in scored:
        assert fold.held_out_qois is not None
        names.extend(n for n in fold.held_out_qois if n not in names)
    score: dict[str, Any] = {"folds": len(scored)}
    failed = [fold.held_out for fold in folds if fold.held_out_qois is None]
    if failed:
        score["failed"] = failed
    for name in names:
        values = [
            fold.held_out_qois[name] for fold in scored
            if fold.held_out_qois is not None and name in fold.held_out_qois
        ]
        score[name] = float(np.mean(values))
    return score


def summarize(folds: list[Fold]) -> dict[str, Any]:
    """The summary written by ``cmad cross_validate``: per fold what it
    trained on, the parameters it reached, its value, the held out
    specimen's accumulated QoIs, and the trained specimens', then the
    score over the folds."""
    summary: dict[str, Any] = {}
    for fold in folds:
        entry: dict[str, Any] = {
            "trained on": list(fold.trained_on),
            "params": dict(zip(
                fold.objective.param_paths, fold.objective.param_values,
                strict=True,
            )),
            "fun": float(fold.result.fun),
            "status": optimize_status(fold.result),
        }
        if fold.held_out_qois is None:
            entry["held out"] = {"failed": fold.failure}
        else:
            entry["held out"] = dict(fold.held_out_qois)
        entry["training"] = {
            tag: dict(qois) for tag, qois in fold.training.items()
        }
        summary[f"held out {fold.held_out}"] = entry
    summary["cv score"] = cv_score(folds)
    return summary
