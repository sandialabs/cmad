"""Implementation of the ``cmad cross_validate`` subcommand.

Takes a multispecimen FE input file, calibrates on every specimen but one
and scores the held out one at that optimum, each specimen in turn or
those the ``cross validation`` section names. Writes
``cross_validation/summary.yaml`` and, per fold, ``held_out_<tag>/`` with
the calibrate outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cmad.calibration import cross_validate, optimize_status, summarize
from cmad.cli.common import load_fe_input, resolve_output
from cmad.io.deck import load_deck, unwrap_top_level
from cmad.io.writers import (
    write_cv_summary,
    write_fe_active_params,
    write_fe_opt_params,
    write_opt_history,
    write_opt_status,
    write_resolved_deck,
)


def run_cross_validate(deck_path: Path) -> int:
    """Execute the cross_validate subcommand on ``deck_path``. Returns an
    exit code."""
    deck = unwrap_top_level(load_deck(deck_path))
    problem_type = deck["problem"]["type"]
    if problem_type != "fe":
        raise ValueError(
            f"unsupported problem.type {problem_type!r}; cross_validate "
            "takes an 'fe' input file with a specimens section"
        )
    resolved = load_fe_input(deck_path, "cross_validate")
    log_params = resolved["optimizer"]["log_params"]
    materials = resolved["residuals"]["local residual"]["materials"]

    folds, data_mean_squares = cross_validate(resolved, log_params=log_params)

    out_dir, prefix, _ = resolve_output(resolved)
    cv_dir = out_dir / "cross_validation"
    cv_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        fold_dir = cv_dir / f"held_out_{fold.held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        objective = fold.objective
        for tag, inserted in objective.inserted_times.items():
            refined_path = fold_dir / f"{prefix}{tag}_refined_times.txt"
            np.savetxt(refined_path, objective.schedules[tag])
            print(f"wrote {refined_path} ({inserted.size} inserted)")
        write_opt_history(
            fold_dir, prefix, objective.history,
            objective.param_paths if log_params else None,
        )
        write_fe_opt_params(
            fold_dir, prefix, materials,
            {block: p.values for block, p in objective.parameters.items()},
        )
        write_fe_active_params(fold_dir, prefix, dict(zip(
            objective.param_paths, objective.param_values, strict=True,
        )))
        write_opt_status(fold_dir, prefix, optimize_status(fold.result))
    write_resolved_deck(cv_dir, prefix, resolved)
    write_cv_summary(cv_dir, prefix, summarize(folds, data_mean_squares))
    return 0
