"""Building a calibration objective from a validated FE input file."""
from __future__ import annotations

from typing import Any

from cmad.calibration.objective import Objective, Specimen
from cmad.cli.common import (
    build_fe_problem_from_sections,
    calibration_data_times,
    specimen_sections,
)
from cmad.fem.time_refinement import TimeRefinement
from cmad.io.params_builder import build_parameters
from cmad.parameters.parameters import Parameters


def build_objective(
        resolved: dict[str, Any],
        *,
        log_params: bool = False,
) -> Objective:
    """The objective from a validated FE input file: one specimen per entry
    of its ``specimens`` section, or one for the whole file, tagged by
    ``problem.name``, or ``specimen`` without one."""
    return Objective(
        build_specimens(resolved), shared_parameters(resolved),
        log_params=log_params,
    )


def build_specimens(resolved: dict[str, Any]) -> dict[str, Specimen]:
    """The specimens of a validated FE input file, by tag."""
    print_global_convergence = bool(
        resolved["residuals"]["global residual"].get("print convergence", False),
    )
    if "specimens" in resolved:
        sections_by_tag = {
            tag: specimen_sections(resolved, tag)
            for tag in resolved["specimens"]
        }
        weights = {
            tag: float(entry.get("weight", 1.0))
            for tag, entry in resolved["specimens"].items()
        }
    else:
        tag = resolved["problem"].get("name") or "specimen"
        sections_by_tag = {tag: resolved}
        weights = {tag: 1.0}
    return {
        tag: Specimen(
            build_fe_problem_from_sections(sections),
            refinement=TimeRefinement.from_deck(
                sections["discretization"].get("time refinement"),
            ),
            snap_to=calibration_data_times(sections),
            weight=weights[tag],
            print_global_convergence=print_global_convergence,
        )
        for tag, sections in sections_by_tag.items()
    }


def shared_parameters(resolved: dict[str, Any]) -> dict[str, Parameters]:
    """The calibrated parameter tree by element block, built from the
    input file's ``materials`` section."""
    materials = resolved["residuals"]["local residual"]["materials"]
    return {block: build_parameters(materials[block]) for block in materials}
