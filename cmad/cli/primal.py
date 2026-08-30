"""Implementation of the ``cmad primal`` subcommand.

Wires the deck loader, schema validator, parameters builder, deformation
loader, registry, Newton solver, and output writers into a single
end-to-end forward-solve pipeline. No numerical logic lives here — it is
all delegated to the numerical core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cmad.cli.common import (
    build_fe_problem_from_deck,
    build_mp_problem,
    calibration_data_times,
    nonlinear_solver_settings,
    resolve_output,
)
from cmad.fem.driver import fe_quasistatic_drive
from cmad.fem.time_refinement import TimeRefinement, refine_schedule
from cmad.io.deck import load_deck, unwrap_top_level
from cmad.io.point_cloud import read_point_cloud, write_point_cloud
from cmad.io.writers import (
    resolve_fe_output_plan,
    write_cauchy,
    write_fe_exodus,
    write_J,
    write_resolved_deck,
    write_solver_log,
    write_xi,
)
from cmad.models.global_fields import mp_U_from_F
from cmad.models.nonlinear_solver import newton_solve
from cmad.qois.qoi import QoI
from cmad.remap.locate import sample_fe_displacement_cloud
from cmad.typing import SupportsPrimalLoop


def run_primal(deck_path: Path) -> int:
    """Execute the primal subcommand on ``deck_path``. Returns an exit code.

    Dispatches on ``problem.type``: ``material_point`` runs the MP
    forward solve and writes ``cauchy`` / ``xi`` arrays plus
    ``solver.json``; ``fe`` runs the FE forward solve and writes the
    Exodus II trajectory. Both branches write ``deck.resolved.yaml``.
    The FE branch additionally writes ``J.json`` when the deck supplies
    an optional ``qoi`` section, threaded through
    :func:`cmad.fem.driver.fe_quasistatic_drive`, and a synthetic DIC point
    cloud when the ``output`` section requests one (the solved displacement
    sampled at measurement points on a sideset).
    """
    deck = unwrap_top_level(load_deck(deck_path))
    problem_type = deck["problem"]["type"]
    if problem_type == "material_point":
        return _run_primal_mp(deck_path)
    if problem_type == "fe":
        return _run_primal_fe(deck_path)
    raise ValueError(
        f"unsupported problem.type {problem_type!r}; expected "
        f"'material_point' or 'fe'"
    )


def _run_primal_mp(deck_path: Path) -> int:
    graph = build_mp_problem(deck_path, "primal")
    num_steps = graph.F.shape[2] - 1

    newton_kwargs = graph.resolved["solver"]["newton"]
    cauchy, xi_trajectory, solver_log, _ = run_primal_pass(
        graph.model, graph.F, num_steps, newton_kwargs,
    )

    if "output" in graph.resolved:
        out_dir, prefix, fmt = resolve_output(graph.resolved)
        write_cauchy(out_dir, prefix, cauchy, fmt)
        write_xi(out_dir, prefix, xi_trajectory, fmt)
        write_solver_log(out_dir, prefix, solver_log)
        write_resolved_deck(out_dir, prefix, graph.resolved)
    return 0


def _run_primal_fe(deck_path: Path) -> int:
    bundle = build_fe_problem_from_deck(deck_path, "primal")
    gr_section = bundle.resolved["residuals"]["global residual"]
    solver_settings = nonlinear_solver_settings(
        gr_section, bool(gr_section.get("print convergence", False)),
    )
    linear_solver_settings = bundle.resolved["linear solver"]
    qoi = bundle.qoi
    write_qoi = (
        qoi if qoi is not None and qoi.produces_primal_output() else None
    )
    # A failed step is cut back and the schedule driven again, the cuts
    # moved onto measured frames when the field data is an archive.
    refinement = TimeRefinement.from_deck(gr_section.get("time refinement"))
    snap_to = calibration_data_times(bundle.resolved)
    schedule = np.asarray(bundle.t_schedule, dtype=np.float64)
    inserted_times: list[float] = []
    for depth in range(refinement.max_depth + 1):
        fe_state, J, status = fe_quasistatic_drive(
            bundle.fe_problem,
            schedule.tolist(),
            nonlinear_solver_settings=solver_settings,
            linear_solver_settings=linear_solver_settings,
            qoi=None if write_qoi is not None else qoi,
        )
        if status.converged or depth == refinement.max_depth:
            break
        schedule, inserted = refine_schedule(
            schedule, status.first_failed_step, refinement.factor, snap_to,
        )
        inserted_times.extend(inserted.tolist())
        print(
            f"time refinement: step {status.first_failed_step + 1} failed; "
            f"inserted t = {', '.join(f'{t:g}' for t in inserted)}"
        )
    if not status.converged:
        raise RuntimeError(status.failure_message())

    if "output" not in bundle.resolved:
        return 0

    out_dir, prefix, _fmt = resolve_output(bundle.resolved)
    if inserted_times:
        refined_path = out_dir / f"{prefix}refined_times.txt"
        np.savetxt(refined_path, schedule)
        print(f"wrote {refined_path} ({len(inserted_times)} inserted)")
    output_section = bundle.resolved["output"]
    if output_section.get("write exodus", True):
        output_plan = resolve_fe_output_plan(
            output_section, bundle.fe_problem,
        )
        if "exodus filename" not in output_section:
            name = bundle.resolved["problem"].get("name") or deck_path.stem
            output_section["exodus filename"] = f"{name}.exo"
        write_fe_exodus(
            out_dir, prefix, bundle.fe_problem, fe_state,
            output_plan, output_section["exodus filename"],
        )
    if "dic cloud" in output_section:
        dic_section = output_section["dic cloud"]
        point_coords = read_point_cloud(dic_section["points file"]).coords
        cloud = sample_fe_displacement_cloud(
            bundle.fe_problem, fe_state, point_coords,
            dic_section["sideset"],
        )
        write_point_cloud(out_dir / dic_section["output file"], cloud)
    write_resolved_deck(out_dir, prefix, bundle.resolved)
    if write_qoi is not None:
        write_qoi.write_primal_outputs(bundle.fe_problem, fe_state)
    elif qoi is not None:
        write_J(out_dir, prefix, float(J))
    return 0


def run_primal_pass(
        model: SupportsPrimalLoop,
        F: NDArray[np.floating],
        num_steps: int,
        newton_kwargs: dict[str, Any],
        qoi: QoI | None = None,
) -> tuple[
    NDArray[np.floating],
    list[list[NDArray[np.floating]]],
    list[dict[str, Any]],
    float,
]:
    """Run a forward pass and return ``(cauchy, xi_trajectory, solver_log, J)``.

    One primal time-step loop with stress and state-variable recording,
    optionally accumulating the scalar QoI value ``J`` when ``qoi`` is
    supplied. Without a QoI the returned ``J`` is ``0.0``. Callable by
    any subcommand that needs primal outputs; the optional-QoI path is
    what ``cmad objective`` uses to get J alongside cauchy/xi/solver_log
    in a single forward pass.
    """
    cauchy = np.zeros((3, 3, num_steps + 1))
    model.set_xi_to_init_vals()
    xi_trajectory: list[list[NDArray[np.floating]]] = [
        [np.asarray(x).copy() for x in model.xi()],
    ]
    solver_log: list[dict[str, Any]] = []
    J = 0.0

    for step in range(1, num_steps + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step]),
            mp_U_from_F(F[:, :, step - 1]),
        )
        iters, final_res = newton_solve(model, **newton_kwargs)
        model.advance_xi()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
        xi_trajectory.append([np.asarray(x).copy() for x in model.xi()])
        solver_log.append(
            {"iters": iters, "final_residual": final_res},
        )
        if qoi is not None:
            model.seed_none()
            qoi.evaluate(step)
            J += float(np.asarray(qoi.J()))

    return cauchy, xi_trajectory, solver_log, J
