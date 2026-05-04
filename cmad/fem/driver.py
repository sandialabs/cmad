"""Quasi-static time-loop driver populating an :class:`FEState`."""
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.global_residuals.modes import GlobalResidualMode


def fe_quasistatic_drive(
        fe_problem: FEProblem,
        t_schedule: Sequence[float],
        U_init: NDArray[np.floating] | None = None,
        **solver_kwargs: Any,
) -> tuple[FEState, list[dict[str, Any]]]:
    """Run a quasi-static time loop and return populated state + log.

    ``t_schedule[0]`` is the initial time and ``t_schedule[1:]`` are
    the step times; the schedule must have at least two entries (an
    initial-only schedule is :meth:`FEState.from_problem` and doesn't
    need the driver). The driver seeds an :class:`FEState` at
    ``t_schedule[0]`` (zeros U or the user-supplied ``U_init``), and
    for each subsequent ``t`` solves one global Newton step via
    :func:`cmad.fem.nonlinear_solver.fe_newton_solve`. The previous
    step's converged ``U`` flows into ``U_prev`` (which is also the
    Newton's starting iterate after the new step's prescribed BC
    values are overlaid); the previous step's converged ``xi`` flows
    into ``xi_prev_by_block`` for COUPLED blocks (the per-IP local
    Newton consumes ``xi_prev`` as both initial guess and the
    previous-step state input for plasticity path continuity).

    The appended-per-step ``xi_by_block`` payload is the union of
    COUPLED and CLOSED_FORM entries: COUPLED keys come from the
    Newton's returned ``xi_solved_by_block``; CLOSED_FORM keys echo
    the previous step's xi (no local Newton runs, so the entry is
    the init-tile from :meth:`FEState.from_problem`). This keeps
    every block's xi-history list aligned at length
    ``len(t_schedule)``.

    ``solver_kwargs`` are forwarded to ``fe_newton_solve`` (accepts
    ``max_iters``, ``abs_tol``, ``rel_tol``); the orchestrator threads
    deck-supplied global-Newton settings through this slot.

    Returns ``(state, solver_log)``. ``solver_log`` holds one entry
    per non-initial step keyed ``{"iters": int, "final_residual":
    float}`` — the step index is the entry's position in the list
    (with a +1 offset against ``t_schedule``).
    """
    if len(t_schedule) < 2:
        raise ValueError(
            f"fe_quasistatic_drive requires t_schedule to have at least "
            f"two entries (initial time + at least one step); got "
            f"length {len(t_schedule)}"
        )

    state = FEState.from_problem(
        fe_problem, t_init=t_schedule[0], U_init=U_init,
    )

    coupled_blocks = [
        b for b, m in fe_problem.modes_by_block.items()
        if m == GlobalResidualMode.COUPLED
    ]

    solver_log: list[dict[str, Any]] = []

    for t in t_schedule[1:]:
        prev_idx = state.step_idx
        U_prev = state.U_at(prev_idx)
        xi_prev_by_block = {
            b: state.xi_at(prev_idx, b) for b in coupled_blocks
        }
        U_solved, xi_solved_by_block, iters, final_R_norm = fe_newton_solve(
            fe_problem,
            U_prev=U_prev,
            t=t,
            xi_prev_by_block=xi_prev_by_block,
            **solver_kwargs,
        )
        xi_by_block = {
            b: (
                xi_solved_by_block[b]
                if m == GlobalResidualMode.COUPLED
                else state.xi_at(prev_idx, b)
            )
            for b, m in fe_problem.modes_by_block.items()
        }
        state.append(U_solved, xi_by_block, t_new=t)
        solver_log.append(
            {"iters": iters, "final_residual": final_R_norm},
        )

    return state, solver_log
