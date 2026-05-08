"""Quasi-static time-loop driver populating an :class:`FEState`.

Two layers:

- :func:`fe_quasistatic_drive_traced` runs :func:`jax.lax.scan` over
  the time schedule with all-JAX carry / output. Each scan step calls
  :func:`cmad.fem.nonlinear_solver.fe_newton_solve` (which is itself
  JAX-traceable end to end). The traced inner is the AD entry point
  for callers that want :func:`jax.grad` over a quantity-of-interest
  defined on the trajectory.

- :func:`fe_quasistatic_drive` is the imperative wrapper used by
  :command:`cmad primal` and tests: builds an :class:`FEState`,
  drives :func:`fe_quasistatic_drive_traced`, and materializes the
  stacked outputs back into FEState's mutable per-step lists.
"""
from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import debug, lax
from numpy.typing import NDArray

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.typing import JaxArray, Params


def fe_quasistatic_drive_traced(
        fe_problem: FEProblem,
        params_by_block: Mapping[str, Params],
        U_init: JaxArray,
        xi_init_by_block: dict[str, JaxArray],
        t_schedule_jax: JaxArray,
        max_iters: int = 20,
        abs_tol: float = 1e-10,
        rel_tol: float = 1e-10,
        print_global_convergence: bool = False,
) -> tuple[JaxArray, dict[str, JaxArray]]:
    """Run :func:`jax.lax.scan` over ``t_schedule_jax[1:]``.

    Carry: ``(U_prev, xi_prev_by_block)`` — ``xi_prev_by_block`` is
    the all-blocks dict (COUPLED + CLOSED_FORM). The CLOSED_FORM
    entries don't change per step, but keeping them in the carry
    stabilizes the pytree shape across iterations and avoids a
    per-iteration filter / merge in the step body.

    Per-step output: ``(U_solved, xi_solved_full)``.
    ``xi_solved_full`` is the merge ``{**xi_prev, **xi_solved_coupled}``
    so COUPLED keys carry the converged xi from
    :func:`fe_newton_solve` and CLOSED_FORM keys echo the initial
    tile across every step (matching the imperative wrapper's
    appended xi-history).

    When ``print_global_convergence`` is True, the scan body emits an
    ``ON PRIMAL STEP (n) at t=...`` header before each
    :func:`fe_newton_solve` call and forwards the flag so the inner
    Newton driver prints per-iter convergence lines.

    Returns stacked ``(N, …)`` outputs where
    ``N = t_schedule_jax.shape[0] - 1``.
    """

    def step_fn(carry, x):
        step_idx, t = x
        U_prev, xi_prev = carry
        if print_global_convergence:
            debug.print(
                "ON PRIMAL STEP ({step}) at t={t:.6e}",
                step=step_idx + 1,
                t=t,
            )
        U_solved, xi_solved_coupled = fe_newton_solve(
            fe_problem,
            params_by_block,
            U_prev=U_prev,
            t=t,
            xi_prev_by_block=xi_prev,
            max_iters=max_iters,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            print_global_convergence=print_global_convergence,
        )
        xi_solved_full = {**xi_prev, **xi_solved_coupled}
        new_carry = (U_solved, xi_solved_full)
        per_step = (U_solved, xi_solved_full)
        return new_carry, per_step

    initial_carry = (U_init, xi_init_by_block)
    n_steps = t_schedule_jax.shape[0] - 1
    step_indices = jnp.arange(n_steps)
    xs = (step_indices, t_schedule_jax[1:])
    _, history = lax.scan(step_fn, initial_carry, xs)
    U_steps, xi_steps_by_block = history
    return U_steps, xi_steps_by_block


def fe_quasistatic_drive(
        fe_problem: FEProblem,
        t_schedule: Sequence[float],
        U_init: NDArray[np.floating] | None = None,
        print_global_convergence: bool = False,
        **solver_kwargs: Any,
) -> FEState:
    """Run a quasi-static time loop and return populated state.

    ``t_schedule[0]`` is the initial time and ``t_schedule[1:]`` are
    the step times; the schedule must have at least two entries (an
    initial-only schedule is :meth:`FEState.from_problem` and doesn't
    need the driver). The driver seeds an :class:`FEState` at
    ``t_schedule[0]`` (zeros U or the user-supplied ``U_init``), then
    delegates the time loop to :func:`fe_quasistatic_drive_traced`
    (a single :func:`jax.lax.scan` over the schedule). The scan's
    stacked outputs are materialized back into FEState's per-step
    lists via :meth:`FEState.append`, mirroring the imperative
    contract callers expect.

    Within each step, the previous step's converged ``U`` flows into
    ``U_prev`` (Newton's starting iterate after the new step's
    prescribed BC values are overlaid); the previous step's
    converged ``xi`` flows into ``xi_prev_by_block`` for COUPLED
    blocks (the per-IP local Newton consumes ``xi_prev`` as both
    initial guess and the previous-step state input for plasticity
    path continuity).

    The appended-per-step ``xi_by_block`` payload is the union of
    COUPLED and CLOSED_FORM entries: COUPLED keys come from the
    Newton's returned ``xi_solved_by_block``; CLOSED_FORM keys echo
    the initial-tile xi from :meth:`FEState.from_problem`. This
    keeps every block's xi-history list aligned at length
    ``len(t_schedule)``.

    ``solver_kwargs`` are forwarded to :func:`fe_newton_solve`
    through :func:`fe_quasistatic_drive_traced` (accepts
    ``max_iters``, ``abs_tol``, ``rel_tol``); the orchestrator
    threads deck-supplied global-Newton settings through this slot.
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

    for t in t_schedule[1:]:
        fe_problem.dof_map.evaluate_prescribed_values(t)

    params_by_block = params_by_block_from_models(fe_problem)

    U_init_jax = jnp.asarray(state.U_at(0), dtype=jnp.float64)
    xi_init_by_block: dict[str, JaxArray] = {
        b: jnp.asarray(state.xi_at(0, b))
        for b in fe_problem.models_by_block
    }
    t_schedule_jax = jnp.asarray(t_schedule, dtype=jnp.float64)

    U_steps, xi_steps_by_block = fe_quasistatic_drive_traced(
        fe_problem,
        params_by_block,
        U_init_jax,
        xi_init_by_block,
        t_schedule_jax,
        print_global_convergence=print_global_convergence,
        **solver_kwargs,
    )

    n_steps = len(t_schedule) - 1
    for i in range(n_steps):
        xi_by_block_i = {
            b: np.asarray(xi_steps_by_block[b][i])
            for b in xi_steps_by_block
        }
        state.append(
            np.asarray(U_steps[i]),
            xi_by_block_i,
            t_new=t_schedule[i + 1],
        )

    return state
