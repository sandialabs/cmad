"""Quasi-static time-loop driver populating an :class:`FEState`.

Two layers:

- :func:`fe_quasistatic_trajectory` is the JAX-pure trajectory
  computation. It runs :func:`jax.lax.scan` over the time schedule
  with all-JAX carry / output. Each scan step calls
  :func:`cmad.fem.nonlinear_solver.fe_newton_solve` (which is itself
  JAX-traceable end to end), and an optional QoI step closure is
  invoked per step to accumulate a scalar functional. This is the
  AD entry point for callers that want :func:`jax.grad` over a
  quantity of interest defined on the trajectory.

- :func:`fe_quasistatic_drive` is the imperative wrapper used by
  :command:`cmad primal` and tests: builds an :class:`FEState`,
  invokes :func:`fe_quasistatic_trajectory`, and materializes the
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
from cmad.qois.fe_qoi import FEQoI, StepContribution
from cmad.typing import JaxArray, Params


def fe_quasistatic_trajectory(
        fe_problem: FEProblem,
        params_by_block: Mapping[str, Params],
        U_init: JaxArray,
        xi_init_by_block: dict[str, JaxArray],
        t_schedule_jax: JaxArray,
        qoi_step_contribution: StepContribution | None = None,
        max_iters: int = 20,
        abs_tol: float = 1e-10,
        rel_tol: float = 1e-10,
        print_global_convergence: bool = False,
        linear_solver: str = "direct",
) -> tuple[JaxArray, dict[str, JaxArray], JaxArray]:
    """Compute the FE trajectory via :func:`jax.lax.scan`.

    Carry slots ``(U, xi, t, J)`` hold the most-recently-converged
    step's outputs and the running QoI accumulator. ``xi`` is the
    all-blocks dict (keys are mesh element block names; entries
    cover both COUPLED and CLOSED_FORM blocks). CLOSED_FORM entries
    don't change per step but are kept in the carry to stabilize
    the pytree shape across iterations and avoid a per-iteration
    filter / merge in the step body. ``J`` is initialized to
    ``jnp.zeros(())`` and is always present, regardless of whether
    ``qoi_step_contribution`` is supplied — keeping the carry pytree
    stable across with / without-QoI calls. ``t`` rolls the
    latest-step time forward (initialized ``t_schedule_jax[0]``)
    so the step body has ``t_prev`` available without re-slicing
    the schedule.

    Per-step input: ``(step_idx, t)`` from
    ``(jnp.arange(n_steps), t_schedule_jax[1:])`` where
    ``n_steps = t_schedule_jax.shape[0] - 1`` is the number of
    advancing steps (does not count the initial condition).
    ``step_idx`` is consumed only by the optional debug print.

    Per-step output: ``(U_solved, xi)``. ``U_solved`` is the
    converged nodal vector from :func:`fe_newton_solve`; ``xi`` is
    the merged all-blocks dict for this step (COUPLED keys
    overwritten with the solver's freshly converged values,
    CLOSED_FORM keys echoed from ``xi_prev``).

    When ``qoi_step_contribution`` is supplied, the scan body invokes
    it after each :func:`fe_newton_solve` with
    ``(U_solved, U_prev, xi, xi_prev, t, t_prev)`` and accumulates
    the returned scalar into ``J``. When ``None``, the accumulator
    stays at zero and the returned ``J`` is ``jnp.zeros(())``.

    When ``print_global_convergence`` is True, the scan body emits an
    ``ON PRIMAL STEP (n) at t=...`` header before each
    :func:`fe_newton_solve` call and forwards the flag so the inner
    Newton driver prints per-iter convergence lines.

    Returns ``(U_steps, xi_steps_by_block, J)`` where the trajectory
    arrays have leading axis ``n_steps`` and ``J`` is the scalar
    QoI accumulated across the time loop.
    """

    def step_fn(carry, step_input):
        step_idx, t = step_input
        U_prev, xi_prev, t_prev, J = carry
        if print_global_convergence:
            debug.print(
                "ON PRIMAL STEP ({step}) at t={t:.6e}",
                step=step_idx + 1,
                t=t,
            )
        U_solved, xi_solved = fe_newton_solve(
            fe_problem,
            params_by_block,
            U_prev=U_prev,
            t=t,
            xi_prev_by_block=xi_prev,
            max_iters=max_iters,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            print_global_convergence=print_global_convergence,
            linear_solver=linear_solver,
        )
        # xi_solved only carries keys for element blocks whose model
        # has time-evolving state; the rest echo forward from xi_prev.
        xi = {**xi_prev, **xi_solved}
        if qoi_step_contribution is not None:
            J = J + qoi_step_contribution(
                U_solved, U_prev, xi, xi_prev, t, t_prev,
            )
        return (U_solved, xi, t, J), (U_solved, xi)

    n_steps = t_schedule_jax.shape[0] - 1
    initial_carry = (
        U_init, xi_init_by_block, t_schedule_jax[0], jnp.zeros(()),
    )
    step_inputs = (jnp.arange(n_steps), t_schedule_jax[1:])
    final_carry, history = lax.scan(step_fn, initial_carry, step_inputs)
    U_steps, xi_steps_by_block = history
    _, _, _, J = final_carry
    return U_steps, xi_steps_by_block, J


def fe_quasistatic_drive(
        fe_problem: FEProblem,
        t_schedule: Sequence[float],
        U_init: NDArray[np.floating] | None = None,
        print_global_convergence: bool = False,
        qoi: FEQoI | None = None,
        **solver_kwargs: Any,
) -> tuple[FEState, JaxArray]:
    """Run a quasi-static time loop and return populated state + QoI.

    ``t_schedule[0]`` is the initial time and ``t_schedule[1:]`` are
    the step times; the schedule must have at least two entries (an
    initial-only schedule is :meth:`FEState.from_problem` and doesn't
    need the driver). The driver seeds an :class:`FEState` at
    ``t_schedule[0]`` (zeros U or the user-supplied ``U_init``), then
    delegates the time loop to :func:`fe_quasistatic_trajectory`
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
    Newton's returned ``xi_solved``; CLOSED_FORM keys echo the
    initial-tile xi from :meth:`FEState.from_problem`. This keeps
    every block's xi-history list aligned at length
    ``len(t_schedule)``.

    When ``qoi`` is supplied, :meth:`FEQoI.step_contribution` is
    called once with ``params_by_block`` to build the per-step
    closure, which the trajectory layer invokes inside the scan to
    accumulate ``J``. The returned ``J`` is the full QoI value over
    the time loop. When ``qoi`` is ``None``, ``J = jnp.zeros(())``.

    ``solver_kwargs`` are forwarded to :func:`fe_newton_solve`
    through :func:`fe_quasistatic_trajectory` (accepts
    ``max_iters``, ``abs_tol``, ``rel_tol``, ``linear_solver``);
    the orchestrator threads deck-supplied global-Newton settings
    through this slot.
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

    qoi_step_contribution: StepContribution | None = (
        qoi.step_contribution(params_by_block)
        if qoi is not None else None
    )

    U_init_jax = jnp.asarray(state.U_at(0), dtype=jnp.float64)
    xi_init_by_block: dict[str, JaxArray] = {
        b: jnp.asarray(state.xi_at(0, b))
        for b in fe_problem.models_by_block
    }
    t_schedule_jax = jnp.asarray(t_schedule, dtype=jnp.float64)

    U_steps, xi_steps_by_block, J = fe_quasistatic_trajectory(
        fe_problem,
        params_by_block,
        U_init_jax,
        xi_init_by_block,
        t_schedule_jax,
        qoi_step_contribution=qoi_step_contribution,
        print_global_convergence=print_global_convergence,
        **solver_kwargs,
    )

    materialize_fe_state(state, U_steps, xi_steps_by_block, t_schedule)

    return state, J


def materialize_fe_state(
        state: FEState,
        U_steps: JaxArray,
        xi_steps_by_block: Mapping[str, JaxArray],
        t_schedule: Sequence[float],
) -> FEState:
    """Append per-step trajectory entries (U, xi, t) to ``state``.

    ``t_schedule`` is the full schedule (length ``n_steps + 1``);
    step ``i`` in ``U_steps`` / ``xi_steps_by_block`` is appended at
    time ``t_schedule[i + 1]``.
    """
    n_steps = U_steps.shape[0]
    for i in range(n_steps):
        xi_by_block_i = {
            b: np.asarray(xi_steps_by_block[b][i])
            for b in xi_steps_by_block
        }
        state.append(
            np.asarray(U_steps[i]),
            xi_by_block_i,
            t_new=float(t_schedule[i + 1]),
        )
    return state
