"""Quasi-static time-loop driver populating an :class:`FEState`.

Two layers:

- :func:`build_fe_quasistatic_trajectory` is a factory: it captures an
  :class:`FEProblem` and the solver settings and returns a JAX-pure
  ``trajectory`` closure. The closure runs :func:`jax.lax.scan` over
  the time schedule with all-JAX carry / output; each scan step calls
  :func:`cmad.fem.nonlinear_solver._fe_newton_solve_ad`, and an
  optional QoI step closure is invoked per step to accumulate a scalar
  functional. This is the AD entry point for callers that want
  :func:`jax.grad` over a quantity of interest defined on the
  trajectory.

- :func:`fe_quasistatic_drive` is the imperative wrapper used by
  :command:`cmad primal` and tests: builds an :class:`FEState`,
  invokes the ``trajectory`` closure, and materializes the stacked
  outputs back into FEState's mutable per-step lists.
"""
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import jax.numpy as jnp
import numpy as np
from jax import debug, lax
from numpy.typing import NDArray

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.kernel_arrays import FEKernelArrays
from cmad.fem.nonlinear_solver import (
    _DEFAULT_LINEAR_SOLVER_SETTINGS,
    _DEFAULT_NONLINEAR_SOLVER_SETTINGS,
    _fe_newton_solve_ad,
    _freeze,
)
from cmad.qois.fe_qoi import FEQoI, StepContribution
from cmad.typing import JaxArray, Params

# (U_init, xi_init_by_block): the initial state seeding the trajectory.
StateInit: TypeAlias = tuple[JaxArray, dict[str, JaxArray]]


def build_fe_quasistatic_trajectory(
        fe_problem: FEProblem,
        nonlinear_solver_settings: dict[str, Any] | None = None,
        linear_solver_settings: dict[str, Any] | None = None,
) -> Callable[..., tuple[JaxArray, dict[str, JaxArray], JaxArray]]:
    """Build the JAX-pure quasi-static ``trajectory`` closure.

    The factory captures ``fe_problem`` and the merged solver settings;
    the returned closure runs the time loop:

        ``trajectory(fe_arrays, params_by_block, state_init,
        t_schedule_jax, qoi_step_contribution=None)``

    ``state_init`` is the ``(U_init, xi_init_by_block)`` pair seeding
    the loop; ``xi_init_by_block`` is the all-blocks dict (keys are
    mesh element block names, covering both COUPLED and CLOSED_FORM
    blocks).

    The closure runs :func:`jax.lax.scan` over the time schedule. Carry
    slots ``(U, xi, t, J)`` hold the most-recently-converged step's
    outputs and the running QoI accumulator. CLOSED_FORM ``xi`` entries
    don't change per step but stay in the carry to keep the pytree
    shape stable. ``J`` starts at ``jnp.zeros(())`` and is always
    present, whether or not ``qoi_step_contribution`` is supplied.
    ``t`` rolls the latest-step time forward (initialized
    ``t_schedule_jax[0]``) so the step body has ``t_prev`` available.

    Per-step input ``(step_idx, t)`` comes from
    ``(jnp.arange(n_steps), t_schedule_jax[1:])`` with
    ``n_steps = t_schedule_jax.shape[0] - 1``; ``step_idx`` feeds only
    the optional debug print. Per-step output is ``(U_solved, xi)``,
    the converged nodal vector and the merged all-blocks xi dict.

    When ``qoi_step_contribution`` is supplied, the scan body invokes
    it after each solve with ``(U_solved, U_prev, xi, xi_prev, t,
    t_prev)`` and accumulates the returned scalar into ``J``; when
    ``None``, ``J`` stays zero.

    When the nonlinear-solver settings carry ``print convergence``,
    the scan body emits an ``ON PRIMAL STEP`` header before each solve
    and the inner Newton driver prints per-iter convergence lines.

    The closure returns ``(U_steps, xi_steps_by_block, J)``: the
    trajectory arrays have leading axis ``n_steps``, and ``J`` is the
    scalar QoI accumulated across the time loop.
    """
    nls = {
        **_DEFAULT_NONLINEAR_SOLVER_SETTINGS,
        **(nonlinear_solver_settings or {}),
    }
    lss = {
        **_DEFAULT_LINEAR_SOLVER_SETTINGS,
        **(linear_solver_settings or {}),
    }
    print_global_convergence = nls["print convergence"]
    nls_frozen = _freeze(nls)
    lss_frozen = _freeze(lss)

    def trajectory(
            fe_arrays: FEKernelArrays,
            params_by_block: Mapping[str, Params],
            state_init: StateInit,
            t_schedule_jax: JaxArray,
            qoi_step_contribution: StepContribution | None = None,
    ) -> tuple[JaxArray, dict[str, JaxArray], JaxArray]:
        U_init, xi_init_by_block = state_init

        def step_fn(carry, step_input):
            step_idx, t = step_input
            U_prev, xi_prev, t_prev, J = carry
            if print_global_convergence:
                debug.print(
                    "ON PRIMAL STEP ({step}) at t={t:.6e}",
                    step=step_idx + 1,
                    t=t,
                )
            U_solved, xi_solved = _fe_newton_solve_ad(
                fe_problem, fe_arrays, params_by_block,
                U_prev, xi_prev, t, nls_frozen, lss_frozen,
            )
            # xi_solved only carries keys for element blocks whose
            # model has time-evolving state; the rest echo forward.
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
        final_carry, history = lax.scan(
            step_fn, initial_carry, step_inputs,
        )
        U_steps, xi_steps_by_block = history
        _, _, _, J = final_carry
        return U_steps, xi_steps_by_block, J

    return trajectory


def fe_quasistatic_drive(
        fe_problem: FEProblem,
        t_schedule: Sequence[float],
        U_init: NDArray[np.floating] | None = None,
        qoi: FEQoI | None = None,
        **solver_kwargs: Any,
) -> tuple[FEState, JaxArray]:
    """Run a quasi-static time loop and return populated state + QoI.

    ``t_schedule[0]`` is the initial time and ``t_schedule[1:]`` are
    the step times; the schedule must have at least two entries (an
    initial-only schedule is :meth:`FEState.from_problem` and doesn't
    need the driver). The driver seeds an :class:`FEState` at
    ``t_schedule[0]`` (zeros U or the user-supplied ``U_init``), then
    delegates the time loop to the ``trajectory`` closure built by
    :func:`build_fe_quasistatic_trajectory` (a single
    :func:`jax.lax.scan` over the schedule). The scan's stacked
    outputs are materialized back into FEState's per-step lists via
    :meth:`FEState.append`, mirroring the imperative contract callers
    expect.

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
    called once with ``params_by_block`` and the kernel arrays to
    build the per-step closure, which the trajectory layer invokes
    inside the scan to accumulate ``J``. The returned ``J`` is the
    full QoI value over the time loop. When ``qoi`` is ``None``,
    ``J = jnp.zeros(())``.

    ``solver_kwargs`` (``nonlinear_solver_settings`` and
    ``linear_solver_settings`` dicts) are forwarded to
    :func:`build_fe_quasistatic_trajectory`; the orchestrator threads
    deck-supplied settings through this slot.
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
    fe_arrays = fe_problem.kernel_arrays

    qoi_step_contribution: StepContribution | None = (
        qoi.step_contribution(params_by_block, fe_arrays)
        if qoi is not None else None
    )

    U_init_jax = jnp.asarray(state.U_at(0), dtype=jnp.float64)
    xi_init_by_block: dict[str, JaxArray] = {
        b: jnp.asarray(state.xi_at(0, b))
        for b in fe_problem.models_by_block
    }
    state_init: StateInit = (U_init_jax, xi_init_by_block)
    t_schedule_jax = jnp.asarray(t_schedule, dtype=jnp.float64)

    trajectory = build_fe_quasistatic_trajectory(
        fe_problem, **solver_kwargs,
    )
    U_steps, xi_steps_by_block, J = trajectory(
        fe_arrays,
        params_by_block,
        state_init,
        t_schedule_jax,
        qoi_step_contribution=qoi_step_contribution,
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
