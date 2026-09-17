from collections.abc import Callable
from functools import partial

import numpy as np
from jax import hessian, jit, value_and_grad
from jax.lax import fori_loop
from numpy.typing import NDArray

from cmad.models.global_fields import StepTime, mp_U_from_F
from cmad.qois.qoi import QoI
from cmad.typing import JaxArray, StateList


class MPJVPObjective:

    evaluate_objective: Callable[..., JaxArray]
    evaluate_objective_and_grad: Callable[..., tuple[JaxArray, JaxArray]]
    evaluate_hessian: Callable[..., JaxArray]

    def __init__(
            self, qoi: QoI, global_state: NDArray[np.floating],
            update_fun: Callable[..., StateList],
            times: NDArray[np.floating] | None = None,
    ) -> None:

        num_steps = global_state.shape[-1] - 1
        # One time per step of the deformation history; left unset the
        # steps are numbered 0, 1, 2, ..., so dt = 1 throughout. Matches
        # the default in MPObjective, so the two families agree on a
        # problem that names no times.
        if times is None:
            times = np.arange(num_steps + 1, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if times.size != num_steps + 1:
            raise ValueError(
                f"times holds {times.size} entries but the deformation "
                f"history runs {num_steps} steps; {num_steps + 1} times "
                f"are required (one per step, including the initial one)",
            )

        pt_compute_objective = partial(self._compute_objective_fun,
            qoi=qoi, F=global_state, times=times, update_fun=update_fun
        )

        self.evaluate_objective = jit(pt_compute_objective)

        self.evaluate_objective_and_grad = jit(
            value_and_grad(pt_compute_objective)
        )

        self.evaluate_hessian = jit(
            hessian(pt_compute_objective)
        )


    @staticmethod
    def _compute_objective_fun(
            flat_active_values: NDArray[np.floating],
            qoi: QoI,
            F: NDArray[np.floating],
            times: NDArray[np.floating],
            update_fun: Callable[..., StateList],
    ) -> JaxArray:

        # consider renaming these for public access
        model = qoi._model
        parameters = model.parameters
        data = qoi._data
        weight = qoi._weight

        model.set_xi_to_init_vals()

        params = \
            parameters.get_params_pytree_from_flat_canonical_active(
            flat_active_values
        )

        num_steps = F.shape[-1] - 1

        def body_fun(step, carry):
            J, xi, xi_prev, params, F, times, data, weight = carry
            U = mp_U_from_F(F[:, :, step])
            U_prev = mp_U_from_F(F[:, :, step - 1])
            # StepTime's t / t_prev are pytree children, so a varying
            # step size is traced rather than baked in: the loop is
            # compiled once no matter how the schedule is spaced.
            step_time = StepTime(times[step], times[step - 1])
            xi = update_fun(xi_prev, params, U, U_prev, step_time)

            carry = (qoi._qoi(xi, xi_prev, params, U, U_prev,
                data[:, :, step], weight) + J,
                xi, xi, params, F, times, data, weight
            )

            return carry

        J = fori_loop(1, num_steps + 1, body_fun,
            (0., model._init_xi, model._init_xi, params, F, times,
             data, weight))[0]

        return J
