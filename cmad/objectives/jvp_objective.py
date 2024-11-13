from functools import partial

from jax import jit, value_and_grad, hessian

from jax.lax import fori_loop


class JVPObjective():


    def __init__(self, qoi, update_fun):

        pt_compute_objective = partial(self._compute_objective_fun,
            qoi=qoi, update_fun=update_fun
        )

        self.evaluate_objective = jit(pt_compute_objective)

        self.evaluate_objective_and_grad = jit(
            value_and_grad(pt_compute_objective)
        )

        self.evaluate_hessian = jit(
            hessian(pt_compute_objective)
        )


    @staticmethod
    def _compute_objective_fun(flat_active_values, qoi, update_fun):

        # consider renaming these for public access
        model = qoi._model
        parameters = model.parameters
        F = qoi._global_state
        data = qoi._data
        weight = qoi._weight

        model.set_xi_to_init_vals()

        params = \
            parameters.get_params_pytree_from_flat_canonical_active(
            flat_active_values
        )

        num_steps = F.shape[-1] - 1

        def body_fun(step, carry):
            J, xi, xi_prev, params, F, data, weight = carry
            xi = update_fun(xi_prev, params,
                [F[:, :, step]], [F[:, :, step - 1]]
            )

            carry = (qoi._qoi(xi, xi_prev, params,
                [F[:, :, step]], [F[:, :, step - 1]],
                data[:, :, step], weight) + J,
                xi, xi, params, F, data, weight
            )

            return carry

        J = fori_loop(1, num_steps + 1, body_fun,
            (0., model._init_xi, model._init_xi, params, F,
             data, weight))[0]

        return J
