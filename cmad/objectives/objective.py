import numpy as np

from cmad.solver.nonlinear_solver import newton_solve


class Objective():

    def __init__(self, qoi, gradient_type, weights=None):
        self._qoi = qoi
        self._model = qoi.model()
        self._parameters = qoi.model().parameters
        self._global_state = qoi.global_state()

        self._num_steps = qoi.data().shape[-1] - 1
        self._xi_at_step = [[None] * self._model.num_residuals
                            for ii in range(self._num_steps + 1)]
        self._model.store_xi(self._xi_at_step, self._model.xi(), 0)

        if gradient_type == "adjoint":
            self._evaluate = self._compute_adjoint_sens_fun_and_grad
        elif gradient_type == "forward_sens":
            self._evaluate = self._compute_forward_sens_fun_and_grad
        else:
            raise NotImplementedError

        if weights is None:
            self.weights = np.ones(self._num_steps+1)
        else:
            self.weights = weights


    def evaluate(self, flat_active_values):

        self._parameters.set_active_values_from_flat(flat_active_values)

        return self._evaluate()

    def _compute_adjoint_sens_fun_and_grad(self):

        qoi = self._qoi
        model = self._model
        F = self._global_state

        xi_at_step = self._xi_at_step
        model.set_xi_to_init_vals()

        J = 0.
        num_active_params = model.parameters.num_active_params
        grad = np.zeros((1, num_active_params))

        num_steps = self._num_steps

        # forward pass
        for step in range(1, num_steps + 1):

            u = [F[:, :, step]]
            u_prev = [F[:, :, step - 1]]
            model.gather_global(u, u_prev)

            newton_solve(model)
            model.store_xi(xi_at_step, model.xi(), step)

            model.seed_none()
            qoi.evaluate(step)
            J += self.weights[step]*qoi.J()

            model.advance_xi()

        # adjoint pass
        num_dofs = model.num_dofs
        history_vec = np.zeros((num_dofs, 1))

        for step in range(num_steps, 0, -1):

            u = [F[:, :, step]]
            u_prev = [F[:, :, step - 1]]
            model.gather_global(u, u_prev)

            xi = xi_at_step[step]
            xi_prev = xi_at_step[step - 1]
            model.gather_xi(xi, xi_prev)

            model.seed_xi()
            model.evaluate()
            dC_dxi = model.Jac()
            qoi.evaluate(step)
            dJ_dxi = self.weights[step]*qoi.dJ()

            phi = np.linalg.solve(dC_dxi.T, -dJ_dxi.T + history_vec)

            model.seed_xi_prev()
            model.evaluate()
            dC_dxi_prev = model.Jac()
            history_vec = -dC_dxi_prev.T @ phi

            model.seed_params()
            model.evaluate()
            dC_dp = model.Jac()
            qoi.evaluate(step)
            dJ_dp = self.weights[step]*qoi.dJ()

            grad += phi.T @ dC_dp + dJ_dp

        grad = model.parameters.transform_grad(grad.squeeze())

        return J, grad

    def _compute_forward_sens_fun_and_grad(self):

        qoi = self._qoi
        model = self._model
        F = self._global_state

        xi_at_step = self._xi_at_step
        model.set_xi_to_init_vals()

        num_active_params = model.parameters.num_active_params
        num_dofs = model.num_dofs

        J = 0.
        grad = np.zeros((1, num_active_params))
        dxi_dp = np.zeros((num_dofs, num_active_params))

        num_steps = self._num_steps

        for step in range(1, num_steps + 1):

            u = [F[:, :, step]]
            u_prev = [F[:, :, step - 1]]
            model.gather_global(u, u_prev)

            newton_solve(model)

            model.seed_none()
            qoi.evaluate(step)
            J += self.weights[step]*qoi.J()

            model.seed_xi()
            model.evaluate()
            dC_dxi = model.Jac()

            qoi.evaluate(step)
            dJ_dxi = self.weights[step]*qoi.dJ()

            model.seed_xi_prev()
            model.evaluate()
            dC_dxi_prev = model.Jac()

            model.seed_params()
            model.evaluate()
            dC_dp = model.Jac()

            qoi.evaluate(step)
            dJ_dp = self.weights[step]*qoi.dJ()

            rhs = -dC_dp - dC_dxi_prev @ dxi_dp
            dxi_dp = np.linalg.solve(dC_dxi, rhs)

            grad += dJ_dxi @ dxi_dp + dJ_dp

            model.advance_xi()

        grad = model.parameters.transform_grad(grad.squeeze())

        return J, grad
