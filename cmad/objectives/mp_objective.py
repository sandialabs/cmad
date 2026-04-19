"""Sensitivity-providing objectives for material-point calibration.

Three sibling classes share an MPObjective ABC, each implementing one
sensitivity strategy. MPJVPObjective in jvp_objective.py is a fourth
end-to-end JAX-traced sibling; it is structurally distinct and does
not subclass MPObjective.
"""
from abc import ABC, abstractmethod
from typing import cast

import numpy as np
from numpy.typing import NDArray

from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.qois.qoi import QoI
from cmad.solver.nonlinear_solver import newton_solve
from cmad.typing import GradientResult, HessianResult, StateList


class MPObjective(ABC):
    """Base class for sensitivity-providing objectives.

    Owns time-step loop scaffolding, parameter injection, forward-pass
    state storage, and the shared forward-pass-with-storage helper used
    by MPAdjointObjective and MPDirectAdjointObjective.
    """

    _qoi: QoI
    _model: Model
    _parameters: Parameters
    _global_state: NDArray[np.floating]
    _num_steps: int
    _xi_at_step: list[StateList]

    def __init__(
            self, qoi: QoI, global_state: NDArray[np.floating],
    ) -> None:
        self._qoi = qoi
        self._model = qoi.model()
        self._parameters = qoi.model().parameters
        self._global_state = global_state

        self._num_steps = qoi.data().shape[-1] - 1
        self._xi_at_step = cast(
            list[StateList],
            [[None] * self._model.num_residuals
             for ii in range(self._num_steps + 1)],
        )
        self._model.store_xi(self._xi_at_step, self._model.xi(), 0)

    def evaluate(
        self, flat_active_values: NDArray[np.floating]
    ) -> GradientResult | HessianResult:
        self._parameters.set_active_values_from_flat(flat_active_values)
        return self._evaluate()

    @abstractmethod
    def _evaluate(self) -> GradientResult | HessianResult: ...

    def _forward_pass_with_storage(self) -> float:
        """Forward time-step loop with xi_at_step storage. Returns J."""
        qoi = self._qoi
        model = self._model
        F = self._global_state
        xi_at_step = self._xi_at_step
        model.set_xi_to_init_vals()

        J: float | NDArray[np.number] = 0.
        num_steps = self._num_steps

        for step in range(1, num_steps + 1):

            u = [F[:, :, step]]
            u_prev = [F[:, :, step - 1]]
            model.gather_global(u, u_prev)

            newton_solve(model)
            model.store_xi(xi_at_step, model.xi(), step)

            model.seed_none()
            qoi.evaluate(step)
            J += qoi.J()

            model.advance_xi()

        return float(J)


class MPAdjointObjective(MPObjective):
    """Gradient via reverse-time adjoint pass after a forward pass."""

    def _evaluate(self) -> GradientResult:

        qoi = self._qoi
        model = self._model
        F = self._global_state
        xi_at_step = self._xi_at_step
        num_steps = self._num_steps

        J = self._forward_pass_with_storage()

        num_active_params = model.parameters.num_active_params
        grad = np.zeros((1, num_active_params))

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
            dJ_dxi = qoi.dJ()

            phi = np.linalg.solve(dC_dxi.T, -dJ_dxi.T + history_vec)

            model.seed_xi_prev()
            model.evaluate()
            dC_dxi_prev = model.Jac()
            history_vec = -dC_dxi_prev.T @ phi

            model.seed_params()
            model.evaluate()
            dC_dp = model.Jac()
            qoi.evaluate(step)
            dJ_dp = qoi.dJ()

            grad += phi.T @ dC_dp + dJ_dp

        grad = grad.squeeze()
        model.parameters.transform_grad(grad)

        return GradientResult(J=J, grad=grad)


class MPDirectObjective(MPObjective):
    """Gradient via forward sensitivity (tangent) pass.

    Forward loop is inlined rather than reusing _forward_pass_with_storage
    because the gradient work is interleaved into the loop body and past
    states are not needed (so xi_at_step is never populated).
    """

    def _evaluate(self) -> GradientResult:

        qoi = self._qoi
        model = self._model
        F = self._global_state
        model.set_xi_to_init_vals()

        num_active_params = model.parameters.num_active_params
        num_dofs = model.num_dofs

        J: float | NDArray[np.number] = 0.
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
            J += qoi.J()

            model.seed_xi()
            model.evaluate()
            dC_dxi = model.Jac()

            qoi.evaluate(step)
            dJ_dxi = qoi.dJ()

            model.seed_xi_prev()
            model.evaluate()
            dC_dxi_prev = model.Jac()

            model.seed_params()
            model.evaluate()
            dC_dp = model.Jac()

            qoi.evaluate(step)
            dJ_dp = qoi.dJ()

            rhs = -dC_dp - dC_dxi_prev @ dxi_dp
            dxi_dp = np.linalg.solve(dC_dxi, rhs)

            grad += dJ_dxi @ dxi_dp + dJ_dp

            model.advance_xi()

        grad = grad.squeeze()
        model.parameters.transform_grad(grad)

        return GradientResult(J=float(J), grad=grad)


class MPDirectAdjointObjective(MPObjective):
    """Gradient + Hessian via direct-adjoint method (arXiv:2501.04584)."""

    def _evaluate(self) -> HessianResult:

        qoi = self._qoi
        model = self._model
        F = self._global_state
        xi_at_step = self._xi_at_step
        num_steps = self._num_steps

        J = self._forward_pass_with_storage()

        num_active_params = model.parameters.num_active_params
        grad = np.zeros((1, num_active_params))

        # adjoint pass
        num_dofs = model.num_dofs
        history_vec = np.zeros((num_dofs, 1))
        phi_at_step = [np.zeros(num_dofs)] * (num_steps + 1)

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
            dJ_dxi = qoi.dJ()

            phi = np.linalg.solve(dC_dxi.T, -dJ_dxi.T + history_vec)
            phi_at_step[step] = phi.squeeze()

            model.seed_xi_prev()
            model.evaluate()
            dC_dxi_prev = model.Jac()
            history_vec = -dC_dxi_prev.T @ phi

            model.seed_params()
            model.evaluate()
            dC_dp = model.Jac()
            qoi.evaluate(step)
            dJ_dp = qoi.dJ()

            grad += phi.T @ dC_dp + dJ_dp

        grad = grad.squeeze()
        untransformed_grad = grad.copy()
        model.parameters.transform_grad(grad)

        # direct-adjoint pass for Hessian
        hessian = np.zeros((num_active_params, num_active_params))
        dxi_dp_prev = np.zeros((num_dofs, num_active_params))
        dxi_dp = np.zeros((num_dofs, num_active_params))

        for step in range(1, num_steps + 1):

            u = [F[:, :, step]]
            u_prev = [F[:, :, step - 1]]
            model.gather_global(u, u_prev)

            xi = xi_at_step[step]
            xi_prev = xi_at_step[step - 1]
            model.gather_xi(xi, xi_prev)

            model.seed_xi()
            model.evaluate()
            dC_dxi = model.Jac()

            model.seed_xi_prev()
            model.evaluate()
            dC_dxi_prev = model.Jac()

            model.seed_params()
            model.evaluate()
            dC_dp = model.Jac()

            rhs = -dC_dp - dC_dxi_prev @ dxi_dp_prev
            dxi_dp = np.linalg.solve(dC_dxi, rhs)

            model.evaluate_hessians()
            d2C_dxi2 = model.d2C_dxi2
            d2C_dxi_dxi_prev = model.d2C_dxi_dxi_prev
            d2C_dxi_prev2 = model.d2C_dxi_prev2
            d2C_dp2 = model.d2C_dparams2
            d2C_dp_dxi = model.d2C_dxi_dparams.transpose((0, 2, 1))
            d2C_dp_dxi_prev = model.d2C_dxi_prev_dparams.transpose((0, 2, 1))

            qoi.evaluate_hessians(step)
            d2J_dxi2 = qoi.d2J_dxi2
            d2J_dp2 = qoi.d2J_dparams2
            d2J_dp_dxi = qoi.d2J_dxi_dparams.T

            phi = phi_at_step[step]

            # compute hessian
            hessian += d2J_dp2 \
                + np.einsum("q,qij->ij", phi, d2C_dp2) \
                + np.einsum("ik,kj->ij", d2J_dp_dxi, dxi_dp) \
                + np.einsum("q,qik,kj->ij", phi, d2C_dp_dxi, dxi_dp) \
                + np.einsum("jk,ki->ij", d2J_dp_dxi, dxi_dp) \
                + np.einsum("q,qjk,ki->ij", phi, d2C_dp_dxi, dxi_dp) \
                + np.einsum("km,ki,mj->ij", d2J_dxi2, dxi_dp, dxi_dp) \
                + np.einsum("q,qkm,ki,mj->ij", phi, d2C_dxi2, dxi_dp,
                                               dxi_dp) \
                + np.einsum("q,qik,kj->ij", phi, d2C_dp_dxi_prev, dxi_dp_prev) \
                + np.einsum("q,qkm,ki,mj->ij", phi, d2C_dxi_dxi_prev,
                                               dxi_dp, dxi_dp_prev) \
                + np.einsum("q,qmk,ki,mj->ij", phi, d2C_dxi_dxi_prev,
                                               dxi_dp_prev, dxi_dp) \
                + np.einsum("q,qkm,ki,mj->ij", phi, d2C_dxi_prev2, dxi_dp_prev,
                                               dxi_dp_prev) \
                + np.einsum("q,qjk,ki->ij", phi, d2C_dp_dxi_prev, dxi_dp_prev)

            dxi_dp_prev = dxi_dp

        model.parameters.transform_hessian(hessian, untransformed_grad)

        return HessianResult(J=J, grad=grad, hessian=hessian)
