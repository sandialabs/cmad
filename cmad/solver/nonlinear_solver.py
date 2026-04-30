from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax import custom_jvp, jacfwd, jvp
from jax.flatten_util import ravel_pytree
from jax.lax import cond, while_loop

from cmad.typing import JaxArray, PyTree, SupportsNewton


def newton_solve(
        model: SupportsNewton,
        max_iters: int = 10,
        abs_tol: float = 1e-14,
        rel_tol: float = 1e-14,
        max_ls_evals: int = 0,
) -> tuple[int, float]:

    converged = False
    ii = 0
    C_norm_0: np.floating | float = 1.
    C_norm: np.floating | float = 0.

    beta = 1e-4
    eta = 0.5


    while ii < max_iters and not converged:
        model.seed_none()
        model.evaluate()
        C = model.C()
        C_norm = np.linalg.norm(C)

        if ii == 0:
            C_norm_0 = C_norm
            C_norm_rel = 1.
        else:
            C_norm_rel = C_norm / C_norm_0

        if C_norm_rel < rel_tol or C_norm < abs_tol:
            converged = True
            break

        model.seed_xi()
        model.evaluate()

        Jac = model.Jac()
        assert Jac is not None  # seed_xi() above ensures Jac is populated
        delta_xi = np.linalg.solve(Jac, -C)
        model.add_to_xi(delta_xi)

        if max_ls_evals > 0:
            model.seed_none()
            model.evaluate()

            C_0 = C_norm
            psi_0 = 0.5 * C_0**2
            psi_0_deriv = -2. * psi_0

            jj = 1
            alpha_j = 1.
            C_j = np.linalg.norm(model.C())
            psi_j = 0.5 * C_j**2

            while psi_j >= ((1. - 2. * beta * alpha_j) * psi_0):
                alpha_prev = alpha_j
                alpha_j = max(eta * alpha_j, -(alpha_j**2 * psi_0_deriv)
                    / (2. * (psi_j - psi_0 - alpha_j * psi_0_deriv)))
                if (jj == max_ls_evals):
                    print("reached max ls evals")
                    break
                jj += 1
                alpha_diff = alpha_j - alpha_prev
                model.add_to_xi(alpha_diff * delta_xi)

                model.evaluate()
                C_j = np.linalg.norm(model.C())
                psi_j = 0.5 * C_j**2

        ii += 1

    return ii, float(C_norm)


def make_newton_solve(
        residual: Callable[..., JaxArray],
        max_iters: int = 10,
        abs_tol: float = 1e-14,
        rel_tol: float = 1e-14,
) -> Callable[..., PyTree]:

    @custom_jvp
    def newton_solve(x_prev, *other_fixed_args):
        flat_x_prev, unravel = ravel_pytree(x_prev)

        def residual_flat(x_flat):
            return ravel_pytree(
                residual(unravel(x_flat), x_prev, *other_fixed_args)
            )[0]

        C0 = residual_flat(flat_x_prev)
        C_norm_0 = jnp.linalg.norm(C0)


        def true_fun(carry):
            ii, _converged, x, C, C_norm = carry
            return ii, True, x, C, C_norm


        def false_fun(carry):
            ii, _converged, x, C, _C_norm = carry

            jac = jacfwd(residual_flat)(x)
            delta_x = jnp.linalg.solve(jac, C)
            x_2 = jnp.subtract(x, delta_x)

            C_2 = residual_flat(x_2)
            C_norm_2 = jnp.linalg.norm(C_2)

            return ii + 1, False, x_2, C_2, C_norm_2


        def cond_fun(carry):
            ii, converged, _x, _C, _C_norm = carry
            return jnp.logical_and(ii < max_iters, jnp.logical_not(converged))


        def body_fun(carry):
            _ii, _converged, _x, _C, C_norm = carry
            C_norm_rel = C_norm / C_norm_0
            pred = jnp.logical_or(C_norm_rel < rel_tol, C_norm < abs_tol)
            return cond(pred, true_fun, false_fun, carry)


        flat_x = while_loop(cond_fun, body_fun,
            (0, False, flat_x_prev, C0, C_norm_0))[2]
        return unravel(flat_x)


    @newton_solve.defjvp
    def newton_solve_jvp(primals, tangents):
        x_prev = primals[0]
        other_fixed_args = primals[1:]
        x = newton_solve(x_prev, *other_fixed_args)
        flat_x, unravel = ravel_pytree(x)

        def residual_flat(x_flat, x_p, *args):
            return ravel_pytree(residual(unravel(x_flat), x_p, *args))[0]

        A = jacfwd(residual_flat, 0)(flat_x, x_prev, *other_fixed_args)
        _, b = jvp(lambda *args: residual_flat(flat_x, *args),
            primals, tangents)
        return x, unravel(-jnp.linalg.solve(A, b))


    return newton_solve
