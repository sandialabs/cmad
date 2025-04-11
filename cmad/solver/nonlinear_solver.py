import numpy as np

import jax.numpy as jnp

from jax import jvp, custom_jvp, jacfwd
from jax.flatten_util import ravel_pytree
from jax.lax import cond, while_loop

from cmad.util.pytree_linear_algebra import make_linop, make_op


def newton_solve(model, max_iters=10, abs_tol=1e-14, rel_tol=1e-14,
        max_ls_evals=0):

    converged = False
    ii = 0
    C_norm_0 = 1.

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

        delta_xi = np.linalg.solve(model.Jac(), -C)
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


def make_newton_solve(residual, x0, max_iters=10, abs_tol=1e-14, rel_tol=1e-14,
        max_ls_evals=0):
    _, tree_x = ravel_pytree(x0)
    linsolve = make_linop(jnp.linalg.solve, tree_x, tree_x)
    subtract = make_op(jnp.subtract, tree_x)
    negate = make_op(lambda p: -p, tree_x)


    @custom_jvp
    def newton_solve(*args):
        x_init = args[0]
        C = residual(x_init, *args)
        C_norm_0 = jnp.linalg.norm(C)


        def true_fun(carry):
            ii, converged, x, C, C_norm = carry
            return ii, True, x, C, C_norm


        def false_fun(carry):
            ii, converged, x, C, C_norm = carry

            jac = jacfwd(residual, 0)(x, *args)
            delta_x = linsolve(jac, C)
            x_2 = subtract(x, delta_x)

            C_2 = residual(x_2, *args)
            C_norm_2 = jnp.linalg.norm(C_2)

            return ii + 1, False, x_2, C_2, C_norm_2


        def cond_fun(carry):
            ii, converged, x, C, C_norm = carry
            return jnp.logical_and(ii < max_iters, jnp.logical_not(converged))


        def body_fun(carry):
            ii, converged, x, C, C_norm = carry
            C_norm_rel = C_norm / C_norm_0
            pred = jnp.logical_or(C_norm_rel < rel_tol, C_norm < abs_tol)
            return cond(pred, true_fun, false_fun, carry)


        return while_loop(cond_fun, body_fun,
            (0, False, x_init, C, C_norm_0))[2]


    @newton_solve.defjvp
    def newton_solve_jvp(primals, tangents):
        x = newton_solve(*primals)
        A = jacfwd(residual, 0)(x, *primals)
        _, b = jvp(lambda *args: residual(x, *args),
            primals, tangents)
        return x, negate(linsolve(A, b))


    return newton_solve
