import numpy as np


def newton_solve(model, max_iters=10, abs_tol=1e-14, rel_tol=1e-14):

    converged = False
    ii = 0
    C_norm_0 = 1.

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

        ii += 1
