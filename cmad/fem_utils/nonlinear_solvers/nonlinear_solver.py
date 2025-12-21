import numpy as np
import scipy.sparse.linalg
import scipy.sparse as sp
import time

def newton_solve(model, num_steps, max_iters, tol):
    model.initialize_variables()
    for step in range(num_steps):
        print('Step: ', step)
        model.compute_surf_tractions(step)
        model.set_prescribed_dofs(step)
        for i in range(max_iters):

            model.set_global_fields()

            model.seed_none()
            model.evaluate()
            RF = model.scatter_rhs()

            print("||R||: ", np.linalg.norm(RF))
            if (np.linalg.norm(RF) < tol):
                break

            model.seed_U()
            model.evaluate()
            KFF = model.scatter_lhs()

            # diag = KFF.diagonal()
            # T = sp.diags(1 / np.sqrt(diag), 0)

            # delta = scipy.sparse.linalg.spsolve(T @ KFF @ T, -T.dot(RF))

            KFF_factorized = scipy.sparse.linalg.factorized(KFF)
            delta = KFF_factorized(-RF)

            model.add_to_UF(delta)

        model.save_global_fields()
        model.advance_model()

def halley_solve(model, num_steps, max_iters, tol, halley_threshold):
    model.initialize_variables()
    for step in range(num_steps):
        print('Timestep', step)
        # set displacement BCs
        model.set_prescribed_dofs(step)
        model.set_global_fields()

        #set traction BCs
        model.compute_surf_tractions(step)

        model.seed_none()
        model.evaluate()
        RF = model.scatter_rhs()
        norm_resid_0 = np.linalg.norm(RF)

        for i in range(max_iters):
            norm_resid = np.linalg.norm(RF)
            print("||R|| = ", norm_resid)
            if (norm_resid < tol):
                break

            model.seed_U()
            model.evaluate()
            KFF = model.scatter_lhs()
            KFF_factorized = scipy.sparse.linalg.factorized(KFF)
            delta = KFF_factorized(-RF)

            UF_curr = model.get_UF()
            UF_new = UF_curr + delta
            model.set_UF(UF_new)
            model.set_global_fields()

            model.seed_none()
            model.evaluate()
            RF_new = model.scatter_rhs()
            norm_resid_new = np.linalg.norm(RF_new)
            S = np.log10(norm_resid_0 / norm_resid)

            if S < halley_threshold:
                RF = RF_new.copy()
            else:
                if norm_resid_new < tol:
                    RF = RF_new.copy()
                else:
                    print("Computing Halley correction...")
                    model.set_newton_increment(delta)
                    halley_rhs = model.evaluate_halley_correction()
                    halley_delta = delta ** 2 / (delta + 1 / 2 * KFF_factorized(halley_rhs))

                    UF_new = UF_curr + halley_delta
                    model.set_UF(UF_new)
                    model.set_global_fields()

                    model.seed_none()
                    model.evaluate()
                    RF = model.scatter_rhs()
        model.advance_model()












