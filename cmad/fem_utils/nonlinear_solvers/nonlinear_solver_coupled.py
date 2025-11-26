from cmad.fem_utils.problems.fem_problem import fem_problem
from cmad.fem_utils.models.elastic_plastic_small import Elastic_plastic_small
from cmad.fem_utils.models.elastic_plastic_finite import Elastic_plastic_finite
from cmad.fem_utils.models.thermoplastic_small import Thermoplastic_small
from cmad.fem_utils.models.elastic_plastic_small_plane_stress import \
    Elastic_plastic_small_plane_stress
import numpy as np
import scipy.sparse.linalg
import time

def newton_solve(model, num_steps, max_iters, tol):
    # model.initialize_plot()
    model.initialize_variables()
    for step in range(num_steps):
        print('Step: ', step)
        model.set_prescribed_dofs(step)
        model.compute_surf_tractions(step)
        for i in range(max_iters):
            model.set_global_fields()

            model.compute_local_state_variables()
            model.evaluate_local()

            model.evaluate_global()
            RF = model.scatter_rhs()
            norm_resid = np.linalg.norm(RF)
            print('Iteration: ', i)
            print(" ||C|| = ", np.max(np.abs(model.C())),
                  " ||R|| = ", norm_resid)
            if (norm_resid < tol):
                break

            model.evaluate_tang()
            KFF = model.scatter_lhs()

            KFF_factorized = scipy.sparse.linalg.factorized(KFF)
            delta = KFF_factorized(-RF)

            model.add_to_UF(delta)
            model.reset_xi()

        model.save_global_fields()
        # model.update_plot()
        model.advance_model()

def newton_solve_line_search(model, num_steps, max_iters, tol, s=0.8, m=8):
    # model.initialize_plot()
    model.initialize_variables()
    for step in range(num_steps):
        print('Timestep', step)
        # set displacement BCs
        model.set_prescribed_dofs(step)
        model.set_global_fields()

        #set traction BCs
        model.compute_surf_tractions(step)

        model.reset_xi()
        print("Computing local state variables...")
        model.compute_local_state_variables()
        model.evaluate_local()
        print("||C|| = ", np.max(np.abs(model.C())))

        model.evaluate_global()
        RF = model.scatter_rhs()

        for i in range(max_iters):
            norm_resid = np.linalg.norm(RF)
            print('Newton Iteration: ', i)
            print("||R|| = ", norm_resid)
            if (norm_resid < tol):
                break

            model.evaluate_tang()
            KFF = model.scatter_lhs()
            # KFF_array = KFF.toarray()
            # print("converted")
            # cond = np.linalg.cond(KFF_array)
            # print(cond)
            KFF_factorized = scipy.sparse.linalg.factorized(KFF)
            delta = KFF_factorized(-RF)

            UF_curr = model.get_UF()
            UF_new = UF_curr + delta
            model.set_UF(UF_new)
            model.set_global_fields()

            model.reset_xi()
            print("Computing local state variables...")
            model.compute_local_state_variables()
            model.evaluate_local()
            print("||C|| = ", np.max(np.abs(model.C())))

            model.evaluate_global()
            RF_new = model.scatter_rhs()
            if (np.abs(np.dot(delta, RF_new)) <= s * np.abs(np.dot(delta, RF))):
                print('No line search')
                RF = RF_new.copy()
            else:
                print('Performing line search')
                for j in range(1, m + 1):
                    # print("Line search iteration: ", j)
                    eta = (m - j + 1) / (m + 1)
                    UF_new = UF_curr + eta * delta
                    model.set_UF(UF_new)
                    model.set_global_fields()

                    model.reset_xi()
                    # print("Computing local state variables...")
                    model.compute_local_state_variables()
                    model.evaluate_local()
                    # print("||C|| = ", np.max(np.abs(model.C())))

                    model.evaluate_global()
                    RF_new = model.scatter_rhs()
                    if (np.abs(np.dot(delta, RF_new)) <= s * np.abs(np.dot(delta, RF)) or j == m):
                        RF = RF_new.copy()
                        break

        model.save_global_fields()
        # model.update_plot()
        model.advance_model()

def halley_solve(model, num_steps, max_iters, tol, halley_threshold):
    # model.initialize_plot()
    model.initialize_variables()
    for step in range(num_steps):
        print('Timestep', step)
        # set displacement BCs
        model.set_prescribed_dofs(step)
        model.set_global_fields()

        #set traction BCs
        model.compute_surf_tractions(step)

        model.reset_xi()
        model.compute_local_state_variables()
        model.evaluate_local()

        model.evaluate_global()
        RF = model.scatter_rhs()
        norm_resid_0 = np.linalg.norm(RF)

        for i in range(max_iters):
            norm_resid = np.linalg.norm(RF)
            print(" ||C|| = ", np.max(np.abs(model.C())),
                  " ||R|| = ", norm_resid)
            if (norm_resid < tol):
                break

            model.evaluate_tang()
            KFF = model.scatter_lhs()
            KFF_factorized = scipy.sparse.linalg.factorized(KFF)
            delta = KFF_factorized(-RF)

            UF_curr = model.get_UF()
            UF_new = UF_curr + delta
            model.set_UF(UF_new)
            model.set_global_fields()

            model.reset_xi()
            model.compute_local_state_variables()
            model.evaluate_local()

            model.evaluate_global()
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
                    halley_rhs = model.evaluate_halley_correction_multi(24)
                    halley_delta = delta ** 2 / (delta + 1 / 2 * KFF_factorized(halley_rhs))

                    UF_new = UF_curr + halley_delta
                    model.set_UF(UF_new)
                    model.set_global_fields()

                    model.reset_xi()
                    model.compute_local_state_variables()
                    model.evaluate_local()

                    model.evaluate_global()
                    RF = model.scatter_rhs()
        model.advance_model()

order = 1
problem = fem_problem("hole_block_disp_sliding", order, mixed=True)
num_steps, dt = problem.num_steps()

max_iters = 20
tol = 1e-10

model = Elastic_plastic_finite(problem)
newton_solve_line_search(model, num_steps, max_iters, tol)
point_data, cell_data = model.get_data()
problem.save_data("test_1.xdmf", point_data, cell_data)

# halley_threshold = 3.0

# model = Elastic_plastic_small(problem)
# halley_solve(model, num_steps, max_iters, tol, halley_threshold)

# # # Save results as .xdmf file
# point_data, cell_data = model.get_data()
# problem.save_data("hole_block_traction.xdmf", point_data, cell_data)