from cmad.fem_utils.problems.fem_problem import fem_problem
from cmad.fem_utils.models.neo_hookean import Neo_hookean
from cmad.fem_utils.models.mooney_rivlin import Mooney_rivlin
from cmad.fem_utils.models.thermoelastic import Thermoelastic
from cmad.fem_utils.models.thermo import Thermo
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

def halley_solve(model, num_steps, max_iters, tol):
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

            KFF_factorized = scipy.sparse.linalg.factorized(KFF)
            delta = KFF_factorized(-RF)

            if (i > 2):
                model.set_newton_increment(delta)
                halley_rhs = model.evaluate_halley_correction()
                delta = delta ** 2 / (delta + 1 / 2 * KFF_factorized(halley_rhs))

            model.add_to_UF(delta)

        model.save_global_fields()
        model.advance_model()

order = 2
problem = fem_problem("hole_block_disp_sliding", order, mixed=False)
num_steps, dt = problem.num_steps()

max_iters = 10
tol = 1e-11

model = Mooney_rivlin(problem)

newton_solve(model, num_steps, max_iters, tol)
# point_data = model.get_data()
# problem.save_data("rect_prism_thermoelastic_1.xdmf", point_data)












