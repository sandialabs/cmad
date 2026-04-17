"""
Verify DirectAdjointObjective and JVPObjective produce consistent J,
gradient, and Hessian on a J2 plane-stress calibration problem.

The two methods compute the same quantity by structurally different
algorithms (numpy einsum across model+QoI Hessian blocks vs end-to-end
JAX hessian of a fori_loop); agreement to roughly float64 precision
modulo accumulated rounding is expected.
"""
import unittest

import numpy as np

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.objectives.jvp_objective import JVPObjective
from cmad.objectives.objective import DirectAdjointObjective
from cmad.qois.calibration import Calibration
from cmad.solver.nonlinear_solver import make_newton_solve, newton_solve
from tests.support.test_problems import J2AnalyticalProblem


def compute_cauchy(model, F):
    num_steps = F.shape[2] - 1
    model.set_xi_to_init_vals()
    cauchy = np.zeros((3, 3, num_steps + 1))
    for step in range(1, num_steps + 1):
        u = [F[:, :, step]]
        u_prev = [F[:, :, step - 1]]
        model.gather_global(u, u_prev)
        newton_solve(model)
        model.advance_xi()
        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
    return cauchy


class TestJVPvsOriginal(unittest.TestCase):
    def test_J_grad_hessian_agreement(self):
        # Plane-stress strain history (uniaxial then biaxial), 100 steps.
        strain_increment = 0.02
        num_pts_per_increment = 50
        init_strain = strain_increment / num_pts_per_increment
        eps_xx = np.r_[np.zeros(1),
                       np.linspace(init_strain, strain_increment,
                                   num_pts_per_increment),
                       np.ones(num_pts_per_increment) * strain_increment]
        eps_yy = np.r_[np.zeros(1),
                       np.zeros(num_pts_per_increment) * strain_increment,
                       np.linspace(init_strain, strain_increment,
                                   num_pts_per_increment)]

        def_type = DefType.PLANE_STRESS
        ndims = def_type_ndims(def_type)
        num_steps = len(eps_xx) - 1
        I = np.eye(ndims)
        F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
        F[0, 0, :] += eps_xx
        F[1, 1, :] += eps_yy

        weight = np.zeros((3, 3))
        weight[0, 0] = 1.
        weight[1, 1] = 1.

        # Generate calibration data via a forward solve of the truth model.
        truth_problem = J2AnalyticalProblem(scale_params=True)
        truth_model = SmallElasticPlastic(truth_problem.J2_parameters, def_type)
        cauchy_data = compute_cauchy(truth_model, F)

        # Evaluate at a canonical perturbation away from the truth so that
        # gradient and Hessian are nontrivial.
        initial_guess = 0.1 * np.ones(truth_model.parameters.num_active_params)

        # Build the two objectives on independent models.
        orig_problem = J2AnalyticalProblem(scale_params=True)
        orig_model = SmallElasticPlastic(orig_problem.J2_parameters, def_type)
        orig_qoi = Calibration(orig_model, F, cauchy_data, weight)
        orig_obj = DirectAdjointObjective(orig_qoi)

        jvp_problem = J2AnalyticalProblem(scale_params=True)
        jvp_model = SmallElasticPlastic(jvp_problem.J2_parameters, def_type)
        jvp_qoi = Calibration(jvp_model, F, cauchy_data, weight)
        update_fun = make_newton_solve(jvp_model._residual,
                                       jvp_model._init_xi)
        jvp_obj = JVPObjective(jvp_qoi, update_fun)

        # Evaluate.
        J_orig, grad_orig, hess_orig = orig_obj.evaluate(initial_guess)
        J_jvp, grad_jvp = jvp_obj.evaluate_objective_and_grad(initial_guess)
        hess_jvp = jvp_obj.evaluate_hessian(initial_guess)

        # Assert agreement.
        np.testing.assert_allclose(J_orig, J_jvp, rtol=1e-12)
        np.testing.assert_allclose(grad_orig, grad_jvp, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hess_orig, hess_jvp, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestJVPvsOriginal)
    unittest.TextTestRunner(verbosity=2).run(suite)
