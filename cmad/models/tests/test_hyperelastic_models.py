import numpy as np
import unittest

from functools import partial

from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.elastic import Elastic
from cmad.models.elastic_potential import (compute_cauchy_from_psi_b,
                                           compressible_neohookean_potential)
from cmad.models.elastic_stress import (compressible_neohookean_cauchy_stress,
                                        isotropic_linear_elastic_cauchy_stress)
from cmad.solver.nonlinear_solver import newton_solve
from cmad.test_support.test_problems import params_hyperelastic


class TestHyperelasticModels(unittest.TestCase):
    def test_potential_vs_analytic_uniaxial_cauchy(self, num_steps=100):
        kappa = 0.5  # MPa
        mu = 0.375  # MPa
        elastic_params = np.array([kappa, mu])
        params = params_hyperelastic(elastic_params)

        potential_model = Elastic(
            params,
            elastic_stress_fun=partial(
                compute_cauchy_from_psi_b,
                psi_b_fun=compressible_neohookean_potential),
            def_type=DefType.UNIAXIAL_STRESS)

        analytic_model = Elastic(
            params,
            elastic_stress_fun=compressible_neohookean_cauchy_stress,
            def_type=DefType.UNIAXIAL_STRESS)

        linear_elastic_model = Elastic(params,
                                       def_type=DefType.UNIAXIAL_STRESS)

        F = np.repeat(np.eye(1)[:, :, np.newaxis], num_steps + 1, axis=2)
        strain_11 = np.linspace(0, 0.4, num_steps + 1)
        F[0, 0, :] += strain_11

        # use the same container to store xi for each model
        # as they all have the same # of residuals and initial values
        num_residuals = potential_model.num_residuals
        xi_at_step = [[None] * num_residuals for ii in range(num_steps + 1)]

        potential_model.set_xi_to_init_vals()
        potential_model.store_xi(xi_at_step, potential_model.xi_prev(), 0)
        potential_cauchy = np.zeros((3, 3, num_steps + 1))
        potential_J = np.ones(num_steps + 1)

        analytic_model.set_xi_to_init_vals()
        analytic_model.store_xi(xi_at_step, analytic_model.xi_prev(), 0)
        analytic_cauchy = np.zeros((3, 3, num_steps + 1))
        analytic_J = np.ones(num_steps + 1)

        linear_elastic_model.set_xi_to_init_vals()
        linear_elastic_model.store_xi(xi_at_step,
                                      linear_elastic_model.xi_prev(), 0)
        linear_elastic_cauchy = np.zeros((3, 3, num_steps + 1))
        linear_elastic_J = np.ones(num_steps + 1)

        models = [linear_elastic_model, potential_model, analytic_model]
        cauchys = [linear_elastic_cauchy, potential_cauchy, analytic_cauchy]
        Js = [linear_elastic_J, potential_J, analytic_J]

        for model, cauchy, J in zip(models, cauchys, Js):

            for step in range(1, num_steps + 1):

                u = [F[:1, :1, step]]
                u_prev = [F[:1, :1, step - 1]]

                model.gather_global(u, u_prev)

                newton_solve(model)
                model.store_xi(xi_at_step, model.xi(), step)

                model.evaluate_cauchy()
                cauchy[:, :, step] = model.Sigma().copy()
                J[step] = u[0][0, 0] \
                    * xi_at_step[step][1][0] * xi_at_step[step][1][1]

                model.advance_xi()

        diff_tol = 1e-14
        cauchy_diff = potential_cauchy - analytic_cauchy
        J_diff = potential_J - analytic_J
        assert np.linalg.norm(cauchy_diff) < diff_tol
        assert np.linalg.norm(J_diff) < diff_tol

        # hyper_sigma_11 = analytic_cauchy[0, 0, :]
        # hyper_tau_11 = analytic_J * hyper_sigma_11
        # le_sigma_11 = linear_elastic_cauchy[0, 0, :]
        # print(f"\n\nNeohookean Cauchy stress = {hyper_sigma_11}\n")
        # print(f"Neohookean Kirchhoff stress = {hyper_tau_11}\n")
        # print(f"Linear elastic cauchy stress = {le_sigma_11}")

        # check consistency with 3D models
        potential_model_3D = Elastic(
            params,
            elastic_stress_fun=partial(
                compute_cauchy_from_psi_b,
                psi_b_fun=compressible_neohookean_potential),
            def_type=DefType.FULL_3D)
        potential_cauchy_3D = np.zeros((3, 3, num_steps + 1))

        analytic_model_3D = Elastic(
            params,
            elastic_stress_fun=compressible_neohookean_cauchy_stress,
            def_type=DefType.FULL_3D)
        analytic_cauchy_3D = np.zeros((3, 3, num_steps + 1))

        F_3D = np.zeros((3, 3, num_steps + 1))
        F_3D[0, 0, :] = F[0, 0, :]
        F_3D[1, 1, :] = \
            np.array([xi_at_step[step][1][0] for step in range(num_steps + 1)])
        F_3D[2, 2, :] = \
            np.array([xi_at_step[step][1][1] for step in range(num_steps + 1)])

        models = [potential_model_3D, analytic_model_3D]
        cauchys = [potential_cauchy_3D, analytic_cauchy_3D]

        for model, cauchy in zip(models, cauchys):

            for step in range(1, num_steps + 1):

                u = [F_3D[:, :, step]]
                u_prev = [F_3D[:, :, step - 1]]

                model.gather_global(u, u_prev)

                newton_solve(model)

                model.evaluate_cauchy()
                cauchy[:, :, step] = model.Sigma().copy()

                model.advance_xi()

        cauchy_3D_diff = potential_cauchy_3D - analytic_cauchy_3D
        assert np.linalg.norm(cauchy_3D_diff) < diff_tol
        cauchy_1D_3D_diff = potential_cauchy - potential_cauchy_3D
        assert np.linalg.norm(cauchy_1D_3D_diff) < diff_tol


if __name__ == "__main__":
    hyperelastic_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestHyperelasticModels)
    unittest.TextTestRunner(verbosity=2).run(hyperelastic_test_suite)
