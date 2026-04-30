"""Contract tests for the Model ABC's named first-derivative methods.

Exercises ``dC_dxi`` / ``dC_dxi_prev`` / ``dC_dp`` / ``dC_dU`` /
``dC_dU_prev`` for callability and pytree-structure-parallel-to-input.
SmallElasticPlastic + J2 parameters because path-dependent
plasticity makes ``xi_prev`` load-bearing (Elastic doesn't depend
on xi_prev so derivatives w.r.t. it would be uniformly zero and
test less of the wiring).
"""
import unittest

import numpy as np
from jax.tree_util import tree_structure

from cmad.models.deformation_types import DefType
from cmad.models.global_fields import mp_U_from_F
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from tests.support.test_problems import J2AnalyticalProblem


def _make_J2_model() -> SmallElasticPlastic:
    problem = J2AnalyticalProblem()
    model = SmallElasticPlastic(problem.J2_parameters, DefType.FULL_3D)
    model.set_xi_to_init_vals()
    F = np.eye(3) + 0.001 * np.eye(3)
    F_prev = np.eye(3)
    model.gather_global(mp_U_from_F(F), mp_U_from_F(F_prev))
    return model


class TestModelNamedDerivatives(unittest.TestCase):
    def test_dC_dxi_pytree_structure_parallels_xi(self):
        model = _make_J2_model()
        xi, xi_prev, params, U, U_prev = model.variables()
        result = model.dC_dxi(xi, xi_prev, params, U, U_prev)
        self.assertEqual(tree_structure(result), tree_structure(xi))

    def test_dC_dxi_prev_pytree_structure_parallels_xi_prev(self):
        model = _make_J2_model()
        xi, xi_prev, params, U, U_prev = model.variables()
        result = model.dC_dxi_prev(xi, xi_prev, params, U, U_prev)
        self.assertEqual(tree_structure(result), tree_structure(xi_prev))

    def test_dC_dp_pytree_structure_parallels_params(self):
        model = _make_J2_model()
        xi, xi_prev, params, U, U_prev = model.variables()
        result = model.dC_dp(xi, xi_prev, params, U, U_prev)
        self.assertEqual(tree_structure(result), tree_structure(params))

    def test_dC_dU_pytree_structure_parallels_U(self):
        model = _make_J2_model()
        xi, xi_prev, params, U, U_prev = model.variables()
        result = model.dC_dU(xi, xi_prev, params, U, U_prev)
        self.assertEqual(tree_structure(result), tree_structure(U))

    def test_dC_dU_prev_pytree_structure_parallels_U_prev(self):
        model = _make_J2_model()
        xi, xi_prev, params, U, U_prev = model.variables()
        result = model.dC_dU_prev(xi, xi_prev, params, U, U_prev)
        self.assertEqual(tree_structure(result), tree_structure(U_prev))


if __name__ == "__main__":
    unittest.main()
