"""Contract tests for the QoI ABC's named first-derivative methods.

Exercises ``dJ_dxi`` / ``dJ_dxi_prev`` / ``dJ_dp`` / ``dJ_dU`` /
``dJ_dU_prev`` for callability and pytree-structure-parallel-to-
input. Test-local toy QoI subclass whose ``qoi_fun`` references
every differentiable input via tree_leaves so all five named
derivatives are non-trivial regardless of model.
"""
import unittest

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_leaves, tree_structure

from cmad.models.deformation_types import DefType
from cmad.models.global_fields import mp_U_from_F
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.qois.qoi import QoI
from tests.support.test_problems import J2AnalyticalProblem


def _make_J2_model() -> SmallElasticPlastic:
    problem = J2AnalyticalProblem()
    model = SmallElasticPlastic(problem.J2_parameters, DefType.FULL_3D)
    model.set_xi_to_init_vals()
    F = np.eye(3) + 0.001 * np.eye(3)
    F_prev = np.eye(3)
    model.gather_global(mp_U_from_F(F), mp_U_from_F(F_prev))
    return model


class _AllInputsQoI(QoI):
    """Touches every differentiable input + data + weight."""
    def __init__(self, model: SmallElasticPlastic) -> None:
        self._model = model
        self._data = np.zeros(1)
        self._weight = np.ones(1)

        def qoi_fun(xi, xi_prev, params, U, U_prev, data, weight):
            leaves = (
                tree_leaves(xi) + tree_leaves(xi_prev)
                + tree_leaves(params) + tree_leaves(U)
                + tree_leaves(U_prev)
            )
            return (sum(jnp.sum(leaf) for leaf in leaves)
                    + jnp.sum(data * weight))

        super().__init__(qoi_fun)

    def data_at_step(self, step):
        return self._data

    def weight_at_step(self, step):
        return self._weight


class TestQoINamedDerivatives(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _make_J2_model()
        self.qoi = _AllInputsQoI(self.model)
        self.data = jnp.ones(1)
        self.weight = jnp.ones(1)

    def test_dJ_dxi_pytree_structure_parallels_xi(self):
        xi, xi_prev, params, U, U_prev = self.model.variables()
        result = self.qoi.dJ_dxi(
            xi, xi_prev, params, U, U_prev, self.data, self.weight)
        self.assertEqual(tree_structure(result), tree_structure(xi))

    def test_dJ_dxi_prev_pytree_structure_parallels_xi_prev(self):
        xi, xi_prev, params, U, U_prev = self.model.variables()
        result = self.qoi.dJ_dxi_prev(
            xi, xi_prev, params, U, U_prev, self.data, self.weight)
        self.assertEqual(tree_structure(result), tree_structure(xi_prev))

    def test_dJ_dp_pytree_structure_parallels_params(self):
        xi, xi_prev, params, U, U_prev = self.model.variables()
        result = self.qoi.dJ_dp(
            xi, xi_prev, params, U, U_prev, self.data, self.weight)
        self.assertEqual(tree_structure(result), tree_structure(params))

    def test_dJ_dU_pytree_structure_parallels_U(self):
        xi, xi_prev, params, U, U_prev = self.model.variables()
        result = self.qoi.dJ_dU(
            xi, xi_prev, params, U, U_prev, self.data, self.weight)
        self.assertEqual(tree_structure(result), tree_structure(U))

    def test_dJ_dU_prev_pytree_structure_parallels_U_prev(self):
        xi, xi_prev, params, U, U_prev = self.model.variables()
        result = self.qoi.dJ_dU_prev(
            xi, xi_prev, params, U, U_prev, self.data, self.weight)
        self.assertEqual(tree_structure(result), tree_structure(U_prev))


if __name__ == "__main__":
    unittest.main()
