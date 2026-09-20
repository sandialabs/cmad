import unittest

import jax.numpy as jnp
import numpy as np

from cmad.models.var_types import (
    VarType,
    get_dev_sym_tensor_from_vector,
    get_num_eqs,
    get_vector_from_dev_sym_tensor,
)


class TestDevSymTensor(unittest.TestCase):

    def test_round_trip_and_zero_trace(self):
        for ndims, vec in ((3, [1., 2., 3., 4., 5.]), (2, [1., 2., 3.])):
            tensor = get_dev_sym_tensor_from_vector(jnp.array(vec), ndims)
            self.assertEqual(tensor.shape, (3, 3))
            np.testing.assert_array_equal(tensor, tensor.T)
            self.assertEqual(float(jnp.trace(tensor)), 0.0)
            np.testing.assert_array_equal(
                get_vector_from_dev_sym_tensor(tensor, ndims), jnp.array(vec))
            self.assertEqual(
                get_num_eqs(VarType.DEV_SYM_TENSOR, ndims), len(vec))


if __name__ == "__main__":
    unittest.main()
