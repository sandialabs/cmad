"""Unit tests for `cmad.fem.bcs.DirichletBC`.

The dataclass surface is small (no resolution, no mesh handle); these
tests cover construction round-trip and the two `__post_init__`
validation paths. Resolution behavior (broadcasting None / Sequence /
Callable values to flat arrays) is exercised in `test_dof.py` because
it lives in the DofMap.
"""
import unittest

import numpy as np

from cmad.fem.bcs import DirichletBC


class TestDirichletBC(unittest.TestCase):

    def test_dataclass_round_trip(self):
        bc = DirichletBC(
            nodeset_name="xmin_nodes",
            field_name="u",
            dofs=[0, 1, 2],
            values=[0.0, 0.5, 1.0],
        )
        self.assertEqual(bc.nodeset_name, "xmin_nodes")
        self.assertEqual(bc.field_name, "u")
        self.assertEqual(list(bc.dofs), [0, 1, 2])
        self.assertEqual(list(bc.values), [0.0, 0.5, 1.0])

    def test_default_homogeneous(self):
        bc = DirichletBC(
            nodeset_name="zmin_nodes", field_name="u", dofs=[2]
        )
        self.assertIsNone(bc.values)

    def test_callable_values_accepted(self):
        def vals(coords, t):
            return np.zeros((coords.shape[0], 1))
        bc = DirichletBC(
            nodeset_name="xmin_nodes",
            field_name="u",
            dofs=[0],
            values=vals,
        )
        self.assertTrue(callable(bc.values))


class TestPostInitValidation(unittest.TestCase):

    def test_empty_dofs_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            DirichletBC(nodeset_name="xmin_nodes", field_name="u", dofs=[])

    def test_sequence_length_mismatch_raises(self):
        with self.assertRaisesRegex(
            ValueError, "values length .* does not match dofs length"
        ):
            DirichletBC(
                nodeset_name="xmin_nodes",
                field_name="u",
                dofs=[0, 1],
                values=[0.0],
            )


if __name__ == "__main__":
    unittest.main()
