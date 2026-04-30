"""Unit tests for `cmad.fem.bcs.DirichletBC`.

The dataclass surface is small (no resolution, no mesh handle); these
tests cover construction round-trip and the three `__post_init__`
validation paths. Resolution behavior (sideset walk, intra-BC dedup,
broadcasting None / Sequence / Callable values to flat arrays) is
exercised in `test_dof.py` because it lives in the DofMap.
"""
import unittest

import numpy as np

from cmad.fem.bcs import DirichletBC


class TestDirichletBC(unittest.TestCase):

    def test_dataclass_round_trip(self):
        bc = DirichletBC(
            sideset_names=["xmin_sides"],
            field_name="u",
            dofs=[0, 1, 2],
            values=[0.0, 0.5, 1.0],
        )
        self.assertEqual(list(bc.sideset_names), ["xmin_sides"])
        self.assertEqual(bc.field_name, "u")
        self.assertEqual(list(bc.dofs), [0, 1, 2])
        self.assertEqual(list(bc.values), [0.0, 0.5, 1.0])

    def test_multi_sideset_round_trip(self):
        bc = DirichletBC(
            sideset_names=["xmin_sides", "xmax_sides", "ymin_sides"],
            field_name="u",
            dofs=[0],
        )
        self.assertEqual(
            list(bc.sideset_names),
            ["xmin_sides", "xmax_sides", "ymin_sides"],
        )

    def test_default_homogeneous(self):
        bc = DirichletBC(
            sideset_names=["zmin_sides"], field_name="u", dofs=[2],
        )
        self.assertIsNone(bc.values)

    def test_callable_values_accepted(self):
        def vals(coords, t):
            return np.zeros((coords.shape[0], 1))
        bc = DirichletBC(
            sideset_names=["xmin_sides"],
            field_name="u",
            dofs=[0],
            values=vals,
        )
        self.assertTrue(callable(bc.values))


class TestPostInitValidation(unittest.TestCase):

    def test_empty_sideset_names_raises(self):
        with self.assertRaisesRegex(ValueError, "sideset_names .* non-empty"):
            DirichletBC(sideset_names=[], field_name="u", dofs=[0])

    def test_empty_dofs_raises(self):
        with self.assertRaisesRegex(ValueError, "dofs .* non-empty"):
            DirichletBC(
                sideset_names=["xmin_sides"], field_name="u", dofs=[],
            )

    def test_sequence_length_mismatch_raises(self):
        with self.assertRaisesRegex(
            ValueError, "values length .* does not match dofs length"
        ):
            DirichletBC(
                sideset_names=["xmin_sides"],
                field_name="u",
                dofs=[0, 1],
                values=[0.0],
            )


if __name__ == "__main__":
    unittest.main()
