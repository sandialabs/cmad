"""Round-trip tests for cmad.models.elastic_constants.ElasticConstants."""
import itertools
import unittest

import numpy as np

from cmad.models.elastic_constants import ElasticConstants

# attribute name on ElasticConstants -> deck/input key name
_NAMES = {"E": "E", "nu": "nu", "mu": "mu", "kappa": "kappa", "lmbda": "lambda"}


class TestElasticConstants(unittest.TestCase):

    def test_round_trip_all_pairs(self) -> None:
        """Any two of the five constants reproduce all five."""
        reference = ElasticConstants.from_params({"E": 200000.0, "nu": 0.3})
        values = {attr: float(getattr(reference, attr)) for attr in _NAMES}

        for a, b in itertools.combinations(_NAMES, 2):
            elastic = {_NAMES[a]: values[a], _NAMES[b]: values[b]}
            derived = ElasticConstants.from_params(elastic)
            for attr in _NAMES:
                np.testing.assert_allclose(
                    float(getattr(derived, attr)), values[attr],
                    rtol=1e-9, err_msg=f"input pair ({a}, {b}) -> {attr}",
                )

    def test_requires_exactly_two(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two"):
            ElasticConstants.from_params({"E": 1.0})
        with self.assertRaisesRegex(ValueError, "exactly two"):
            ElasticConstants.from_params({"E": 1.0, "nu": 0.3, "mu": 1.0})


if __name__ == "__main__":
    unittest.main()
