"""Resolution tests for the model, QoI, and global residual namespaces."""

import unittest

from cmad.global_residuals.mechanics import Mechanics
from cmad.io.registry import (
    resolve_global_residual,
    resolve_model,
    resolve_qoi,
)
from cmad.models.elastic import Elastic
from cmad.qois.fe_load_match import FELoadMatch


class TestResolution(unittest.TestCase):
    def test_model_round_trip(self) -> None:
        self.assertIs(resolve_model("elastic"), Elastic)

    def test_qoi_round_trip(self) -> None:
        self.assertIs(resolve_qoi("fe_load_match"), FELoadMatch)

    def test_global_residual_round_trip(self) -> None:
        self.assertIs(resolve_global_residual("mechanics"), Mechanics)

    def test_unknown_model_raises_with_listing(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_model("not_a_real_model")
        msg = str(ctx.exception)
        self.assertIn("not_a_real_model", msg)
        self.assertIn("not available", msg)
        self.assertIn("elastic", msg)
        self.assertIn("hypo_elastic_plastic", msg)

    def test_unknown_qoi_raises_with_listing(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_qoi("not_a_real_qoi")
        msg = str(ctx.exception)
        self.assertIn("not_a_real_qoi", msg)
        self.assertIn("not available", msg)
        self.assertIn("fe_load_match", msg)

    def test_unknown_global_residual_raises_with_listing(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_global_residual("not_a_real_gr")
        msg = str(ctx.exception)
        self.assertIn("not_a_real_gr", msg)
        self.assertIn("not available", msg)
        self.assertIn("mechanics", msg)


if __name__ == "__main__":
    unittest.main()
