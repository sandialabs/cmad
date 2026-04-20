"""Round-trip checks for the text-file paths of ``load_history``."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cmad.io.deformation import load_history


class TestDeformationTextFiles(unittest.TestCase):
    def test_csv_roundtrip(self) -> None:
        N, n = 4, 2
        rng = np.random.default_rng(42)
        history = rng.standard_normal((N, n, n)).astype(np.float64)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            flat = history.reshape(N, n * n)
            np.savetxt(tmp / "F.csv", flat, delimiter=",")

            arr = load_history(
                {"history_file": "F.csv"}, tmp, expected_ndims=n,
            )
            self.assertEqual(arr.shape, (n, n, N))
            np.testing.assert_allclose(
                arr, history.transpose(1, 2, 0), rtol=0.0, atol=1e-12,
            )

    def test_txt_roundtrip(self) -> None:
        N, n = 5, 3
        rng = np.random.default_rng(7)
        history = rng.standard_normal((N, n, n)).astype(np.float64)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            flat = history.reshape(N, n * n)
            np.savetxt(tmp / "F.txt", flat)

            arr = load_history(
                {"history_file": "F.txt"}, tmp, expected_ndims=n,
            )
            self.assertEqual(arr.shape, (n, n, N))
            np.testing.assert_allclose(
                arr, history.transpose(1, 2, 0), rtol=0.0, atol=1e-12,
            )

    def test_csv_non_square_column_count_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bogus = np.arange(10, dtype=np.float64).reshape(2, 5)
            np.savetxt(tmp / "F.csv", bogus, delimiter=",")

            with self.assertRaises(ValueError) as cm:
                load_history(
                    {"history_file": "F.csv"}, tmp, expected_ndims=3,
                )
            self.assertIn("5 columns", str(cm.exception))
            self.assertIn("n*n", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
