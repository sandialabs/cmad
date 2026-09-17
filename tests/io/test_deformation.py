"""Round-trip checks for the text-file paths of ``load_history``,
and for the optional time history ``load_times`` reads beside it."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cmad.io.deformation import load_history, load_times


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
                {"history_file": str(tmp / "F.csv")}, expected_ndims=n,
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
                {"history_file": str(tmp / "F.txt")}, expected_ndims=n,
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
                    {"history_file": str(tmp / "F.csv")}, expected_ndims=3,
                )
            self.assertIn("5 columns", str(cm.exception))
            self.assertIn("n*n", str(cm.exception))


class TestLoadTimes(unittest.TestCase):
    """The three spellings of a time history, and the checks around them."""

    def test_no_time_keys_returns_none(self) -> None:
        self.assertIsNone(
            load_times({"history_file": "F.npy"}, num_steps=4),
        )

    def test_inline_times(self) -> None:
        times = load_times(
            {"times": [0.0, 0.1, 0.3, 0.6]}, num_steps=3,
        )
        np.testing.assert_allclose(times, [0.0, 0.1, 0.3, 0.6])

    def test_num_steps_and_step_size(self) -> None:
        times = load_times(
            {"num steps": 4, "step size": 0.25}, num_steps=4,
        )
        np.testing.assert_allclose(times, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_times_file_npy_and_txt(self) -> None:
        expected = np.array([0.0, 0.5, 1.5, 3.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "t.npy", expected)
            np.savetxt(tmp / "t.txt", expected)

            for name in ("t.npy", "t.txt"):
                times = load_times(
                    {"times file": str(tmp / name)}, num_steps=3,
                )
                np.testing.assert_allclose(times, expected)

    def test_times_file_unsupported_extension_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            load_times({"times file": "t.xml"}, num_steps=3)
        self.assertIn(".xml", str(cm.exception))
        self.assertIn("deformation.times file", str(cm.exception))

    def test_length_mismatch_with_F_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            load_times({"times": [0.0, 1.0, 2.0]}, num_steps=5)
        self.assertIn("3 times", str(cm.exception))
        self.assertIn("6 steps", str(cm.exception))

    def test_non_increasing_times_raise(self) -> None:
        # A repeated time is dt = 0, which a viscoplastic law divides by.
        with self.assertRaises(ValueError) as cm:
            load_times({"times": [0.0, 0.5, 0.5, 1.0]}, num_steps=3)
        self.assertIn("increase strictly", str(cm.exception))
        self.assertIn("times[1]", str(cm.exception))

        with self.assertRaises(ValueError):
            load_times({"times": [0.0, 0.5, 0.25]}, num_steps=2)


if __name__ == "__main__":
    unittest.main()
