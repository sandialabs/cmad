"""Tests for :mod:`cmad.io.calibration_data`."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cmad.io.calibration_data import CalibrationData
from cmad.io.qoi_data import load_roi


def _record() -> CalibrationData:
    rng = np.random.default_rng(0)
    return CalibrationData(
        times=np.array([0.0, 1.0, 2.5, 4.0]),
        frame_ids=np.array([-1, 3, 5, 8], dtype=np.intp),
        load=np.array([0.0, 10.0, 25.0, 40.0]),
        node_ids=np.array([2, 5, 7, 11, 12], dtype=np.intp),
        sidesets={
            "ymin_sides": np.array([2, 5], dtype=np.intp),
            "ymax_sides": np.array([11, 12], dtype=np.intp),
        },
        values=rng.standard_normal((4, 5, 2)),
        mesh_file="mesh.msh",
        mesh_num_nodes=20,
        roi={
            "elements": np.array([0, 3], dtype=np.intp),
            "band": np.float64(0.5),
            "max_gap": np.float64(0.3),
        },
    )


class TestCalibrationData(unittest.TestCase):
    def test_round_trip(self) -> None:
        data = _record()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration_data.npz"
            data.write(path)
            back = CalibrationData.read(path)
        np.testing.assert_array_equal(back.times, data.times)
        np.testing.assert_array_equal(back.frame_ids, data.frame_ids)
        np.testing.assert_array_equal(back.load, data.load)
        np.testing.assert_array_equal(back.node_ids, data.node_ids)
        self.assertEqual(list(back.sidesets), list(data.sidesets))
        for name, ids in data.sidesets.items():
            np.testing.assert_array_equal(back.sidesets[name], ids)
        np.testing.assert_array_equal(back.values, data.values)
        self.assertEqual(back.mesh_file, "mesh.msh")
        self.assertEqual(back.mesh_num_nodes, 20)
        self.assertEqual(set(back.roi), {"elements", "band", "max_gap"})
        np.testing.assert_array_equal(back.roi["elements"], [0, 3])
        self.assertEqual(float(back.roi["band"]), 0.5)
        self.assertEqual(float(back.roi["max_gap"]), 0.3)

    def test_frame_lookup(self) -> None:
        data = _record()
        self.assertEqual(data.frame_at(2.5), 2)
        self.assertIsNone(data.frame_at(2.4))
        self.assertEqual(data.nearest_frame(2.4), 2)
        np.testing.assert_array_equal(data.frames_within(0.0, 4.0), [1, 2])
        np.testing.assert_array_equal(data.frames_at([1.0, 4.0]), [1, 3])
        with self.assertRaisesRegex(ValueError, "3 is not a frame"):
            data.frames_at([1.0, 3.0])

    def test_check_mesh(self) -> None:
        data = _record()
        data.check_mesh(20)
        with self.assertRaisesRegex(ValueError, "mesh has 21"):
            data.check_mesh(21)

    def test_rows_by_node_ids(self) -> None:
        data = _record()
        rows = data.rows([1, 3], [11, 5])
        np.testing.assert_array_equal(rows, data.values[[1, 3]][:, [3, 1]])
        np.testing.assert_array_equal(data.rows([2]), data.values[2:3])

    def test_missing_node_raises(self) -> None:
        data = _record()
        with self.assertRaises(ValueError) as ctx:
            data.positions([2, 6])
        self.assertIn("node 6", str(ctx.exception))
        self.assertIn("mesh.msh", str(ctx.exception))

    def test_sideset_outside_node_ids_rejected(self) -> None:
        data = _record()
        with self.assertRaises(ValueError):
            CalibrationData(
                times=data.times, frame_ids=data.frame_ids, load=data.load,
                node_ids=data.node_ids,
                sidesets={"ymin_sides": np.array([2, 9], dtype=np.intp)},
                values=data.values, mesh_file=data.mesh_file,
                mesh_num_nodes=data.mesh_num_nodes,
            )

    def test_load_roi_reads_the_archive(self) -> None:
        data = _record()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration_data.npz"
            data.write(path)
            roi = load_roi({"roi_file": str(path)}, ndims=2)
        np.testing.assert_array_equal(roi, [0, 3])


if __name__ == "__main__":
    unittest.main()
