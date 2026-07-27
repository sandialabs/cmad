"""Tests for :mod:`cmad.io.point_cloud`: the PointCloud container and its
text / HDF5 I/O plus the XDMF index.

Round-trip tests confirm coords, times, and every field survive a
write -> read through both the HDF5 layout and the text layout. The XDMF
tests parse the written XML and check the structure ParaView relies on (a
temporal collection, one Polyvertex grid per step, the geometry type, and a
HyperSlab per field selecting that step's slice of the HDF5 dataset).
"""
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from cmad.io.point_cloud import (
    PointCloud,
    read_point_cloud,
    read_point_cloud_hdf5,
    read_point_cloud_text,
    write_point_cloud,
    write_point_cloud_hdf5,
    write_point_cloud_text,
)


def _sample_cloud() -> PointCloud:
    """A 4-point, 3-step 3D cloud with a vector and a scalar field."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
    )
    times = np.array([0.0, 0.5, 1.0])
    displacement = np.stack([t * coords for t in times], axis=0)
    sigma = np.stack([np.full((4, 1), t) for t in times], axis=0)
    return PointCloud(
        coords=coords, times=times,
        fields={"displacement": displacement, "sigma": sigma},
    )


class TestPointCloudValidation(unittest.TestCase):
    def test_properties(self):
        cloud = _sample_cloud()
        self.assertEqual(cloud.num_points, 4)
        self.assertEqual(cloud.num_steps, 3)
        self.assertEqual(cloud.dim, 3)

    def test_bad_coords_shape_raises(self):
        with self.assertRaises(ValueError):
            PointCloud(np.zeros((4,)), np.zeros(2), {})
        with self.assertRaises(ValueError):
            PointCloud(np.zeros((4, 4)), np.zeros(2), {})

    def test_bad_times_shape_raises(self):
        with self.assertRaises(ValueError):
            PointCloud(np.zeros((4, 3)), np.zeros((2, 1)), {})

    def test_field_shape_mismatch_raises(self):
        coords, times = np.zeros((4, 3)), np.zeros(3)
        with self.assertRaises(ValueError):  # wrong num_points
            PointCloud(coords, times, {"d": np.zeros((3, 5, 3))})
        with self.assertRaises(ValueError):  # wrong num_steps
            PointCloud(coords, times, {"d": np.zeros((2, 4, 3))})
        with self.assertRaises(ValueError):  # not 3D
            PointCloud(coords, times, {"d": np.zeros((3, 4))})


class TestHDF5RoundTrip(unittest.TestCase):
    def test_round_trip(self):
        cloud = _sample_cloud()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.h5"
            write_point_cloud_hdf5(path, cloud)
            back = read_point_cloud_hdf5(path)
        self.assertTrue(np.allclose(back.coords, cloud.coords))
        self.assertTrue(np.allclose(back.times, cloud.times))
        self.assertEqual(set(back.fields), {"displacement", "sigma"})
        for name in cloud.fields:
            self.assertTrue(
                np.allclose(back.fields[name], cloud.fields[name]),
            )

    def test_write_point_cloud_emits_both_files(self):
        cloud = _sample_cloud()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.h5"
            write_point_cloud(path, cloud)
            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".xdmf").exists())
            self.assertTrue(
                np.allclose(read_point_cloud_hdf5(path).coords, cloud.coords),
            )

    def test_write_point_cloud_can_skip_xdmf(self):
        cloud = _sample_cloud()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.h5"
            write_point_cloud(path, cloud, write_xdmf=False)
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".xdmf").exists())


class TestTextIO(unittest.TestCase):
    def test_single_frame(self):
        rows = np.array(
            [
                [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
                [1.0, 0.0, 0.0, 0.4, 0.5, 0.6],
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.txt"
            np.savetxt(path, rows)
            cloud = read_point_cloud_text(
                path, coord_cols=[0, 1, 2],
                field_cols={"displacement": [3, 4, 5]},
            )
        self.assertEqual((cloud.num_points, cloud.num_steps), (2, 1))
        self.assertTrue(np.allclose(cloud.coords, rows[:, :3]))
        self.assertTrue(
            np.allclose(cloud.fields["displacement"][0], rows[:, 3:]),
        )
        self.assertTrue(np.allclose(cloud.times, [0.0]))

    def test_multi_frame_default_times(self):
        frames = [
            np.array(
                [
                    [0.0, 0.0, 0.0, float(k), 0.0, 0.0],
                    [1.0, 0.0, 0.0, float(k) + 0.1, 0.0, 0.0],
                ],
            )
            for k in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for k, rows in enumerate(frames):
                p = Path(tmp) / f"frame_{k}.txt"
                np.savetxt(p, rows)
                paths.append(p)
            cloud = read_point_cloud_text(
                paths, coord_cols=[0, 1, 2],
                field_cols={"displacement": [3, 4, 5]},
            )
        self.assertEqual(cloud.num_steps, 3)
        self.assertTrue(np.allclose(cloud.times, [0, 1, 2]))
        self.assertTrue(np.allclose(cloud.coords, frames[0][:, :3]))
        for k in range(3):
            self.assertTrue(
                np.allclose(cloud.fields["displacement"][k], frames[k][:, 3:]),
            )

    def test_explicit_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for k in range(2):
                p = Path(tmp) / f"frame_{k}.txt"
                np.savetxt(p, np.array([[0.0, 0.0, 0.0, float(k)]]))
                paths.append(p)
            cloud = read_point_cloud_text(
                paths, coord_cols=[0, 1, 2],
                field_cols={"d": [3]}, times=[0.0, 2.5],
            )
        self.assertTrue(np.allclose(cloud.times, [0.0, 2.5]))

    def test_mismatched_point_counts_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            p0 = Path(tmp) / "a.txt"
            p1 = Path(tmp) / "b.txt"
            np.savetxt(p0, np.array([[0.0, 0.0, 0.0, 1.0]]))
            np.savetxt(p1, np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0]]))
            with self.assertRaises(ValueError):
                read_point_cloud_text(
                    [p0, p1], coord_cols=[0, 1, 2], field_cols={"d": [3]},
                )

    def test_writer_round_trip(self):
        cloud = _sample_cloud()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.txt"
            write_point_cloud_text(path, cloud, field="displacement")
            paths = [
                path.with_name(f"out_{k:04d}.txt")
                for k in range(cloud.num_steps)
            ]
            for p in paths:
                self.assertTrue(p.exists())
            back = read_point_cloud_text(
                paths, coord_cols=[0, 1, 2],
                field_cols={"displacement": [3, 4, 5]},
            )
        self.assertTrue(np.allclose(back.coords, cloud.coords))
        self.assertTrue(
            np.allclose(
                back.fields["displacement"], cloud.fields["displacement"],
            ),
        )


class TestReadDispatch(unittest.TestCase):
    def test_hdf5_dispatch(self):
        cloud = _sample_cloud()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.h5"
            write_point_cloud_hdf5(path, cloud)
            back = read_point_cloud(path)
        self.assertTrue(np.allclose(back.coords, cloud.coords))

    def test_text_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.txt"
            np.savetxt(path, np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]]))
            back = read_point_cloud(
                path, coord_cols=[0, 1, 2],
                field_cols={"displacement": [3, 4, 5]},
            )
        self.assertEqual(back.num_points, 1)

    def test_unsupported_extension_raises(self):
        with self.assertRaises(ValueError):
            read_point_cloud("foo.bin")


class TestXDMFStructure(unittest.TestCase):
    def test_temporal_collection_3d(self):
        cloud = _sample_cloud()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.h5"
            write_point_cloud(path, cloud)
            root = ET.parse(path.with_suffix(".xdmf")).getroot()

        self.assertEqual(root.tag, "Xdmf")
        collection = root.find("./Domain/Grid")
        self.assertEqual(collection.get("CollectionType"), "Temporal")
        grids = collection.findall("Grid")
        self.assertEqual(len(grids), cloud.num_steps)

        grid0 = grids[0]
        topology = grid0.find("Topology")
        self.assertEqual(topology.get("TopologyType"), "Polyvertex")
        self.assertEqual(
            topology.get("NumberOfElements"), str(cloud.num_points),
        )
        geometry = grid0.find("Geometry")
        self.assertEqual(geometry.get("GeometryType"), "XYZ")
        self.assertIn(":/coords", geometry.find("DataItem").text)

        attributes = {a.get("Name"): a for a in grid0.findall("Attribute")}
        self.assertEqual(
            attributes["displacement"].get("AttributeType"), "Vector",
        )
        self.assertEqual(attributes["sigma"].get("AttributeType"), "Scalar")
        item = attributes["displacement"].find("DataItem")
        self.assertIsNone(item.get("ItemType"))
        self.assertEqual(item.get("Dimensions"), f"{cloud.num_points} 3")
        self.assertTrue(item.text.strip().endswith("/fields/displacement/0000"))

    def test_each_step_references_its_dataset(self):
        cloud = _sample_cloud()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.h5"
            write_point_cloud(path, cloud)
            root = ET.parse(path.with_suffix(".xdmf")).getroot()
        grids = root.find("./Domain/Grid").findall("Grid")
        for step, grid in enumerate(grids):
            attr = {a.get("Name"): a for a in grid.findall("Attribute")}
            item = attr["displacement"].find("DataItem")
            self.assertTrue(
                item.text.strip().endswith(
                    f"/fields/displacement/{step:04d}",
                ),
            )

    def test_geometry_type_2d(self):
        cloud = PointCloud(
            coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            times=np.array([0.0]),
            fields={"displacement": np.zeros((1, 3, 2))},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.h5"
            write_point_cloud(path, cloud)
            root = ET.parse(path.with_suffix(".xdmf")).getroot()
        geometry = root.find("./Domain/Grid/Grid/Geometry")
        self.assertEqual(geometry.get("GeometryType"), "XY")


if __name__ == "__main__":
    unittest.main()
