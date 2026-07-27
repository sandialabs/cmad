"""Tests for surface strain reconstruction from a displacement cloud.

A linear displacement field has a constant gradient that GMLS reconstructs
exactly, so the strain comes out equal to the closed-form measure of that
gradient at every point. The field is laid on planes in several
orientations to check that the reconstruction works in the surface frame the
projection picks, and a tilted plane confirms the rotation invariants of the
strain tensor are preserved.
"""
import unittest

import numpy as np

from cmad.io.point_cloud import PointCloud
from cmad.remap.strain import cloud_strain

# H[a, k] = du_a / dx_k, a constant in-plane displacement gradient.
_H = np.array([[0.02, 0.01], [0.005, 0.03]])


def _plane_grid(n, to_3d):
    s = np.linspace(0.0, 1.0, n)
    a, b = np.meshgrid(s, s)
    return to_3d(np.column_stack([a.ravel(), b.ravel()]))


def _cloud(coords, in_plane_disp):
    """Two-step cloud: zero displacement, then ``in_plane_disp``."""
    frames = np.stack([np.zeros_like(in_plane_disp), in_plane_disp], axis=0)
    return PointCloud(
        coords=coords, times=np.array([0.0, 1.0]),
        fields={"displacement": frames},
    )


def _rotation(theta, phi):
    cz, sz, cy, sy = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return rot_z @ rot_y


class TestCloudStrain(unittest.TestCase):
    def _components(self, cloud, measure, step=1):
        out = cloud_strain(cloud, measure)
        return (
            out.fields[f"{measure}_11"][step, :, 0],
            out.fields[f"{measure}_22"][step, :, 0],
            out.fields[f"{measure}_12"][step, :, 0],
        )

    def test_small_strain_matches_closed_form_on_z_plane(self) -> None:
        coords = _plane_grid(
            15, lambda p: np.column_stack([p, np.zeros(len(p))]),
        )
        disp = np.column_stack([coords[:, :2] @ _H.T, np.zeros(len(coords))])
        e11, e22, e12 = self._components(_cloud(coords, disp), "small_strain")
        expected = 0.5 * (_H + _H.T)
        np.testing.assert_allclose(e11, expected[0, 0], atol=1e-10)
        np.testing.assert_allclose(e22, expected[1, 1], atol=1e-10)
        np.testing.assert_allclose(e12, expected[0, 1], atol=1e-10)

    def test_green_lagrange_matches_closed_form_on_z_plane(self) -> None:
        coords = _plane_grid(
            15, lambda p: np.column_stack([p, np.zeros(len(p))]),
        )
        disp = np.column_stack([coords[:, :2] @ _H.T, np.zeros(len(coords))])
        e11, e22, e12 = self._components(_cloud(coords, disp), "green_lagrange")
        deformation = np.eye(2) + _H
        expected = 0.5 * (deformation.T @ deformation - np.eye(2))
        np.testing.assert_allclose(e11, expected[0, 0], atol=1e-10)
        np.testing.assert_allclose(e22, expected[1, 1], atol=1e-10)
        np.testing.assert_allclose(e12, expected[0, 1], atol=1e-10)

    def test_initial_step_is_zero(self) -> None:
        coords = _plane_grid(
            12, lambda p: np.column_stack([p, np.zeros(len(p))]),
        )
        disp = np.column_stack([coords[:, :2] @ _H.T, np.zeros(len(coords))])
        e11, e22, e12 = self._components(
            _cloud(coords, disp), "small_strain", step=0,
        )
        for component in (e11, e22, e12):
            np.testing.assert_allclose(component, 0.0, atol=1e-12)

    def test_frame_follows_an_x_plane_surface(self) -> None:
        # Points on x = 1; the surface frame is the (y, z) global axes.
        coords = _plane_grid(
            15, lambda p: np.column_stack([np.ones(len(p)), p]),
        )
        in_plane = coords[:, 1:] @ _H.T
        disp = np.column_stack([np.zeros(len(coords)), in_plane])
        e11, e22, e12 = self._components(_cloud(coords, disp), "small_strain")
        expected = 0.5 * (_H + _H.T)
        np.testing.assert_allclose(e11, expected[0, 0], atol=1e-10)
        np.testing.assert_allclose(e22, expected[1, 1], atol=1e-10)
        np.testing.assert_allclose(e12, expected[0, 1], atol=1e-10)

    def test_tilted_plane_preserves_strain_invariants(self) -> None:
        rot = _rotation(0.3, 0.5)
        coords0 = _plane_grid(
            15, lambda p: np.column_stack([p, np.zeros(len(p))]),
        )
        disp0 = np.column_stack([coords0[:, :2] @ _H.T, np.zeros(len(coords0))])
        cloud = _cloud(coords0 @ rot.T, disp0 @ rot.T)
        e11, e22, e12 = self._components(cloud, "small_strain")
        expected = 0.5 * (_H + _H.T)
        np.testing.assert_allclose(
            e11 + e22, np.trace(expected), atol=1e-9,
        )
        np.testing.assert_allclose(
            e11 * e22 - e12 ** 2, np.linalg.det(expected), atol=1e-9,
        )

    def test_field_keys_name_measure_and_component(self) -> None:
        coords = _plane_grid(
            10, lambda p: np.column_stack([p, np.zeros(len(p))]),
        )
        disp = np.column_stack([coords[:, :2] @ _H.T, np.zeros(len(coords))])
        out = cloud_strain(_cloud(coords, disp), "green_lagrange")
        for component in ("11", "22", "12"):
            key = f"green_lagrange_{component}"
            self.assertIn(key, out.fields)
            self.assertEqual(out.fields[key].shape, (2, coords.shape[0], 1))
        self.assertIn("displacement", out.fields)

    def test_unknown_measure_raises(self) -> None:
        coords = _plane_grid(
            8, lambda p: np.column_stack([p, np.zeros(len(p))]),
        )
        disp = np.zeros_like(coords)
        with self.assertRaises(ValueError):
            cloud_strain(_cloud(coords, disp), "logarithmic")

    def test_missing_field_raises(self) -> None:
        coords = _plane_grid(
            8, lambda p: np.column_stack([p, np.zeros(len(p))]),
        )
        cloud = PointCloud(
            coords=coords, times=np.array([0.0]),
            fields={"velocity": np.zeros((1, coords.shape[0], 3))},
        )
        with self.assertRaises(ValueError):
            cloud_strain(cloud, "small_strain")


if __name__ == "__main__":
    unittest.main()
