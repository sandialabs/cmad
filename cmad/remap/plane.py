"""Best fit plane of coplanar points and a deterministic frame on it.

A DIC measurement surface is nearly planar. Strain reconstruction needs the
displacement gradient within that plane, so points and displacements are
expressed in a 2D frame on it. The two axes are seeded from the global
coordinate axes rather than the point distribution, so the frame, and
therefore the strain components, are reproducible even when the points are
symmetric (a square grid has a degenerate plane SVD). On a face aligned with
the global axes the frame is just the two axes spanning it.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fit_plane_normal(
        points: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Centroid and unit normal of the best fit plane through ``points``.

    The normal is the least significant right singular vector of the
    centered points, with its sign fixed so the dominant component is
    positive (a reproducible choice independent of the SVD sign convention).
    """
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal * np.sign(normal[int(np.argmax(np.abs(normal)))])
    return centroid, normal


def in_plane_axes(normal: NDArray[np.floating]) -> NDArray[np.floating]:
    """A deterministic orthonormal basis ``(2, 3)`` spanning the plane.

    Seeds the first axis from the global axis least aligned with ``normal``,
    so the frame is fixed by the geometry rather than the point sampling.
    """
    seed = np.eye(3)[int(np.argmin(np.abs(normal)))]
    axis1 = seed - (seed @ normal) * normal
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = np.cross(normal, axis1)
    return np.stack([axis1, axis2])


def plane_coords(
        points: NDArray[np.floating],
        centroid: NDArray[np.floating],
        axes: NDArray[np.floating],
) -> NDArray[np.floating]:
    """2D coordinates of ``points`` within the plane ``(centroid, axes)``."""
    return (points - centroid) @ axes.T


def plane_vectors(
        vectors: NDArray[np.floating],
        axes: NDArray[np.floating],
) -> NDArray[np.floating]:
    """2D components of ``vectors`` within the plane (rotation only, no shift)."""
    return vectors @ axes.T
