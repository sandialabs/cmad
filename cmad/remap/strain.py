"""Surface strain from a DIC displacement cloud via GMLS gradients.

The displacement gradient within the measurement plane is reconstructed at
each point with GMLS (the cloud is both source and target), giving the
deformation gradient F = I + grad u in that plane. A chosen measure maps F
to a strain: small strain sym(grad u) or Green-Lagrange 1/2 (F^T F - I).

The strain is a symmetric 2x2 tensor in the plane. It is written as three
scalar fields keyed by the measure and the tensor component, for example
``green_lagrange_11``, ``green_lagrange_22``, ``green_lagrange_12``. Indices
1 and 2 are the two axes of the surface frame (equal to the global axes
spanning an aligned face).

Intended for dense, roughly uniform clouds, as DIC provides: strain is a
spatial derivative of measured data, so the GMLS support, which adapts to
the local point spacing through support_multiplier, sets the smoothing scale
against noise. On sparse or ragged clouds the reconstruction degrades.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cmad.io.point_cloud import PointCloud
from cmad.remap.gmls import build_gmls_operators
from cmad.remap.plane import (
    fit_plane_normal,
    in_plane_axes,
    plane_coords,
    plane_vectors,
)

_MEASURES = ("small_strain", "green_lagrange")


def _small_strain(grad_u: NDArray[np.floating]) -> NDArray[np.floating]:
    """Small strain sym(grad u), per point ``(..., 2, 2)``."""
    return 0.5 * (grad_u + np.swapaxes(grad_u, -1, -2))


def _green_lagrange(grad_u: NDArray[np.floating]) -> NDArray[np.floating]:
    """Green-Lagrange strain 1/2 (F^T F - I), F = I + grad u, per point."""
    deformation = grad_u + np.eye(2)
    return 0.5 * (np.swapaxes(deformation, -1, -2) @ deformation - np.eye(2))


def cloud_strain(
        cloud: PointCloud,
        measure: str = "green_lagrange",
        *,
        field: str = "displacement",
        poly_order: int = 2,
        support_multiplier: float = 1.6,
) -> PointCloud:
    """Return ``cloud`` with three scalar strain fields added.

    Reconstructs the displacement gradient within the measurement plane at
    every point with GMLS (the cloud is both source and target) and maps
    F = I + grad u to ``measure`` ("small_strain" or "green_lagrange"). Adds
    the tensor components as fields ``<measure>_11``, ``<measure>_22``,
    ``<measure>_12``, each of shape ``(num_steps, num_points, 1)``.
    """
    if measure not in _MEASURES:
        raise ValueError(
            f"cloud_strain: unknown measure '{measure}'; "
            f"expected one of {_MEASURES}"
        )
    if field not in cloud.fields:
        raise ValueError(
            f"cloud_strain: cloud has no field '{field}' "
            f"(has: {sorted(cloud.fields)})"
        )

    axes: NDArray[np.floating] | None
    if cloud.dim == 3:
        centroid, normal = fit_plane_normal(cloud.coords)
        axes = in_plane_axes(normal)
        coords_2d = plane_coords(cloud.coords, centroid, axes)
    else:
        axes = None
        coords_2d = cloud.coords

    ops = build_gmls_operators(
        coords_2d, coords_2d, poly_order=poly_order,
        support_multiplier=support_multiplier,
    )
    measure_fn = (
        _small_strain if measure == "small_strain" else _green_lagrange
    )

    disp = cloud.fields[field]
    shape = (cloud.num_steps, cloud.num_points, 1)
    components = {"11": np.zeros(shape), "22": np.zeros(shape),
                 "12": np.zeros(shape)}
    for step in range(cloud.num_steps):
        u_2d = (
            plane_vectors(disp[step], axes) if axes is not None
            else disp[step]
        )
        grad_u = np.stack(
            [np.stack([ops.grad[k] @ u_2d[:, a] for k in range(2)], axis=1)
             for a in range(2)],
            axis=1,
        )  # grad_u[point, a, k] = du_a / dx_k
        strained = measure_fn(grad_u)
        components["11"][step, :, 0] = strained[:, 0, 0]
        components["22"][step, :, 0] = strained[:, 1, 1]
        components["12"][step, :, 0] = strained[:, 0, 1]

    new_fields = dict(cloud.fields)
    for label, values in components.items():
        new_fields[f"{measure}_{label}"] = values
    return PointCloud(
        coords=cloud.coords, times=cloud.times, fields=new_fields,
    )
