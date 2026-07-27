"""Locate points in a sideset's facets and sample an FE field there.

Given measurement coordinates on a sideset, find the facet that contains
each point and interpolate the solved FE displacement, producing a point
cloud of the surface displacement. Used to manufacture synthetic DIC data
from an FE solve. Triangular facets interpolate with barycentric (P1)
weights; quadrilateral facets invert the bilinear map to give exact Q1
weights.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from cmad.fem.dof import GlobalDofMap
from cmad.fem.mesh import Mesh
from cmad.io.point_cloud import PointCloud

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem, FEState


def _sideset_facet_nodes(
        mesh: Mesh, dof_map: GlobalDofMap, field_name: str, sideset: str,
) -> NDArray[np.intp]:
    """Global node ids of each facet on ``sideset`` (one row per facet)."""
    name_to_idx = {fl.name: i for i, fl in enumerate(dof_map.field_layouts)}
    fe = dof_map.field_layouts[name_to_idx[field_name]].finite_element
    if sideset not in mesh.side_sets:
        raise KeyError(
            f"sideset '{sideset}' not in mesh.side_sets (known: "
            f"{sorted(mesh.side_sets)})"
        )
    rows = [
        mesh.connectivity[
            int(elem_id), fe.side_basis_fns(int(local_side_id)),
        ].astype(np.intp)
        for elem_id, local_side_id in mesh.side_sets[sideset]
    ]
    return np.asarray(rows, dtype=np.intp)


def _barycentric(point: NDArray[np.floating], tri: NDArray[np.floating],
                 ) -> NDArray[np.floating]:
    """Barycentric weights of ``point`` projected onto triangle ``tri`` (3, 3)."""
    p0, p1, p2 = tri
    v0, v1, v2 = p1 - p0, p2 - p0, point - p0
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.array([1.0 - v - w, v, w])


def _cross2(u: NDArray[np.floating], v: NDArray[np.floating]) -> float:
    """Scalar cross product of two 2D vectors."""
    return float(u[0] * v[1] - u[1] * v[0])


def _inverse_bilinear(point: NDArray[np.floating], quad: NDArray[np.floating],
                      ) -> NDArray[np.floating]:
    r"""Q1 weights of ``point`` within a quad facet ``quad`` (4, 3).

    The four corners are in cyclic order (as :meth:`side_basis_fns`
    returns them), mapped to the unit-square corners ``(0,0), (1,0),
    (1,1), (0,1)``. The facet is projected onto its own plane and the
    bilinear map ``Q = P0 + xi*A + eta*B + xi*eta*C`` is inverted in
    closed form: eliminating ``xi`` leaves a quadratic in ``eta`` (linear
    when the facet is a parallelogram, ``C = 0``). The returned weights are
    the bilinear shape functions at ``(xi, eta)`` and sum to one.
    """
    p0, p1, p2, p3 = quad
    normal = np.cross(p2 - p0, p3 - p1)
    axis1 = (p1 - p0) / np.linalg.norm(p1 - p0)
    axis2 = np.cross(normal, axis1)
    axis2 = axis2 / np.linalg.norm(axis2)

    def to_plane(x: NDArray[np.floating]) -> NDArray[np.floating]:
        delta = x - p0
        return np.array([delta @ axis1, delta @ axis2])

    a = to_plane(p1)
    b = to_plane(p3)
    c = to_plane(p2) - a - b
    h = to_plane(point)

    k2 = -_cross2(b, c)
    k1 = _cross2(h, c) - _cross2(b, a)
    k0 = _cross2(h, a)
    if abs(k2) <= 1e-12 * (abs(k1) + 1.0):
        etas = [-k0 / k1]
    else:
        disc = max(k1 * k1 - 4.0 * k2 * k0, 0.0)
        root = np.sqrt(disc)
        etas = [(-k1 + root) / (2.0 * k2), (-k1 - root) / (2.0 * k2)]

    best_weights, best_outside = np.zeros(4), np.inf
    for eta in etas:
        denom = a + eta * c
        xi = float((h - eta * b) @ denom / (denom @ denom))
        weights = np.array([
            (1.0 - xi) * (1.0 - eta), xi * (1.0 - eta),
            xi * eta, (1.0 - xi) * eta,
        ])
        outside = max(0.0, -float(weights.min()))
        if outside < best_outside:
            best_weights, best_outside = weights, outside
    return best_weights


def locate_points_in_sideset(
        mesh: Mesh,
        dof_map: GlobalDofMap,
        field_name: str,
        sideset: str,
        points: NDArray[np.floating],
) -> tuple[NDArray[np.intp], NDArray[np.floating]]:
    """Locate each point in the sideset's facets.

    Returns ``(facet_nodes, weights)``: ``facet_nodes``
    ``(num_points, nodes_per_facet)`` holds the global node ids of the
    facet each point fell in, and ``weights`` the matching interpolation
    weights (barycentric for triangles, bilinear Q1 for quadrilaterals).
    Each point is assigned to the facet it lies most cleanly inside, found
    among its nearest facets by centroid.
    """
    facet_nodes = _sideset_facet_nodes(mesh, dof_map, field_name, sideset)
    nodes_per_facet = facet_nodes.shape[1]
    facet_weights: Callable[
        [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
    ]
    if nodes_per_facet == 3:
        facet_weights = _barycentric
    elif nodes_per_facet == 4:
        facet_weights = _inverse_bilinear
    else:
        raise NotImplementedError(
            f"point location supports triangular and quadrilateral facets "
            f"only; sideset '{sideset}' has {nodes_per_facet} nodes per facet"
        )
    facet_coords = mesh.nodes[facet_nodes]
    centroids = facet_coords.mean(axis=1)
    tree = cKDTree(centroids)
    # A point inside a facet is no farther from that facet's centroid than
    # the centroid-to-vertex distance, so a ball of the largest such radius
    # over the sideset is guaranteed to include the containing facet, however
    # facet sizes vary. (Exact for on-surface points; real DIC off-surface
    # scatter would widen this by its tolerance.)
    search_radius = float(
        np.linalg.norm(facet_coords - centroids[:, None, :], axis=2).max()
    )

    points = np.asarray(points, dtype=np.float64)
    out_nodes = np.zeros((points.shape[0], nodes_per_facet), dtype=np.intp)
    out_weights = np.zeros((points.shape[0], nodes_per_facet), dtype=np.float64)
    for i, point in enumerate(points):
        candidates = tree.query_ball_point(point, search_radius)
        if not candidates:
            candidates = [int(tree.query(point)[1])]
        best_facet, best_weights, best_outside = -1, out_weights[0], np.inf
        for facet in candidates:
            weights = facet_weights(point, facet_coords[facet])
            outside = max(0.0, -float(weights.min()))
            if outside < best_outside:
                best_facet, best_weights, best_outside = (
                    facet, weights, outside,
                )
            if outside == 0.0:
                break
        out_nodes[i] = facet_nodes[best_facet]
        out_weights[i] = best_weights
    return out_nodes, out_weights


def sample_fe_displacement_cloud(
        fe_problem: FEProblem,
        fe_state: FEState,
        points: NDArray[np.floating],
        sideset: str,
        field_name: str = "u",
) -> PointCloud:
    """Sample the solved displacement at ``points`` on ``sideset`` per step.

    Locates each point in the sideset's facets, interpolates the solved
    displacement field there at every step, and returns a
    :class:`~cmad.io.point_cloud.PointCloud` with a ``"displacement"``
    field. The geometry of a real DIC measurement, filled with the values
    an FE solve predicts.
    """
    mesh = fe_problem.mesh
    dof_map = fe_problem.dof_map
    name_to_idx = {fl.name: i for i, fl in enumerate(dof_map.field_layouts)}
    field_idx = name_to_idx[field_name]
    block_offset = int(dof_map.block_offsets[field_idx])
    num_components = int(dof_map.num_dofs_per_basis_fn[field_idx])

    facet_nodes, weights = locate_points_in_sideset(
        mesh, dof_map, field_name, sideset, points,
    )
    comps = np.arange(num_components, dtype=np.intp)
    eq = block_offset + facet_nodes[:, :, None] * num_components + comps

    times = np.asarray(fe_state.t_history, dtype=np.float64)
    frames = []
    for step in range(times.shape[0]):
        u_step = np.asarray(fe_state.U_at(step), dtype=np.float64)
        gathered = u_step[eq]
        frames.append(np.einsum("pn,pnc->pc", weights, gathered))

    return PointCloud(
        coords=np.asarray(points, dtype=np.float64),
        times=times,
        fields={"displacement": np.stack(frames, axis=0)},
    )
