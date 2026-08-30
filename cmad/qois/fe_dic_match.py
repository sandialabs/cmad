"""DIC surface displacement-match QoI.

A measured point cloud of surface displacements is GMLS-remapped onto a
sideset's displacement coefficients, and the squared mismatch is
integrated over that sideset.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.dof import dof_physical_coords
from cmad.io.point_cloud import PointCloud, read_point_cloud
from cmad.qois.fe_qoi import FEQoI, MatchTimes, StepContribution
from cmad.qois.surface_match import (
    surface_groups_and_norm,
    surface_l2_step_closure,
)
from cmad.remap.gmls import build_gmls_operators
from cmad.typing import Params

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem
    from cmad.fem.kernel_arrays import FEKernelArrays


def _fit_plane(
        anchors: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Centroid and two in-plane axes of a set of coplanar points."""
    centroid = anchors.mean(axis=0)
    _, _, vt = np.linalg.svd(anchors - centroid, full_matrices=False)
    return centroid, vt[:2]


def _plane_coords(
        points: NDArray[np.floating],
        centroid: NDArray[np.floating],
        axes: NDArray[np.floating],
) -> NDArray[np.floating]:
    """In-plane 2D coordinates of ``points`` for the plane ``(centroid, axes)``."""
    return (points - centroid) @ axes.T


class FEDicMatch(FEQoI):
    r"""Squared displacement mismatch against DIC data on a sideset.

    The DIC point cloud is GMLS-remapped onto the sideset's displacement
    coefficients to give a per-step nodal surface target, and the
    objective is the time- and area-averaged
    :math:`|u - u^\mathrm{DIC}|^2` integrated over the sideset.

    The cloud points and the sideset anchors are projected onto the
    measurement plane (the anchors are coplanar on a flat face); the GMLS
    fit and the remap are in those 2D coordinates.
    """

    problem_type: ClassVar[str] = "fe"

    def __init__(
            self,
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
            cloud: PointCloud,
            sideset: str,
            *,
            field: str = "displacement",
            poly_order: int = 2,
            support_multiplier: float = 1.6,
            weight: float = 1.0,
    ) -> None:
        num_steps = len(t_schedule)
        if cloud.num_steps != num_steps:
            raise ValueError(
                f"FEDicMatch: cloud has {cloud.num_steps} frames but the "
                f"time schedule has {num_steps}; one DIC frame per step is "
                f"required"
            )
        if field not in cloud.fields:
            raise ValueError(
                f"FEDicMatch: cloud has no field '{field}' "
                f"(has: {sorted(cloud.fields)})"
            )

        anchors, eq = dof_physical_coords(
            fe_problem.mesh, fe_problem.dof_map, "u", sideset,
        )
        num_components = eq.shape[1]
        if cloud.dim != anchors.shape[1]:
            raise ValueError(
                f"FEDicMatch: cloud is {cloud.dim}D but the mesh is "
                f"{anchors.shape[1]}D"
            )
        disp = cloud.fields[field]
        if disp.shape[2] != num_components:
            raise ValueError(
                f"FEDicMatch: DIC field '{field}' has {disp.shape[2]} "
                f"components but field 'u' has {num_components}"
            )

        centroid, axes = _fit_plane(anchors)
        anchor_2d = _plane_coords(anchors, centroid, axes)
        source_2d = _plane_coords(cloud.coords, centroid, axes)
        ops = build_gmls_operators(
            source_2d, anchor_2d, poly_order=poly_order,
            support_multiplier=support_multiplier,
        )

        num_total_dofs = fe_problem.dof_map.num_total_dofs
        data_flat = np.zeros((num_steps, num_total_dofs))
        for step in range(num_steps):
            for c in range(num_components):
                data_flat[step, eq[:, c]] = ops.value @ disp[step, :, c]

        self._match_times = MatchTimes.from_times(t_schedule)
        self._groups, self._norm_factor = surface_groups_and_norm(
            fe_problem, sideset, "u", weight, self._match_times,
        )
        self._data_flat = jnp.asarray(data_flat, dtype=jnp.float64)

    @classmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
    ) -> FEDicMatch:
        cloud = read_point_cloud(qoi_section["dic_file"])
        return cls(
            fe_problem, t_schedule, cloud, qoi_section["sideset"],
            field=qoi_section.get("field", "displacement"),
            poly_order=int(qoi_section.get("poly_order", 2)),
            support_multiplier=float(
                qoi_section.get("support_multiplier", 1.6)
            ),
            weight=float(qoi_section.get("weight", 1.0)),
        )

    def step_contribution(
            self,
            params_by_block: Mapping[str, Params],
            fe_arrays: FEKernelArrays,
    ) -> StepContribution:
        del params_by_block, fe_arrays  # params enter through the solved U
        return surface_l2_step_closure(
            self._groups, self._data_flat, self._match_times,
            self._norm_factor,
        )
