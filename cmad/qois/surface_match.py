"""Squared displacement mismatch integrated over a surface.

Shared by the QoIs that compare an FE field to data on a measured
surface: :func:`surface_groups_and_area` builds the per-facet surface
cache and gives the area it covers, and :func:`surface_squared_mismatch`
gathers the field at each facet, interpolates the mismatch to the side
quadrature points, and integrates its square over the surface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.surface_integration import (
    SurfaceIntegrationGroup,
    build_surface_integration_groups,
)
from cmad.qois.fe_match_term import SquaredMismatch
from cmad.typing import JaxArray

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem


def surface_groups_and_area(
        fe_problem: FEProblem,
        sides: str | NDArray[np.intp],
        field_name: str,
) -> tuple[list[SurfaceIntegrationGroup], float]:
    """Surface cache for ``field_name`` on ``sides`` and the area of those
    sides alone, the surface analogue of the volume a mismatch off a
    sideset is averaged over.

    ``sides`` names a sideset or gives its ``(elem_id, local_side_id)``
    pairs.
    """
    groups = build_surface_integration_groups(
        fe_problem.mesh, fe_problem.dof_map, field_name, sides,
        fe_problem.side_quadrature,
    )
    area = sum(float(jnp.sum(g.dA * g.side_w[None, :])) for g in groups)
    return groups, area


def surface_squared_mismatch(
        groups: list[SurfaceIntegrationGroup],
        data_flat: JaxArray,
) -> SquaredMismatch:
    """``mismatch(U, step)``: the squared difference between ``U`` and the
    data at match time ``step``, integrated over the sideset.
    ``data_flat`` holds one row per match time,
    ``(num_match_times, num_total_dofs)``."""
    def _mismatch(U: JaxArray, step: int | JaxArray) -> JaxArray:
        U_data = data_flat[step]
        total = jnp.zeros(())
        for g in groups:
            diff = U[g.eq] - U_data[g.eq]
            diff_at_ip = jnp.einsum("pa,eac->epc", g.N_side, diff)
            diff_sq = jnp.sum(diff_at_ip * diff_at_ip, axis=-1)
            total = total + jnp.sum(diff_sq * g.dA * g.side_w[None, :])
        return total

    return _mismatch
