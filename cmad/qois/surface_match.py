"""Squared displacement mismatch integrated over a sideset surface.

Shared by the QoIs that compare an FE field to data on a measured
surface: :func:`surface_groups_and_norm` builds the per-facet surface
cache and the ``1 / (time span * area)`` normalization, and
:func:`surface_l2_step_closure` is the per-step closure that gathers the
field at each facet, interpolates the mismatch to the side quadrature
points, and integrates its square over the sideset.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp

from cmad.fem.surface_integration import (
    SurfaceIntegrationGroup,
    build_surface_integration_groups,
)
from cmad.qois.fe_qoi import StepContribution
from cmad.typing import JaxArray

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem


def surface_groups_and_norm(
        fe_problem: FEProblem,
        sideset: str,
        field_name: str,
        weight: float,
        t_schedule: Sequence[float],
) -> tuple[list[SurfaceIntegrationGroup], float]:
    """Surface cache for ``field_name`` on ``sideset`` and its normalization.

    The normalization is ``weight / (time span * surface area)``, the
    surface analogue of the volume averaging used off a sideset.
    """
    groups = build_surface_integration_groups(
        fe_problem.mesh, fe_problem.dof_map, field_name, sideset,
        fe_problem.side_quadrature,
    )
    area = sum(float(jnp.sum(g.dA * g.side_w[None, :])) for g in groups)
    span = float(t_schedule[-1]) - float(t_schedule[0])
    return groups, float(weight) / (span * area)


def surface_l2_step_closure(
        groups: list[SurfaceIntegrationGroup],
        data_flat: JaxArray,
        t_schedule: JaxArray,
        norm_factor: float,
) -> StepContribution:
    """Per-step closure for the squared mismatch over the sideset.

    ``data_flat`` is ``(num_steps, num_total_dofs)``; the step is the one
    in ``t_schedule`` nearest ``t``. The field is gathered at each facet's
    equation numbers, the mismatch is interpolated to the side quadrature
    points, and its square is integrated over the sideset.
    """
    def _closure(
            U: JaxArray,
            U_prev: JaxArray,
            xi: Mapping[str, JaxArray],
            xi_prev: Mapping[str, JaxArray],
            t: JaxArray,
            t_prev: JaxArray,
    ) -> JaxArray:
        del U_prev, xi, xi_prev
        dt = t - t_prev
        step = jnp.argmin(jnp.abs(t_schedule - t))
        U_data = data_flat[step]
        total = jnp.zeros(())
        for g in groups:
            diff = U[g.eq] - U_data[g.eq]
            diff_at_ip = jnp.einsum("pa,eac->epc", g.N_side, diff)
            diff_sq = jnp.sum(diff_at_ip * diff_at_ip, axis=-1)
            total = total + jnp.sum(diff_sq * g.dA * g.side_w[None, :])
        return norm_factor * dt * total

    return _closure
