"""Time- and space-averaged squared-displacement QoI for FE problems."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp

from cmad.fem.assembly import _gather_element_U
from cmad.fem.precompute import compute_ip_quadrature_weights
from cmad.io.registry import register_qoi
from cmad.qois.fe_qoi import FEQoI, StepContribution
from cmad.typing import JaxArray, Params

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem
    from cmad.fem.kernel_arrays import FEKernelArrays


@register_qoi("fe_displacement_l2")
class FEDisplacementL2(FEQoI):
    r"""Time- and space-averaged squared displacement.

    .. math::

       J = \frac{1}{T \, |\Omega|}
            \sum_n \Delta t_n \int_\Omega |u_n|^2 \, dV

    with :math:`T = t_N - t_0`, :math:`|\Omega|` the total domain
    volume (``Σ_blocks Σ_elem Σ_ip iso_jac_det · w``), and
    :math:`u_0 = 0`. Operates on the residual block whose
    ``var_name`` is ``"u"`` (the displacement field); mixed-field
    problems (e.g. u-p) are supported because the closure indexes
    that block specifically and ignores others.
    """

    problem_type: ClassVar[str] = "fe"

    def __init__(
            self,
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
    ) -> None:
        var_names = list(fe_problem.gr.var_names)
        try:
            r_disp = var_names.index("u")
        except ValueError as exc:
            raise ValueError(
                f"FEDisplacementL2 requires a residual block with "
                f"var_name 'u'; got var_names={var_names}"
            ) from exc

        # Total volume: sum of iso_jac_det · w over all blocks/elems/IPs.
        ip_weights = compute_ip_quadrature_weights(fe_problem.geometry_cache)
        total_volume = float(sum(arr.sum() for arr in ip_weights.values()))

        T = float(t_schedule[-1]) - float(t_schedule[0])

        self._fe_problem = fe_problem
        # r_disp indexes per-residual-block arrays (gr.var_names,
        # geometry_cache.shared.field_N_per_block); field_idx_disp
        # indexes per-field-layout arrays (_gather_element_U output).
        # field_idx_per_block[r] bridges them; the two coincide for
        # single-displacement problems.
        self._r_disp = r_disp
        self._field_idx_disp = fe_problem.field_idx_per_block[r_disp]
        self._norm_factor = 1.0 / (T * total_volume)

    @classmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
    ) -> FEDisplacementL2:
        return cls(fe_problem, t_schedule)

    def step_contribution(
            self,
            params_by_block: Mapping[str, Params],
            fe_arrays: FEKernelArrays,
    ) -> StepContribution:
        del params_by_block  # forward functional: params not consumed
        fe_problem = self._fe_problem
        r_disp = self._r_disp
        field_idx_disp = self._field_idx_disp
        norm_factor = self._norm_factor

        block_data: list[tuple[str, JaxArray, JaxArray]] = []
        for block_name in fe_problem.models_by_block:
            geom_cache = fe_arrays.geometry_cache[block_name]
            N_disp = geom_cache.shared.field_N_per_block[r_disp]
            quad_w = geom_cache.shared.quad_w
            iso_jac_det = geom_cache.per_elem.iso_jac_det
            weighted_iso_jac_det = iso_jac_det * quad_w
            block_data.append(
                (block_name, N_disp, weighted_iso_jac_det),
            )

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
            total_integral = jnp.zeros(())
            for (block_name, N_disp,
                 weighted_iso_jac_det) in block_data:
                U_elem_blocks = _gather_element_U(
                    U, fe_arrays, block_name,
                )
                U_disp_per_elem = U_elem_blocks[field_idx_disp]
                U_at_ip = jnp.einsum(
                    "pa,eak->epk", N_disp, U_disp_per_elem,
                )
                u_sq = jnp.sum(U_at_ip * U_at_ip, axis=-1)
                block_integral = jnp.sum(u_sq * weighted_iso_jac_det)
                total_integral = total_integral + block_integral
            return norm_factor * dt * total_integral

        return _closure
