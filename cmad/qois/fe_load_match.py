"""Boundary-reaction QoI for FE problems: match a measured load, or write
the computed reaction series from a solve."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp
import numpy as np

from cmad.fem.assembly import (
    assemble_global_residual,
    params_by_block_from_models,
)
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.io.qoi_data import load_reaction_data
from cmad.io.registry import register_qoi
from cmad.qois.fe_qoi import FEQoI, StepContribution
from cmad.typing import JaxArray, Params, Scalar

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem, FEState
    from cmad.fem.kernel_arrays import FEKernelArrays


@register_qoi("fe_load_match")
class FELoadMatch(FEQoI):
    r"""Net boundary reaction on a displacement-controlled sideset.

    Two mutually-exclusive modes, set by the deck:

    - **match** (``data_file``): the time-averaged squared mismatch against a
      measured load is the objective,

      .. math::

         J = \frac{w}{T} \sum_n \Delta t_n
              \sum_c \left( R_{c,n} - d_{c,n} \right)^2

    - **write** (``output_file``): ``cmad primal`` writes the computed
      reaction series to a CSV (synthetic data / plotting); no objective.

    ``R_{c,n}`` is the net reaction in component ``c`` of the displacement
    field on the sideset at step ``n`` -- the global residual summed over
    that sideset's Dirichlet-prescribed dofs. The reaction depends on the
    parameters both through the solved state and directly through the
    residual assembly, so the match closure captures ``params_by_block`` and
    passes it to :func:`cmad.fem.assembly.assemble_global_residual`.
    """

    problem_type: ClassVar[str] = "fe"

    def __init__(
            self,
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
            sideset: str,
            components: Sequence[int],
            data: JaxArray | None = None,
            output_file: str | None = None,
            weight: float = 1.0,
    ) -> None:
        comps = [int(c) for c in components]
        n_comp = len(comps)
        num_steps = len(t_schedule)

        self._fe_problem = fe_problem
        self._eq_per_component = [
            jnp.asarray(
                fe_problem.dof_map.dirichlet_eqs_for_component(
                    sideset, "u", c,
                ),
            )
            for c in comps
        ]
        self._t_schedule = jnp.asarray(t_schedule, dtype=jnp.float64)
        self._norm_factor = float(weight) / (
            float(t_schedule[-1]) - float(t_schedule[0])
        )
        self._output_file = output_file

        self._data: JaxArray | None
        if data is None:
            self._data = None
        else:
            data_arr = jnp.asarray(data, dtype=jnp.float64)
            if data_arr.ndim == 1 and n_comp == 1:
                data_arr = data_arr.reshape(num_steps, 1)
            if data_arr.shape != (num_steps, n_comp):
                raise ValueError(
                    f"FELoadMatch: data has shape {tuple(data_arr.shape)} "
                    f"but expected (num_steps={num_steps}, "
                    f"num_components={n_comp})"
                )
            self._data = data_arr

    @classmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
    ) -> FELoadMatch:
        sideset = qoi_section["sideset"]
        components = qoi_section["components"]
        if "data_file" in qoi_section:
            data = jnp.asarray(
                load_reaction_data(qoi_section), dtype=jnp.float64,
            )
            return cls(
                fe_problem, t_schedule, sideset, components,
                data=data, weight=float(qoi_section.get("weight", 1.0)),
            )
        return cls(
            fe_problem, t_schedule, sideset, components,
            output_file=qoi_section["output_file"],
        )

    def step_contribution(
            self,
            params_by_block: Mapping[str, Params],
            fe_arrays: FEKernelArrays,
    ) -> StepContribution:
        if self._data is None:
            raise ValueError(
                "fe_load_match in write mode (output_file) has no objective; "
                "use it under cmad primal, not objective/calibrate"
            )
        data = self._data
        t_schedule = self._t_schedule
        norm_factor = self._norm_factor

        def _closure(
                U: JaxArray,
                U_prev: JaxArray,
                xi: Mapping[str, JaxArray],
                xi_prev: Mapping[str, JaxArray],
                t: JaxArray,
                t_prev: JaxArray,
        ) -> JaxArray:
            del xi
            dt = t - t_prev
            step = jnp.argmin(jnp.abs(t_schedule - t))
            reaction = self._reaction_at(
                params_by_block, fe_arrays, U, U_prev, t, xi_prev,
            )
            mismatch = jnp.sum((reaction - data[step]) ** 2)
            return norm_factor * dt * mismatch

        return _closure

    def produces_primal_output(self) -> bool:
        return self._output_file is not None

    def write_primal_outputs(
            self, fe_problem: FEProblem, fe_state: FEState,
    ) -> None:
        assert self._output_file is not None
        params = params_by_block_from_models(fe_problem)
        fe_arrays = fe_problem.kernel_arrays
        coupled = [
            b for b, mode in fe_problem.modes_by_block.items()
            if mode == GlobalResidualMode.COUPLED
        ]
        num_steps = len(fe_state.t_history)
        series = np.zeros((num_steps, len(self._eq_per_component)))
        for k in range(num_steps):
            kp = max(k - 1, 0)
            U = jnp.asarray(fe_state.U_at(k))
            U_prev = jnp.asarray(fe_state.U_at(kp))
            xi_prev = {b: jnp.asarray(fe_state.xi_at(kp, b)) for b in coupled}
            reaction = self._reaction_at(
                params, fe_arrays, U, U_prev,
                float(fe_state.t_history[k]), xi_prev,
            )
            series[k] = np.asarray(reaction)
        np.savetxt(self._output_file, series, delimiter=",")

    def _reaction_at(
            self,
            params_by_block: Mapping[str, Params],
            fe_arrays: FEKernelArrays,
            U: JaxArray,
            U_prev: JaxArray,
            t: Scalar,
            xi_prev: Mapping[str, JaxArray],
    ) -> JaxArray:
        """Net reaction per requested component at one state: the residual
        summed over each component's sideset dofs."""
        R = assemble_global_residual(
            self._fe_problem, fe_arrays, params_by_block,
            U, U_prev, t, xi_prev,
        )
        return jnp.stack(
            [jnp.sum(R[eq_c]) for eq_c in self._eq_per_component],
        )
