"""Time- and space-averaged squared-displacement QoI for FE problems."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from cmad.qois.cell_match import cell_arrays_and_measure, cell_squared_mismatch
from cmad.qois.fe_qoi import FEQoI, StepContribution
from cmad.typing import JaxArray, Params

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem
    from cmad.fem.kernel_arrays import FEKernelArrays
    from cmad.models.global_fields import StepTime


class FEDisplacementL2(FEQoI):
    r"""Time- and space-averaged squared displacement.

    .. math::

       J = \frac{1}{T \, |\Omega|}
            \sum_n \Delta t_n \int_\Omega |u_n|^2 \, dV

    with :math:`T = t_N - t_0`, :math:`|\Omega|` the total domain
    volume, and :math:`u_0 = 0`. Operates on the residual block whose
    ``var_name`` is ``"u"`` (the displacement field); mixed-field
    problems (e.g. u-p) are supported because the closure indexes
    that block specifically and ignores others. The integral uses a rule
    exact for the square of a linear field, whatever rule the assembly
    uses (:mod:`cmad.qois.cell_match`).
    """

    problem_type: ClassVar[str] = "fe"

    def __init__(
            self,
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
    ) -> None:
        super().__init__()
        var_names = list(fe_problem.gr.var_names)
        try:
            r_disp = var_names.index("u")
        except ValueError as exc:
            raise ValueError(
                f"FEDisplacementL2 requires a residual block with "
                f"var_name 'u'; got var_names={var_names}"
            ) from exc

        T = float(t_schedule[-1]) - float(t_schedule[0])

        # r_disp indexes per-residual-block arrays (gr.var_names);
        # field_idx_disp indexes per-field-layout arrays (the element
        # gather). field_idx_per_block[r] bridges them; the two coincide
        # when the displacement is the problem's only field.
        self._field_idx_disp = fe_problem.field_idx_per_block[r_disp]
        self._cell_arrays, total_volume = cell_arrays_and_measure(
            fe_problem, "u", None,
        )
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
        del params_by_block  # params enter only through the solved state U
        norm_factor = self._norm_factor
        squared_field = cell_squared_mismatch(
            self._cell_arrays, self._field_idx_disp, fe_arrays,
        )

        def _closure(
                U: JaxArray,
                U_prev: JaxArray,
                xi: Mapping[str, JaxArray],
                xi_prev: Mapping[str, JaxArray],
                step_time: StepTime,
        ) -> JaxArray:
            del U_prev, xi, xi_prev
            return norm_factor * step_time.dt * squared_field(U, 0)

        return _closure
