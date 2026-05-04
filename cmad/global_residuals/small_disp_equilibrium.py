"""3D u-only quasi-static small-deformation equilibrium global residual."""
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.postprocess import evaluate_cauchy_at_ips
from cmad.global_residuals.global_residual import GlobalResidual
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.io.registry import register_global_residual
from cmad.io.results import FieldSpec, ip_average_to_element
from cmad.models.var_types import VarType


@register_global_residual("small_disp_equilibrium")
class SmallDispEquilibrium(GlobalResidual):
    """3D u-only quasi-static small-deformation equilibrium.

    Per-IP residual: ``R_nodal[a, i] = grad_N_phys[a, j] *
    sigma[j, i] * w * dv``, with sigma sourced per the ``mode``
    arg from ``model.cauchy_closed_form(params, U_ip, U_ip_prev)``
    (CLOSED_FORM) or ``model.cauchy(xi, xi_prev, params, U_ip,
    U_ip_prev)`` (COUPLED). The body-force contribution
    ``f_ext = N · b · w · dv`` is applied by the assembly layer
    (not inside residual_fn) so this GR stays internal-force
    only.

    Single residual block: ``resid_names[0] = "displacement"``,
    ``var_names[0] = "u"``. Per-element basis-fn count comes from the
    paired field's ``FiniteElement.num_dofs_per_element`` at FEProblem
    assembly time, so this GR is element-family-agnostic.

    Pairs via ``for_model(model, mode)`` in either CLOSED_FORM or
    COUPLED. CLOSED_FORM requires a Model exposing
    ``cauchy_closed_form`` (e.g. ``Elastic`` with
    ``def_type=FULL_3D``) and bypasses the per-IP local Newton.
    COUPLED runs the local Newton inside ``for_model``'s closures
    (``R_and_dR_dU_and_xi`` and ``dR_dU``) for path-dependent
    plasticity, where ``model.cauchy(xi, xi_prev, ...)`` reads the
    converged ξ.
    """

    def __init__(self, ndims: int = 3) -> None:
        self._is_complex = False
        self.dtype = float
        self._ndims = ndims

        self._init_residuals(1)
        self._var_types[0] = VarType.VECTOR
        self._num_eqs[0] = ndims
        self.resid_names[0] = "displacement"
        self.var_names[0] = "u"

        def residual_fn(xi, xi_prev, params, U, U_prev,
                        model, mode, shapes_ip, w, dv, ip_set):
            U_ip = self.interpolate_global_fields_at_ip(U, shapes_ip)
            U_ip_prev = self.interpolate_global_fields_at_ip(U_prev, shapes_ip)
            if mode == GlobalResidualMode.CLOSED_FORM:
                sigma = model.cauchy_closed_form(params, U_ip, U_ip_prev)
            else:
                sigma = model.cauchy(xi, xi_prev, params, U_ip, U_ip_prev)
            R_internal = (shapes_ip[0].grad_N @ sigma) * w * dv
            return [R_internal]

        super().__init__(residual_fn)

    def default_output_fields(self) -> dict[str, list[FieldSpec]]:
        return {
            "nodal": [FieldSpec("displacement", VarType.VECTOR)],
            "element": [FieldSpec("cauchy", VarType.SYM_TENSOR)],
        }

    def evaluate_nodal_field(
            self,
            name: str,
            fe_problem: FEProblem,
            fe_state: FEState,
            step: int,
    ) -> NDArray[np.floating]:
        if name == "displacement":
            U = np.asarray(fe_state.U_at(step))
            return U.reshape(-1, int(self._num_eqs[0]))
        return super().evaluate_nodal_field(
            name, fe_problem, fe_state, step,
        )

    def evaluate_element_field(
            self,
            name: str,
            fe_problem: FEProblem,
            fe_state: FEState,
            step: int,
            block: str,
    ) -> NDArray[np.floating]:
        if name == "cauchy":
            ip_data = evaluate_cauchy_at_ips(
                fe_problem, fe_state, step, block,
            )
            return ip_average_to_element(
                ip_data, fe_problem.geometry_cache, block,
            )
        return super().evaluate_element_field(
            name, fe_problem, fe_state, step, block,
        )

    @classmethod
    def from_deck(
            cls,
            gr_section: dict[str, Any],
            ndims: int,
    ) -> "SmallDispEquilibrium":
        """Construct from the resolved ``residuals.global residual`` section.

        ``gr_section`` carries the GR-side fields (``type``, nonlinear-
        solver kwargs); ``ndims`` is sourced from the mesh by the deck-
        side builder. This GR has no GR-specific config beyond ``ndims``,
        so the section dict is unread; future GRs that need extra deck
        keys will read them here.
        """
        return cls(ndims=ndims)
