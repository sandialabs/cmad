"""Heat transfer global residual: the energy balance."""
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cmad.fem.fe_problem import FEProblem, FEState
from cmad.global_residuals.balance_laws import energy_balance
from cmad.global_residuals.global_residual import GlobalResidual
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.model import Model
from cmad.models.thermal_model import ThermalModel
from cmad.models.var_types import VarType
from cmad.typing import GREvaluators


class HeatTransfer(GlobalResidual):
    """The energy balance on one scalar block, the temperature ``T``
    (residual name "energy balance"); the body is
    :func:`cmad.global_residuals.balance_laws.energy_balance`.

    A source comes from the volumetric sources section and a prescribed
    flux from the surface flux bcs section, both keyed by "energy
    balance". A
    model without a heat capacity makes every step steady.
    """

    def __init__(self, ndims: int = 3) -> None:
        self._is_complex = False
        self.dtype = float
        self._ndims = ndims

        self._init_residuals(1)
        self._var_types[0] = VarType.SCALAR
        self._num_eqs[0] = 1
        self.resid_names[0] = "energy balance"
        self.var_names[0] = "T"

        def residual_fn(xi, xi_prev, params, U_ip, U_ip_prev,
                        model, mode, shapes_ip, w, dv, h, step_time):
            return [energy_balance(
                xi, xi_prev, params, U_ip, U_ip_prev, model, mode,
                shapes_ip[0], w, dv, step_time,
            )]

        super().__init__(residual_fn)

    def for_model(
            self,
            model: Model,
            mode: GlobalResidualMode = GlobalResidualMode.COUPLED,
            local_newton_settings: dict[str, Any] | None = None,
            print_local_convergence: bool = False,
    ) -> GREvaluators:
        """Bind to a model, which must be a :class:`ThermalModel`."""
        if not isinstance(model, ThermalModel):
            raise ValueError(
                f"heat transfer needs a thermal model; got "
                f"{type(model).__name__}",
            )
        return super().for_model(
            model, mode, local_newton_settings, print_local_convergence,
        )

    def evaluate_nodal_field(
            self,
            name: str,
            fe_problem: FEProblem,
            fe_state: FEState,
            step: int,
    ) -> NDArray[np.floating]:
        if name == "T":
            return np.asarray(fe_state.U_at(step)).reshape(-1, 1)
        return super().evaluate_nodal_field(
            name, fe_problem, fe_state, step,
        )

    @classmethod
    def from_deck(
            cls,
            gr_section: dict[str, Any],
            ndims: int,
    ) -> "HeatTransfer":
        """Construct from the resolved ``residuals.global residual``
        section; ``ndims`` comes from the mesh and no ``def_type`` is
        needed."""
        return cls(ndims=ndims)
