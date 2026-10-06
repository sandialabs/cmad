"""Thermomechanics global residual: the momentum balance and the energy
balance on one mesh."""
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.mesh import Mesh
from cmad.global_residuals.balance_laws import (
    energy_balance,
    momentum_balance,
    pressure_equation,
)
from cmad.global_residuals.global_residual import GlobalResidual
from cmad.global_residuals.heat_transfer import (
    require_thickness_for_face_flux,
)
from cmad.global_residuals.mechanics import (
    def_type_from_section,
    rigid_body_modes,
)
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.model import Model
from cmad.models.thermomechanics_model import ThermomechanicsModel
from cmad.models.var_types import VarType
from cmad.typing import GREvaluators


class Thermomechanics(GlobalResidual):
    """The quasi-static momentum balance on ``u`` (residual name
    "equilibrium"), with the pressure equation on ``p`` ("pressure") when
    ``mixed``, and the energy balance on ``T`` ("energy balance"), bound to
    a :class:`ThermomechanicsModel`. The bodies are the balance laws of
    :mod:`cmad.global_residuals.balance_laws`; the two formulations are
    those of :class:`cmad.global_residuals.mechanics.Mechanics`.
    ``thickness`` is the out of plane extent of a 2D mesh, which a model
    with a face flux (a plate modeled in its plane) needs.
    """

    def __init__(
            self, ndims: int = 3, mixed: bool = False,
            stabilization_multiplier: float = 1.0,
            thickness: float | None = None,
    ) -> None:
        self._is_complex = False
        self.dtype = float
        self._ndims = ndims
        self._mixed = mixed
        self._stabilization_multiplier = stabilization_multiplier
        self._thickness = thickness

        if mixed and ndims not in (2, 3):
            raise NotImplementedError(
                f"mixed formulation supports ndims 2 (plane strain) or 3; "
                f"got ndims={ndims}",
            )

        self._init_residuals(3 if mixed else 2)
        self._var_types[0] = VarType.VECTOR
        self._num_eqs[0] = ndims
        self.resid_names[0] = "equilibrium"
        self.var_names[0] = "u"
        if mixed:
            self._var_types[1] = VarType.SCALAR
            self._num_eqs[1] = 1
            self.resid_names[1] = "pressure"
            self.var_names[1] = "p"
        T = self.num_residuals - 1
        self._var_types[T] = VarType.SCALAR
        self._num_eqs[T] = 1
        self.resid_names[T] = "energy balance"
        self.var_names[T] = "T"

        def residual_fn(xi, xi_prev, params, U_ip, U_ip_prev,
                        model, mode, shapes_ip, w, dv, h, step_time):
            R = [momentum_balance(
                xi, xi_prev, params, U_ip, U_ip_prev, model, mode,
                shapes_ip[0], w, dv, self._ndims, self._mixed,
            )]
            if self._mixed:
                R.append(pressure_equation(
                    xi, xi_prev, params, U_ip, U_ip_prev, model, mode,
                    shapes_ip[1], w, dv, h, self._ndims,
                    self._stabilization_multiplier,
                ))
            R.append(energy_balance(
                xi, xi_prev, params, U_ip, U_ip_prev, model, mode,
                shapes_ip[T], w, dv, step_time, self._thickness,
            ))
            return R

        super().__init__(residual_fn)

    @property
    def mixed(self) -> bool:
        """True for the displacement-pressure formulation."""
        return self._mixed

    def for_model(
            self,
            model: Model,
            mode: GlobalResidualMode = GlobalResidualMode.COUPLED,
            local_newton_settings: dict[str, Any] | None = None,
            print_local_convergence: bool = False,
            prescribed_field_names: Sequence[str] = (),
    ) -> GREvaluators:
        """Bind to a model, which must be a :class:`ThermomechanicsModel`
        and, when ``mixed``, support the mixed formulation; one with a
        face flux needs the thickness."""
        if not isinstance(model, ThermomechanicsModel):
            raise ValueError(
                f"thermomechanics needs a thermomechanics model, a mechanics "
                f"model with a thermal model; got {type(model).__name__}",
            )
        if self._mixed and not model.supports_mixed:
            raise ValueError(
                f"mixed formulation requires a model with supports_mixed; "
                f"got {type(model.mechanics).__name__} with the flag False",
            )
        require_thickness_for_face_flux(model, self._thickness)
        return super().for_model(
            model, mode, local_newton_settings, print_local_convergence,
            prescribed_field_names,
        )

    def near_null_space(self, mesh: Mesh) -> NDArray[np.floating]:
        """The rigid body modes on the ``u`` rows, zero elsewhere, plus one
        column per scalar field that is one on its rows: the constant
        pressure when ``mixed``, and the constant temperature."""
        u_modes = rigid_body_modes(mesh, self._ndims)
        n_u, n_rbm = u_modes.shape
        n = mesh.nodes.shape[0]
        n_scalar = self.num_residuals - 1
        modes = np.zeros((n_u + n_scalar * n, n_rbm + n_scalar))
        modes[:n_u, :n_rbm] = u_modes
        for k in range(n_scalar):
            modes[n_u + k * n:n_u + (k + 1) * n, n_rbm + k] = 1.0
        return modes

    def evaluate_nodal_field(
            self,
            name: str,
            fe_problem: FEProblem,
            fe_state: FEState,
            step: int,
    ) -> NDArray[np.floating]:
        var_names = [str(v) for v in self.var_names]
        if name in var_names:
            r = var_names.index(name)
            offsets = fe_problem.dof_map.block_offsets
            U = np.asarray(fe_state.U_at(step))
            return U[offsets[r]:offsets[r + 1]].reshape(
                -1, int(self._num_eqs[r]))
        return super().evaluate_nodal_field(
            name, fe_problem, fe_state, step,
        )

    @classmethod
    def from_deck(
            cls,
            gr_section: dict[str, Any],
            ndims: int,
            thickness: float | None = None,
    ) -> "Thermomechanics":
        """Construct from the resolved ``residuals.global residual``
        section: ``def_type`` is required and checked against the mesh,
        ``mixed`` and ``stabilization multiplier`` as for mechanics,
        ``thickness`` from the discretization section."""
        def_type_from_section(gr_section, ndims)
        return cls(
            ndims=ndims,
            mixed=bool(gr_section.get("mixed", False)),
            stabilization_multiplier=gr_section.get(
                "stabilization multiplier", 1.0),
            thickness=thickness,
        )
