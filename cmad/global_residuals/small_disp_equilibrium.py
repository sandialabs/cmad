"""3D quasi-static small-deformation equilibrium global residual."""
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.mesh import Mesh
from cmad.global_residuals.global_residual import GlobalResidual
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.io.registry import register_global_residual
from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.model import Model
from cmad.models.var_types import VarType
from cmad.typing import GREvaluators


@register_global_residual("small_disp_equilibrium")
class SmallDispEquilibrium(GlobalResidual):
    """3D quasi-static small-deformation equilibrium.

    Two formulations, selected at construction:

    - **displacement** (default): one residual block, ``u``. Per-IP
      residual ``R_nodal[a, i] = grad_N_phys[a, j] * sigma[j, i] * w *
      dv``, with sigma sourced per the ``mode`` arg from
      ``model.cauchy_closed_form(params, U_ip, U_ip_prev)`` (CLOSED_FORM)
      or ``model.cauchy(xi, xi_prev, params, U_ip, U_ip_prev)`` (COUPLED).

    - **mixed** (displacement-pressure, stabilized equal order): two
      blocks, ``u`` (VECTOR, "equilibrium") and ``p`` (SCALAR,
      "pressure"). The momentum stress is ``sigma = dev - p*I`` with
      ``dev = model.dev_cauchy_closed_form(...)`` -- the deviatoric part
      from the displacement, the hydrostatic part from the pressure dof.
      The pressure block weakly ties ``p`` to
      ``-model.hydro_cauchy_closed_form(...)`` and adds a pressure-
      gradient term ``tau * grad(p).grad(q)`` that supplies the inf-sup
      stability equal-order pairs lack, with ``tau = mult * 0.5 * h^2 /
      mu`` (``mu = model.shear_scale_factor``, ``h`` the element size).
      Mixed needs a model with ``supports_mixed`` and is bound
      CLOSED_FORM (the COUPLED plastic case is a later stage). It is
      restricted to ``ndims == 3`` for now.

    The body-force contribution ``f_ext = N · b · w · dv`` is applied by
    the assembly layer (not inside residual_fn) so this GR stays
    internal-force only. Per-element basis-fn counts come from each
    paired field's ``FiniteElement.num_dofs_per_element`` at FEProblem
    assembly time, so the GR is element-family-agnostic.
    """

    def __init__(
            self, ndims: int = 3, mixed: bool = False,
            stabilization_multiplier: float = 1.0,
    ) -> None:
        self._is_complex = False
        self.dtype = float
        self._ndims = ndims
        self._mixed = mixed
        self._stabilization_multiplier = stabilization_multiplier

        if mixed and ndims != 3:
            raise NotImplementedError(
                f"mixed formulation currently supports ndims=3 only; "
                f"got ndims={ndims}",
            )

        if mixed:
            self._init_residuals(2)
            self._var_types[1] = VarType.SCALAR
            self._num_eqs[1] = 1
            self.resid_names[1] = "pressure"
            self.var_names[1] = "p"
        else:
            self._init_residuals(1)
        self._var_types[0] = VarType.VECTOR
        self._num_eqs[0] = ndims
        self.resid_names[0] = "equilibrium"
        self.var_names[0] = "u"

        def residual_fn(xi, xi_prev, params, U, U_prev,
                        model, mode, shapes_ip, w, dv, h, ip_set):
            U_ip = self.interpolate_global_fields_at_ip(U, shapes_ip)
            U_ip_prev = self.interpolate_global_fields_at_ip(U_prev, shapes_ip)

            if self._mixed:
                dev = model.dev_cauchy_closed_form(params, U_ip, U_ip_prev)
                p = U_ip.fields["p"][0]
                sigma = dev - p * jnp.eye(self._ndims)
                R_u = (shapes_ip[0].grad_N @ sigma) * w * dv

                hydro = model.hydro_cauchy_closed_form(params, U_ip, U_ip_prev)
                psf = model.pressure_scale_factor(params)
                mu = model.shear_scale_factor(params)
                tau = self._stabilization_multiplier * 0.5 * h ** 2 / mu
                N_p = shapes_ip[1].N
                grad_p = U_ip.grad_fields["p"][0]
                R_p = (
                    -(p + hydro) / psf * N_p
                    - tau * (shapes_ip[1].grad_N @ grad_p)
                ) * w * dv
                return [R_u, R_p[:, None]]

            if mode == GlobalResidualMode.CLOSED_FORM:
                sigma = model.cauchy_closed_form(params, U_ip, U_ip_prev)
            else:
                sigma = model.cauchy(xi, xi_prev, params, U_ip, U_ip_prev)
            R_internal = (shapes_ip[0].grad_N @ sigma) * w * dv
            return [R_internal]

        super().__init__(residual_fn)

    def for_model(
            self,
            model: Model,
            mode: GlobalResidualMode = GlobalResidualMode.COUPLED,
            local_newton_settings: dict[str, Any] | None = None,
            print_local_convergence: bool = False,
    ) -> GREvaluators:
        """Bind to a model, rejecting a mixed binding to a model that does
        not support the mixed formulation (``supports_mixed``), as the
        base rejects a CLOSED_FORM binding to one lacking
        ``supports_closed_form_cauchy``.
        """
        if self._mixed and not model.supports_mixed:
            raise ValueError(
                f"mixed formulation requires a model with supports_mixed; "
                f"got {type(model).__name__} with the flag False",
            )
        return super().for_model(
            model, mode, local_newton_settings, print_local_convergence,
        )

    def near_null_space(self, mesh: Mesh) -> NDArray[np.floating]:
        """Near-null-space modes at the mesh nodes.

        Displacement formulation: the 6 rigid-body modes of 3D
        elasticity (3 translations + 3 rotations ``e_k × r`` per node)
        via :func:`pyamg.util.utils.coord_to_rbm`, in interleaved-by-node
        DOF order (matching :meth:`cmad.fem.dof.GlobalDofMap.eq_index`'s
        ``basis_fn * ndofs + dof`` layout). Shape ``(3 * n_nodes, 6)``.

        Mixed formulation: a block-diagonal near-null-space over the
        block-major ``(u, p)`` dofs -- the 6 rigid-body modes on the
        ``u`` block (zero on ``p``) plus the constant mode on the ``p``
        block (zero on ``u``), since the pressure-gradient term
        annihilates a constant pressure. Shape ``(4 * n_nodes, 7)``.
        """
        from pyamg.util.utils import coord_to_rbm
        coords = np.asarray(mesh.nodes, dtype=np.float64)
        n = coords.shape[0]
        u_modes = coord_to_rbm(
            n, 3, coords[:, 0], coords[:, 1], coords[:, 2],
        )
        if not self._mixed:
            return u_modes
        n_u = u_modes.shape[0]
        modes = np.zeros((n_u + n, 7))
        modes[:n_u, :6] = u_modes
        modes[n_u:, 6] = 1.0
        return modes

    def evaluate_nodal_field(
            self,
            name: str,
            fe_problem: FEProblem,
            fe_state: FEState,
            step: int,
    ) -> NDArray[np.floating]:
        if name == "u":
            U = np.asarray(fe_state.U_at(step))
            if self._mixed:
                u_end = fe_problem.dof_map.block_offsets[1]
                return U[:u_end].reshape(-1, int(self._num_eqs[0]))
            return U.reshape(-1, int(self._num_eqs[0]))
        if name == "p" and self._mixed:
            U = np.asarray(fe_state.U_at(step))
            p_start = fe_problem.dof_map.block_offsets[1]
            return U[p_start:].reshape(-1, 1)
        return super().evaluate_nodal_field(
            name, fe_problem, fe_state, step,
        )

    @classmethod
    def from_deck(
            cls,
            gr_section: dict[str, Any],
            ndims: int,
    ) -> "SmallDispEquilibrium":
        """Construct from the resolved ``residuals.global residual`` section.

        Requires ``def_type`` and cross-checks the spatial dimension it
        implies against ``ndims``, which the deck builder sources from the
        mesh. A def_type whose dimension disagrees with the mesh raises.
        ``formulation`` selects ``displacement`` (default) or ``mixed``
        u-p; ``stabilization multiplier`` scales the mixed pressure
        stabilization (default 1.0).
        """
        def_type_name = gr_section.get("def_type")
        if def_type_name is None:
            raise ValueError(
                "residuals.global residual: small_disp_equilibrium "
                "requires 'def_type'",
            )
        def_type = DefType[def_type_name.upper()]
        expected_ndims = def_type_ndims(def_type)
        if expected_ndims != ndims:
            raise ValueError(
                f"residuals.global residual: def_type '{def_type_name}' "
                f"implies ndims={expected_ndims} but the mesh has "
                f"ndims={ndims}",
            )
        formulation = gr_section.get("formulation", "displacement")
        if formulation not in ("displacement", "mixed"):
            raise ValueError(
                "residuals.global residual: formulation must be "
                f"'displacement' or 'mixed'; got '{formulation}'",
            )
        return cls(
            ndims=ndims,
            mixed=(formulation == "mixed"),
            stabilization_multiplier=gr_section.get(
                "stabilization multiplier", 1.0),
        )
