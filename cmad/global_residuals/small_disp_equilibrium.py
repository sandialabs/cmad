"""3D u-only quasi-static small-deformation equilibrium global residual."""
from typing import Any

import jax.numpy as jnp

from cmad.global_residuals.global_residual import GlobalResidual
from cmad.models.var_types import VarType
from cmad.parameters.parameters import Parameters


class SmallDispEquilibrium(GlobalResidual):
    """3D u-only quasi-static small-deformation equilibrium.

    Per-IP residual: ``R_nodal[a, i] = grad_N_phys[a, j] *
    sigma[j, i] * w * dv``, with sigma from
    ``model.cauchy_closed_form(params, U_ip, U_ip_prev)``. The body-
    force contribution ``f_ext = N · b · w · dv`` is applied by the
    assembly layer (not inside residual_fn) so this GR stays internal
    force only.

    Single residual block: ``resid_names[0] = "displacement"``,
    ``var_names[0] = "u"``. Constructor takes the per-element basis-
    function count (8 for linear hex, 4 for linear tet) so a single
    GR instance pairs with one element family.

    Pairs via ``for_model(model, mode=CLOSED_FORM)`` with a Model
    exposing ``cauchy_closed_form`` (e.g. ``Elastic`` with
    ``def_type=FULL_3D``); the per-IP local Newton is bypassed in
    that mode. COUPLED-mode pairing — needed when plasticity threads
    through the IFT chain — requires either a sibling subclass
    calling ``model.cauchy(xi, xi_prev, params, U_ip, U_ip_prev)`` or
    an internal mode dispatch; not supported here.
    """

    def __init__(self, num_basis_fns: int, ndims: int = 3) -> None:
        self._is_complex = False
        self.dtype = float
        self._ndims = ndims

        self._init_residuals(1)
        self._var_types[0] = VarType.VECTOR
        self._num_eqs[0] = ndims
        self._num_basis_fns[0] = num_basis_fns
        self.resid_names[0] = "displacement"
        self.var_names[0] = "u"
        self._init_element_dof_layout()

        def residual_fn(xi, xi_prev, params, U, U_prev,
                        model, shapes_ip, w, dv, ip_set):
            U_ip = self.interpolate_global_fields_at_ip(U, shapes_ip)
            U_ip_prev = self.interpolate_global_fields_at_ip(U_prev, shapes_ip)
            sigma = model.cauchy_closed_form(params, U_ip, U_ip_prev)
            R_internal = (shapes_ip[0].grad_N @ sigma) * w * dv
            return [R_internal]

        super().__init__(residual_fn)

    @classmethod
    def from_deck(
            cls,
            gr_section: dict[str, Any],
            parameters: Parameters,
    ) -> "SmallDispEquilibrium":
        raise NotImplementedError(
            "SmallDispEquilibrium.from_deck is wired by the FE "
            "schema/registry layer when it lands"
        )
