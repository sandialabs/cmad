"""Global-residual abstract contract and composed-helper builder."""
from abc import ABC
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from jax import jacfwd, jacrev, jit
from numpy.typing import NDArray

from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.global_residuals.interpolation import interpolate_global_fields_at_ip
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray, PyTree, ResidualFnGR


class GlobalResidual(ABC):
    """Finite-element residual contract at an integration point.

    Subclasses set up a residual function (pure, no side effects) and
    pass it to super().__init__(). See :data:`cmad.typing.ResidualFnGR`
    for the required callable signature:

      (xi, xi_prev, params, U, U_prev, model, shapes_ip, w, dv) -> Array

    where xi/xi_prev are Model's per-integration-point local state
    (threaded through GR's call to model.cauchy; GR has no xi of its
    own), U/U_prev are per-residual-block lists of element-local
    basis-coefficient arrays with ``U[i].shape ==
    (num_basis_fns[i], num_eqs[i])``, shapes_ip is a per-block list of
    :class:`ShapeFunctionsAtIP`, model is the bound :class:`Model`, and
    w/dv are the quadrature weight and reference-volume factor.

    Subclasses pair with a concrete Model via
    :meth:`for_model(model, mode)`, which returns a ``dict[str,
    Callable]`` of jit'd evaluators closed over ``model._residual`` and
    the mode-selected cauchy method. The string keys ("R", "dR_dU",
    ...) identify each evaluator and replace Model's mutable
    ``_deriv_mode`` state machine: FE assembly vmaps closures over
    element batches, which works on pure functions but not on methods
    that mutate instance state.
    """

    # ---- attributes the subclass must set before super().__init__() ----
    dtype: type
    _is_complex: bool
    _num_eqs: NDArray[np.intp]
    _num_basis_fns: NDArray[np.intp]
    _var_types: NDArray[np.intp]
    _ndims: int

    # ---- attributes set by self._init_residuals() ----
    num_residuals: int
    resid_names: list[str | None]

    # ---- attributes set by self._init_element_dof_layout() ----
    num_element_dofs: int
    _block_offsets: NDArray[np.intp]

    # ---- attribute set by __init__() ----
    _residual_fn: ResidualFnGR

    @classmethod
    def from_deck(
            cls,
            gr_section: dict[str, Any],
            parameters: Parameters,
    ) -> "GlobalResidual":
        """Build a :class:`GlobalResidual` instance from the deck's
        ``global_residual:`` section. The base stub raises
        ``NotImplementedError`` so concrete subclasses are forced to
        override; the stub exists so the registry's
        ``type[GlobalResidual]`` return is statically callable via
        ``cls.from_deck(...)``.
        """
        raise NotImplementedError

    def __init__(self, residual_fn: ResidualFnGR) -> None:
        self._residual_fn = residual_fn

    def _init_residuals(self, num_residuals: int) -> None:
        self.num_residuals = num_residuals
        self._num_eqs = np.zeros(num_residuals, dtype=int)
        self._num_basis_fns = np.zeros(num_residuals, dtype=int)
        self._var_types = np.zeros(num_residuals, dtype=int)
        self.resid_names = [None] * num_residuals

    def _init_element_dof_layout(self) -> None:
        """Compute per-block flat-DOF offsets and the total element-DOF
        count. Packing convention P1 (basis-outer, equation-inner within
        a block — see :meth:`delta_U_offset`). Requires ``_num_eqs`` and
        ``_num_basis_fns`` populated by the subclass.
        """
        self._block_offsets = np.zeros(self.num_residuals, dtype=int)
        self.num_element_dofs = 0
        for ii in range(self.num_residuals):
            self._block_offsets[ii] = self.num_element_dofs
            self.num_element_dofs += int(
                self._num_basis_fns[ii] * self._num_eqs[ii]
            )

    def delta_U_offset(
            self, res_idx: int, eq_idx: int, dof_idx: int,
    ) -> int:
        """Flat element-DOF offset for (block ``res_idx``, equation
        ``eq_idx``, basis-coefficient ``dof_idx``).

        Packing convention P1: basis-outer, equation-inner within a
        block — standard FE-code convention (Calibr8 lineage). Keeps
        element-stiffness sub-blocks contiguous in the
        ``(num_eqs, num_basis_fns)`` sub-block layout used by assembly.
        """
        return (
            int(self._block_offsets[res_idx])
            + dof_idx * int(self._num_eqs[res_idx])
            + eq_idx
        )

    def var_type(self, residual: int) -> int:
        return int(self._var_types[residual])

    def resid_name(self, residual: int) -> str | None:
        return self.resid_names[residual]

    @property
    def ndims(self) -> int:
        return self._ndims

    def interpolate_global_fields_at_ip(
            self,
            U: Sequence[JaxArray],
            shapes_ip: Sequence[ShapeFunctionsAtIP],
    ) -> GlobalFieldsAtPoint:
        """Thin method wrapper around the module-level
        :func:`interpolate_global_fields_at_ip`, closing over
        ``self.resid_names``. Subclasses with mixed-basis interpolation
        logic that can't be expressed through per-block iteration
        should override this method.
        """
        return interpolate_global_fields_at_ip(
            U, shapes_ip, self.resid_names,
        )

    def for_model(
            self,
            model: Model,
            mode: GlobalResidualMode = GlobalResidualMode.COUPLED,
    ) -> dict[str, Callable[..., PyTree]]:
        """Bind this GR to a concrete Model in a specific operational
        mode. Returns a dict of jit'd evaluators keyed by string names:

        - always: ``R``, ``dR_dU``, ``dR_dU_prev``, ``dR_dparams``.
        - COUPLED only: ``dR_dxi``, ``dR_dxi_prev``, plus the
          C-composed family ``C``, ``dC_dU``, ``dC_dU_prev``,
          ``dC_dxi``, ``dC_dxi_prev``, ``dC_dparams``.

        The same GR can be bound to multiple (model, mode) combinations
        by repeated ``for_model`` calls; ``mode`` is not stored on the
        instance.

        Raises ``ValueError`` if ``mode == CLOSED_FORM`` and
        ``model.supports_closed_form_cauchy`` is False.
        """
        if (mode == GlobalResidualMode.CLOSED_FORM
                and not model.supports_closed_form_cauchy):
            raise ValueError(
                f"CLOSED_FORM mode requires "
                f"model.supports_closed_form_cauchy; got "
                f"{type(model).__name__} with the flag False"
            )

        residual_fn = self._residual_fn
        resid_names = self.resid_names

        # R-composed: close the model into the public residual function,
        # then jit. Post-closure argnums: xi=0, xi_prev=1, params=2,
        # U=3, U_prev=4, shapes_ip=5, w=6, dv=7.
        def r_at_ip(xi, xi_prev, params, U, U_prev, shapes_ip, w, dv):
            return residual_fn(
                xi, xi_prev, params, U, U_prev,
                model, shapes_ip, w, dv,
            )

        evaluators: dict[str, Callable[..., PyTree]] = {
            "R":          jit(r_at_ip),
            "dR_dU":      jit(jacfwd(r_at_ip, argnums=3)),
            "dR_dU_prev": jit(jacfwd(r_at_ip, argnums=4)),
            "dR_dparams": jit(jacrev(r_at_ip, argnums=2)),
        }

        if mode == GlobalResidualMode.COUPLED:
            evaluators["dR_dxi"] = jit(jacfwd(r_at_ip, argnums=0))
            evaluators["dR_dxi_prev"] = jit(jacfwd(r_at_ip, argnums=1))

            # C-composed: Model's local residual threaded through GR's
            # per-block interpolation, fused under jit with the outer
            # closure. Argnums: xi=0, xi_prev=1, params=2, U=3,
            # U_prev=4, shapes_ip=5.
            def c_at_ip(xi, xi_prev, params, U, U_prev, shapes_ip):
                U_ip = interpolate_global_fields_at_ip(
                    U, shapes_ip, resid_names)
                U_ip_prev = interpolate_global_fields_at_ip(
                    U_prev, shapes_ip, resid_names)
                return model._residual(
                    xi, xi_prev, params, U_ip, U_ip_prev,
                )

            evaluators["C"] = jit(c_at_ip)
            evaluators["dC_dU"] = jit(jacfwd(c_at_ip, argnums=3))
            evaluators["dC_dU_prev"] = jit(jacfwd(c_at_ip, argnums=4))
            evaluators["dC_dxi"] = jit(jacfwd(c_at_ip, argnums=0))
            evaluators["dC_dxi_prev"] = jit(jacfwd(c_at_ip, argnums=1))
            evaluators["dC_dparams"] = jit(jacrev(c_at_ip, argnums=2))

        return evaluators
