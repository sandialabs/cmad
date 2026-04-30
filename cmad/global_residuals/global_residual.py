"""Global-residual abstract contract and composed-helper builder."""
from abc import ABC
from collections.abc import Sequence
from typing import Any

import numpy as np
from jax import jacfwd, jit
from numpy.typing import NDArray

from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.global_residuals.interpolation import interpolate_global_fields_at_ip
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.typing import GREvaluators, JaxArray, ResidualFnGR


class GlobalResidual(ABC):
    """Finite-element residual contract at an integration point.

    Subclasses set up a residual function (pure, no side effects) and
    pass it to super().__init__(). See :data:`cmad.typing.ResidualFnGR`
    for the required callable signature:

      (xi, xi_prev, params, U, U_prev, model, shapes_ip, w, dv) -> Sequence[Array]

    where xi/xi_prev are Model's per-integration-point local state
    (threaded through GR's call to model.cauchy; GR has no xi of its
    own), U/U_prev are per-residual-block lists of element-local
    basis-coefficient arrays with ``U[i].shape ==
    (num_basis_fns[i], num_eqs[i])``, shapes_ip is a per-block list of
    :class:`ShapeFunctionsAtIP`, model is the bound :class:`Model`,
    w/dv are the quadrature weight and reference-volume factor, and
    ip_set is the integration-point-set index dispatched by the
    assembly layer (single-ip_set GRs ignore it; multi-ip_set GRs
    use it to dispatch term-specific contributions).

    Subclasses pair with a concrete Model via
    :meth:`for_model(model, mode)`, which returns a ``dict[str,
    Callable]`` of jit'd evaluators closed over ``model._residual``
    and the mode-selected cauchy method. CLOSED_FORM mode exposes
    ``"R"`` (residual-only, for linesearch trial points and other
    cheap residual probes) and ``"R_and_dR_dU"`` (fused R + tangent
    for the Newton step). The string-key dict replaces Model's
    mutable ``_deriv_mode`` state machine: FE assembly vmaps closures
    over element batches, which works on pure functions but not on
    methods that mutate instance state.
    """

    # ---- attributes the subclass must set before super().__init__() ----
    dtype: type
    _is_complex: bool
    _num_eqs: NDArray[np.intp]
    _var_types: NDArray[np.intp]
    _ndims: int

    # ---- attributes set by self._init_residuals() ----
    num_residuals: int
    resid_names: list[str | None]
    var_names: list[str | None]

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
        self._var_types = np.zeros(num_residuals, dtype=int)
        self.resid_names = [None] * num_residuals
        self.var_names = [None] * num_residuals

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
        ``self.var_names`` (the field-symbol / dict-key carrier; the
        parallel ``self.resid_names`` carries the governing-equation
        label and is not consumed here). Subclasses with mixed-basis
        interpolation logic that can't be expressed through per-block
        iteration should override this method.
        """
        return interpolate_global_fields_at_ip(
            U, shapes_ip, self.var_names,
        )

    def for_model(
            self,
            model: Model,
            mode: GlobalResidualMode = GlobalResidualMode.COUPLED,
    ) -> GREvaluators:
        """Bind this GR to a concrete Model in a specific operational
        mode. Returns a dict of jit'd evaluators keyed by string names:

        - CLOSED_FORM: ``"R"`` (residual only, returning ``R_blocks``)
          and ``"R_and_dR_dU"`` (fused, returning
          ``(R_blocks, dR_dU_blocks)``). ``R_blocks`` is a list with
          entry ``r`` shaped ``(num_basis_fns_r, num_eqs_r)`` matching
          the GR's residual_fn return; ``dR_dU_blocks`` is a list-of-
          lists with ``dR_dU_blocks[r][s]`` shaped
          ``(num_basis_fns_r, num_eqs_r, num_basis_fns_s,
          num_eqs_s)``. The fused evaluator exists so XLA can CSE
          the shared work (interpolation, kinematics, Cauchy stress)
          when both R and its tangent are needed (Newton step); the
          R-only evaluator covers cheap-residual paths (linesearch
          trial points, perturbation probes).
        - COUPLED: not yet implemented; raises NotImplementedError.
          The coupled-mode evaluator must wrap the per-IP local
          Newton solve in custom_vjp so the global outer Newton sees
          the IFT-corrected ``dR/dU`` rather than differentiating
          through the inner solver.

        The same GR can be bound to multiple (model, mode) combinations
        by repeated ``for_model`` calls; ``mode`` is not stored on the
        instance.

        Raises ``NotImplementedError`` if ``mode == COUPLED``.
        Raises ``ValueError`` if ``mode == CLOSED_FORM`` and
        ``model.supports_closed_form_cauchy`` is False.
        """
        if mode == GlobalResidualMode.COUPLED:
            raise NotImplementedError(
                "GlobalResidual.for_model COUPLED mode is not yet "
                "implemented. The coupled-mode evaluator requires the "
                "per-IP local Newton solve to be wrapped in "
                "custom_vjp so the outer global Newton sees an "
                "IFT-corrected dR/dU, not a differentiated solver. "
                "Use CLOSED_FORM for now."
            )
        if (mode == GlobalResidualMode.CLOSED_FORM
                and not model.supports_closed_form_cauchy):
            raise ValueError(
                f"CLOSED_FORM mode requires "
                f"model.supports_closed_form_cauchy; got "
                f"{type(model).__name__} with the flag False"
            )

        residual_fn = self._residual_fn

        # R-composed: close the model into the public residual function.
        # Post-closure argnums: xi=0, xi_prev=1, params=2, U=3,
        # U_prev=4, shapes_ip=5, w=6, dv=7, ip_set=8.
        def r_at_ip(xi, xi_prev, params, U, U_prev, shapes_ip, w, dv, ip_set):
            return residual_fn(
                xi, xi_prev, params, U, U_prev,
                model, shapes_ip, w, dv, ip_set,
            )

        dR_dU_at_ip = jacfwd(r_at_ip, argnums=3)

        def r_and_dR_dU_at_ip(
                xi, xi_prev, params, U, U_prev, shapes_ip, w, dv, ip_set,
        ):
            R = r_at_ip(
                xi, xi_prev, params, U, U_prev,
                shapes_ip, w, dv, ip_set,
            )
            dR_dU = dR_dU_at_ip(
                xi, xi_prev, params, U, U_prev,
                shapes_ip, w, dv, ip_set,
            )
            return R, dR_dU

        return {
            "R": jit(r_at_ip),
            "R_and_dR_dU": jit(r_and_dR_dU_at_ip),
        }
