"""Global-residual abstract contract and composed-helper builder."""
from abc import ABC
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit
from numpy.typing import NDArray

from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.global_residuals.interpolation import interpolate_global_fields_at_ip
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.solver.nonlinear_solver import make_newton_solve
from cmad.typing import GREvaluators, JaxArray, ResidualFnGR


class GlobalResidual(ABC):
    """Finite-element residual contract at an integration point.

    Subclasses set up a residual function (pure, no side effects) and
    pass it to super().__init__(). See :data:`cmad.typing.ResidualFnGR`
    for the required underlying-callable signature:

      (xi, xi_prev, params, U, U_prev, model, shapes_ip, w, dv,
       ip_set) -> Sequence[Array]

    where xi/xi_prev are Model's per-integration-point local state
    (threaded through GR's call to model.cauchy; GR has no xi of its
    own), U/U_prev are per-residual-block lists of element-local
    basis-coefficient arrays with ``U[i].shape ==
    (num_basis_fns[i], num_eqs[i])``, shapes_ip is a per-block list of
    :class:`ShapeFunctionsAtIP`, model is the bound :class:`Model`,
    w/dv are the quadrature weight and reference-volume factor, and
    ip_set is the integration-point-set index dispatched by the
    assembly layer (single-ip_set GRs ignore it; multi-ip_set GRs
    use it to dispatch term-specific contributions). The xi/xi_prev
    args are part of the underlying-callable contract because they
    are load-bearing for path-dependent mode-COUPLED bindings; in
    CLOSED_FORM mode the per-GR body ignores them.

    Subclasses pair with a concrete Model via
    :meth:`for_model(model, mode)`, which sets ``self._mode`` (so
    the GR subclass ``residual_fn`` body can branch on it for
    cauchy dispatch) and returns a mode-specific ``dict[str,
    Callable]`` of jit'd public evaluators closed over
    ``model._residual``. CLOSED_FORM mode exposes 5 keys: ``"R"``
    (residual-only, for linesearch trial points and other cheap
    residual probes), ``"R_and_dR_dU"`` (fused R + tangent for the
    Newton step), and standalone first-derivative evaluators
    ``"dR_dU"`` / ``"dR_dU_prev"`` / ``"dR_dp"`` (one per
    differentiable input of the U-only closure, for adjoint and
    gradient-assembly consumers that want a named derivative
    without R recomputation). The CLOSED_FORM public closures are
    U-only — xi/xi_prev are bound to zeros internally because
    ``model.cauchy_closed_form`` is consulted instead. COUPLED
    mode exposes a different 7-key dict mixing raw partials at
    supplied ξ (9-arg sig including xi) with Newton-running totals
    (8-arg sig; xi internally solved from ``xi_prev``); see
    :meth:`for_model` for the full contract. A single GR instance
    binds to one mode at a time — repeated ``for_model`` calls
    mutate ``self._mode``. The string-key dict replaces Model's
    mutable ``_deriv_mode`` state machine: FE assembly vmaps
    closures over element batches, which works on pure functions
    but not on methods that mutate instance state.
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

    # ---- attribute set by for_model() ----
    _mode: GlobalResidualMode

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
            coupled_newton_kwargs: dict[str, Any] | None = None,
    ) -> GREvaluators:
        """Bind this GR to a concrete Model in a specific operational
        mode. Sets ``self._mode = mode`` so the GR subclass
        ``residual_fn`` body can branch on it for cauchy dispatch
        (``model.cauchy_closed_form(params, U_ip, U_ip_prev)`` for
        CLOSED_FORM, ``model.cauchy(xi, xi_prev, params, U_ip,
        U_ip_prev)`` for COUPLED). Returns a mode-specific dict of
        jit'd public evaluators keyed by string names:

        - CLOSED_FORM (5 keys, all 7-arg sig
          ``(params, U, U_prev, shapes_ip, w, dv, ip_set)`` —
          U-only because ``cauchy_closed_form`` never consults xi):
          ``"R"`` (residual only, for linesearch trial points and
          other cheap residual probes), ``"R_and_dR_dU"`` (fused
          ``(R_blocks, dR_dU_blocks)`` with XLA CSE between R and
          its tangent), and standalone first-derivative evaluators
          ``"dR_dU"`` / ``"dR_dU_prev"`` / ``"dR_dp"`` (each is
          ``jacfwd`` / ``jacrev`` over the U-only closure wrt one
          differentiable input — for adjoint and gradient-assembly
          consumers that don't need R recomputed). The closure
          binds xi/xi_prev to zero pytrees so the underlying
          ``residual_fn`` contract is preserved unchanged.
          ``R_blocks`` is a list with entry ``r`` shaped
          ``(num_basis_fns_r, num_eqs_r)``; ``dR_dU_blocks`` is a
          list-of-lists with ``dR_dU_blocks[r][s]`` shaped
          ``(num_basis_fns_r, num_eqs_r, num_basis_fns_s,
          num_eqs_s)``; ``"dR_dp"`` returns a per-residual-block
          pytree parallel to ``params``.

        - COUPLED (7 keys, mixed sigs). Raw partials at supplied ξ
          (9-arg sig
          ``(params, U, U_prev, xi, xi_prev, shapes_ip, w, dv,
          ip_set)``): ``"R"``, ``"dR_dU_prev"``, ``"dR_dp"``,
          ``"dR_dxi"``, ``"dR_dxi_prev"``. No local Newton, no IFT
          correction — the consumer supplies xi at the public
          boundary. Newton-running totals (8-arg sig
          ``(params, U, U_prev, xi_prev, shapes_ip, w, dv,
          ip_set)``; xi internally solved from ``xi_prev`` via
          ``make_newton_solve`` wrapping ``model._residual``):
          ``"dR_dU"`` (IFT-corrected total via ``custom_jvp`` —
          the global tangent K, transpose K^T for the adjoint) and
          the bundled ``"R_and_dR_dU_and_xi"`` returning
          ``(R_blocks, dR_dU_blocks, xi_converged)`` for the FE-
          Newton hot path (state-history storage at FE-Newton
          convergence is the bundled call's third element — no
          extra solve).

        ``coupled_newton_kwargs`` (COUPLED only) is forwarded to
        ``make_newton_solve`` as ``**kwargs``. Defaults
        ``abs_tol=1e-12, rel_tol=1e-12, max_iters=20`` apply when
        None. Passing this kwarg in CLOSED_FORM raises
        ``ValueError`` to match the deck-pluck pattern (don't
        accept-and-ignore).

        A single GR instance binds to one mode at a time — calling
        ``for_model`` again mutates ``self._mode``, so closures
        from a previous binding will read the new mode. Tests that
        compare both modes simultaneously construct two GR
        instances.

        Raises ``ValueError`` if ``mode == CLOSED_FORM`` and
        ``model.supports_closed_form_cauchy`` is False, or if
        ``coupled_newton_kwargs`` is passed in CLOSED_FORM.
        """
        self._mode = mode

        if mode == GlobalResidualMode.CLOSED_FORM:
            if coupled_newton_kwargs is not None:
                raise ValueError(
                    "coupled_newton_kwargs is only valid in COUPLED "
                    "mode; got non-None value with mode=CLOSED_FORM"
                )
            if not model.supports_closed_form_cauchy:
                raise ValueError(
                    f"CLOSED_FORM mode requires "
                    f"model.supports_closed_form_cauchy; got "
                    f"{type(model).__name__} with the flag False"
                )
            return self._for_model_closed_form(model)

        if mode == GlobalResidualMode.COUPLED:
            if coupled_newton_kwargs is None:
                coupled_newton_kwargs = {
                    "abs_tol": 1e-12,
                    "rel_tol": 1e-12,
                    "max_iters": 20,
                }
            return self._for_model_coupled(model, coupled_newton_kwargs)

        raise ValueError(f"Unknown GlobalResidualMode: {mode}")

    def _for_model_closed_form(self, model: Model) -> GREvaluators:
        residual_fn = self._residual_fn

        # CLOSED_FORM closures are U-only on the public boundary;
        # the underlying residual_fn keeps xi/xi_prev because the
        # contract is shared with COUPLED-capable bindings. Bind
        # zero pytrees once so both public closures capture the same
        # reference.
        xi_zeros = [jnp.zeros_like(b) for b in model._init_xi]

        # Public-closure argnums: params=0, U=1, U_prev=2,
        # shapes_ip=3, w=4, dv=5, ip_set=6.
        def r_at_ip(params, U, U_prev, shapes_ip, w, dv, ip_set):
            return residual_fn(
                xi_zeros, xi_zeros, params, U, U_prev,
                model, shapes_ip, w, dv, ip_set,
            )

        dR_dU_at_ip = jacfwd(r_at_ip, argnums=1)
        dR_dU_prev_at_ip = jacfwd(r_at_ip, argnums=2)
        dR_dp_at_ip = jacrev(r_at_ip, argnums=0)

        def r_and_dR_dU_at_ip(
                params, U, U_prev, shapes_ip, w, dv, ip_set,
        ):
            R = r_at_ip(
                params, U, U_prev, shapes_ip, w, dv, ip_set,
            )
            dR_dU = dR_dU_at_ip(
                params, U, U_prev, shapes_ip, w, dv, ip_set,
            )
            return R, dR_dU

        return {
            "R": jit(r_at_ip),
            "R_and_dR_dU": jit(r_and_dR_dU_at_ip),
            "dR_dU": jit(dR_dU_at_ip),
            "dR_dU_prev": jit(dR_dU_prev_at_ip),
            "dR_dp": jit(dR_dp_at_ip),
        }

    def _for_model_coupled(
            self,
            model: Model,
            coupled_newton_kwargs: dict[str, Any],
    ) -> GREvaluators:
        residual_fn = self._residual_fn

        # ─── Raw closures (9-arg sig: xi as input) ────────────────
        # Public-closure argnums:
        #   params=0, U=1, U_prev=2, xi=3, xi_prev=4,
        #   shapes_ip=5, w=6, dv=7, ip_set=8.
        def r_at_supplied_xi(params, U, U_prev, xi, xi_prev,
                             shapes_ip, w, dv, ip_set):
            return residual_fn(
                xi, xi_prev, params, U, U_prev,
                model, shapes_ip, w, dv, ip_set,
            )

        dR_dU_prev_raw = jacfwd(r_at_supplied_xi, argnums=2)
        dR_dp_raw = jacrev(r_at_supplied_xi, argnums=0)
        dR_dxi_raw = jacfwd(r_at_supplied_xi, argnums=3)
        dR_dxi_prev_raw = jacfwd(r_at_supplied_xi, argnums=4)

        # ─── Newton-running closures (8-arg sig: no xi) ──────────
        # `make_newton_solve` conflates init guess and held-in-
        # residual x_prev (uses xi_prev as both); this matches
        # xi_init = xi_prev path continuity for plasticity.
        local_newton = make_newton_solve(
            model._residual, **coupled_newton_kwargs,
        )

        # Public-closure argnums:
        #   params=0, U=1, U_prev=2, xi_prev=3,
        #   shapes_ip=4, w=5, dv=6, ip_set=7.
        def coupled_r_total(params, U, U_prev, xi_prev,
                            shapes_ip, w, dv, ip_set):
            U_ip = self.interpolate_global_fields_at_ip(U, shapes_ip)
            U_ip_prev = self.interpolate_global_fields_at_ip(
                U_prev, shapes_ip)
            xi = local_newton(xi_prev, params, U_ip, U_ip_prev)
            return residual_fn(
                xi, xi_prev, params, U, U_prev,
                model, shapes_ip, w, dv, ip_set,
            )

        dR_dU_total = jacfwd(coupled_r_total, argnums=1)

        def r_and_dR_dU_and_xi_at_ip(params, U, U_prev, xi_prev,
                                     shapes_ip, w, dv, ip_set):
            U_ip = self.interpolate_global_fields_at_ip(U, shapes_ip)
            U_ip_prev = self.interpolate_global_fields_at_ip(
                U_prev, shapes_ip)
            xi = local_newton(xi_prev, params, U_ip, U_ip_prev)
            R = residual_fn(
                xi, xi_prev, params, U, U_prev,
                model, shapes_ip, w, dv, ip_set,
            )
            dR_dU = dR_dU_total(
                params, U, U_prev, xi_prev,
                shapes_ip, w, dv, ip_set,
            )
            return R, dR_dU, xi

        return {
            # Raw partials at supplied ξ (9-arg sig)
            "R": jit(r_at_supplied_xi),
            "dR_dU_prev": jit(dR_dU_prev_raw),
            "dR_dp": jit(dR_dp_raw),
            "dR_dxi": jit(dR_dxi_raw),
            "dR_dxi_prev": jit(dR_dxi_prev_raw),
            # Newton-running totals (8-arg sig)
            "dR_dU": jit(dR_dU_total),
            "R_and_dR_dU_and_xi": jit(r_and_dR_dU_and_xi_at_ip),
        }
