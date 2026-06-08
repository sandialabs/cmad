"""Global-residual abstract contract and composed-helper builder."""
from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
import numpy as np
from jax import debug, jacfwd, jit
from jax.lax import axis_index
from numpy.typing import NDArray

from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.global_residuals.interpolation import interpolate_global_fields_at_ip
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.models.model import Model
from cmad.models.nonlinear_solver import make_newton_solve
from cmad.models.var_types import VarType
from cmad.typing import GREvaluators, JaxArray, ResidualFnGR

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem, FEState
    from cmad.fem.mesh import Mesh


class GlobalResidual(ABC):
    """Finite-element residual contract at an integration point.

    Subclasses set up a residual function (pure, no side effects) and
    pass it to super().__init__(). See :data:`cmad.typing.ResidualFnGR`
    for the required underlying-callable signature:

      (xi, xi_prev, params, U, U_prev, model, mode, shapes_ip, w,
       dv, h, ip_set) -> Sequence[Array]

    where xi/xi_prev are Model's per-integration-point local state
    (threaded through GR's call to model.cauchy; GR has no xi of its
    own), U/U_prev are per-residual-block lists of element-local
    basis-coefficient arrays with ``U[i].shape ==
    (num_basis_fns[i], num_eqs[i])``, model is the bound
    :class:`Model`, mode is the :class:`GlobalResidualMode` value the
    closure was built for (the body branches on it to dispatch the
    per-physics flux call), shapes_ip is a per-block list of
    :class:`ShapeFunctionsAtIP`, w/dv are the quadrature weight and
    reference-volume factor, h is the characteristic element size
    (RMS edge length) the stabilized mixed formulation reads, and
    ip_set is the integration-point-set
    index dispatched by the assembly layer (single-ip_set GRs ignore
    it; multi-ip_set GRs use it to dispatch term-specific
    contributions). The xi/xi_prev args are part of the underlying-
    callable contract because they are load-bearing for path-
    dependent mode-COUPLED bindings; in CLOSED_FORM mode the per-GR
    body ignores them.

    Subclasses pair with a concrete Model via
    :meth:`for_model(model, mode)`, which captures ``mode`` lexically
    in the closures it builds and returns a mode-specific ``dict[str,
    Callable]`` of jit'd public evaluators closed over
    ``model._residual``. CLOSED_FORM mode exposes 2 keys: ``"R"``
    (residual-only, for linesearch trial points and other cheap
    residual probes) and ``"R_and_dR_dU"`` (fused R + tangent with
    XLA CSE between R and its tangent — the Newton-step call). The
    CLOSED_FORM public closures are U-only — xi/xi_prev are bound to
    zeros internally because ``model.cauchy_closed_form`` is consulted
    instead. COUPLED mode exposes 2 keys, both 9-arg Newton-running
    (xi internally solved from ``xi_prev``): ``"R"`` (R-only, for
    global-FE linesearch trial points) and ``"R_and_dR_dU_and_xi"``
    (fused R + IFT-corrected tangent + converged xi for the FE-
    Newton hot path); see :meth:`for_model` for the full contract.
    Multiple ``for_model`` calls on the same GR instance produce
    independent closures — each captures its own ``mode`` value, so
    a mixed-mode FE problem that binds the same GR per block
    (CLOSED_FORM on one, COUPLED on another) works without
    instance-state contamination. The string-key dict replaces
    Model's mutable ``_deriv_mode`` state machine: FE assembly
    vmaps closures over element batches, which works on pure
    functions but not on methods that mutate instance state.
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
            ndims: int,
    ) -> "GlobalResidual":
        """Build a :class:`GlobalResidual` instance from the deck's
        ``residuals.global residual`` section. ``ndims`` is sourced
        from the mesh by the deck-side builder. The base stub raises
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

    def near_null_space(
            self, mesh: "Mesh",
    ) -> NDArray[np.floating] | None:
        """Near-null-space basis ``B`` for AMG hierarchy construction.

        :class:`cmad.fem.fe_problem.FEProblem` calls this once at
        construction and caches the result on
        :attr:`FEProblem.near_null_space`; the FE-Newton linear-solver
        dispatch threads it into
        :func:`pyamg.smoothed_aggregation_solver`'s ``B`` argument so
        the AMG coarse spaces preserve the GR's near-null modes
        through coarsening. Shape ``(num_total_dofs, num_modes)``,
        matching the global DOF layout from
        :class:`cmad.fem.dof.GlobalDofMap`.

        Default ``None`` selects pyamg's constant-vector fallback,
        correct for scalar diffusion-like operators. Mechanics GRs
        override to return rigid-body modes.
        """
        return None

    def primary_output_fields(self) -> list[tuple[str, VarType]]:
        """The GR's primary (nodal) output fields as ``(name, var_type)``.

        Sourced generically from ``var_names`` + ``_var_types`` -- the
        basis-coefficient fields this GR solves for, named by var_name
        (the field symbol, e.g. ``"u"``). These are the GR's nodal
        output surface: the deck's ``output.global residual`` selection
        resolves against this catalog, and an omitted selection writes
        all of them. Model-derived element fields (cauchy, local state
        variables) are Model-owned and resolved off the GR (see
        :mod:`cmad.fem.postprocess` and
        :func:`cmad.io.writers.resolve_fe_output_plan`).
        """
        return [
            (cast(str, self.var_names[r]), VarType(int(self._var_types[r])))
            for r in range(self.num_residuals)
        ]

    def evaluate_nodal_field(
            self,
            name: str,
            fe_problem: "FEProblem",
            fe_state: "FEState",
            step: int,
    ) -> NDArray[np.floating]:
        """Materialize one nodal field at one history step.

        ``step`` indexes ``fe_state.U_history`` /
        ``fe_state.t_history``. The returned array has shape
        ``(n_nodes, *components)`` with component count and meaning
        consistent with the field's ``var_type``;
        components are in cmad-internal order (the writer permutes
        SYM_TENSOR to Exodus order on the way out).

        Default raises ``ValueError``. Subclasses dispatch on
        ``name`` against their declared ``primary_output_fields()``
        entries.
        """
        raise ValueError(
            f"{type(self).__name__} does not implement nodal field "
            f"{name!r}; subclasses override evaluate_nodal_field to "
            f"dispatch on primary_output_fields()"
        )

    def for_model(
            self,
            model: Model,
            mode: GlobalResidualMode = GlobalResidualMode.COUPLED,
            local_newton_settings: dict[str, Any] | None = None,
            print_local_convergence: bool = False,
    ) -> GREvaluators:
        """Bind this GR to a concrete Model in a specific operational
        mode. ``mode`` is captured lexically in the closures this
        method returns and threaded as a positional arg into every
        call of the GR's ``residual_fn``; the residual body branches
        on it for the per-physics flux dispatch
        (``model.cauchy_closed_form(params, U_ip, U_ip_prev)`` for
        CLOSED_FORM, ``model.cauchy(xi, xi_prev, params, U_ip,
        U_ip_prev)`` for COUPLED). Returns a mode-specific dict of
        jit'd public evaluators keyed by string names:

        - CLOSED_FORM (2 keys, both 8-arg sig
          ``(params, U, U_prev, shapes_ip, w, dv, h, ip_set)`` —
          U-only because ``cauchy_closed_form`` never consults xi):
          ``"R"`` (residual only, for linesearch trial points and
          other cheap residual probes) and ``"R_and_dR_dU"`` (fused
          ``(R_blocks, dR_dU_blocks)`` with XLA CSE between R and
          its tangent — the Newton-step call). The closure binds
          xi/xi_prev to zero pytrees so the underlying ``residual_fn``
          contract is preserved unchanged. ``R_blocks`` is a list
          with entry ``r`` shaped ``(num_basis_fns_r, num_eqs_r)``;
          ``dR_dU_blocks`` is a list-of-lists with
          ``dR_dU_blocks[r][s]`` shaped
          ``(num_basis_fns_r, num_eqs_r, num_basis_fns_s,
          num_eqs_s)``.

        - COUPLED (2 keys, both 9-arg sig
          ``(params, U, U_prev, xi_prev, shapes_ip, w, dv, h,
          ip_set)``; xi internally solved from ``xi_prev`` via
          ``make_newton_solve`` wrapping ``model._residual``):
          ``"R"`` returning ``R_blocks`` (R-only — the per-IP
          Newton runs and the converged xi is discarded; intended
          for global-FE linesearch trial points and other R-norm
          probes), and ``"R_and_dR_dU_and_xi"`` returning
          ``(R_blocks, dR_dU_blocks, xi_converged)`` for the
          FE-Newton hot path. ``dR_dU_blocks`` is the IFT-
          corrected total via the ``custom_vjp`` rule on the
          per-IP Newton (the global tangent K, transpose K^T for
          the adjoint); state-history storage at FE-Newton
          convergence is the third return — no extra solve.

        ``local_newton_settings`` (COUPLED only) is forwarded to
        ``make_newton_solve`` as ``**kwargs``. Defaults
        ``abs_tol=1e-12, rel_tol=1e-12, max_iters=20`` apply when
        None. Passing this kwarg in CLOSED_FORM raises
        ``ValueError`` to match the deck-pluck pattern (don't
        accept-and-ignore).

        Multiple ``for_model`` calls on the same GR instance produce
        independent closures, each with its own captured ``mode``;
        a mixed-mode FE problem that binds the same GR per block
        (one CLOSED_FORM, one COUPLED) works without
        instance-state contamination.

        Raises ``ValueError`` if ``mode == CLOSED_FORM`` and
        ``model.supports_closed_form_cauchy`` is False, or if
        ``local_newton_settings`` is passed in CLOSED_FORM.
        """
        if mode == GlobalResidualMode.CLOSED_FORM:
            if local_newton_settings is not None:
                raise ValueError(
                    "local_newton_settings is only valid in COUPLED "
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
            if local_newton_settings is None:
                local_newton_settings = {
                    "abs_tol": 1e-12,
                    "rel_tol": 1e-12,
                    "max_iters": 20,
                }
            return self._for_model_coupled(
                model, local_newton_settings, print_local_convergence,
            )

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
        # shapes_ip=3, w=4, dv=5, h=6, ip_set=7.
        def r_at_ip(params, U, U_prev, shapes_ip, w, dv, h, ip_set):
            return residual_fn(
                xi_zeros, xi_zeros, params, U, U_prev,
                model, GlobalResidualMode.CLOSED_FORM,
                shapes_ip, w, dv, h, ip_set,
            )

        dR_dU_at_ip = jacfwd(r_at_ip, argnums=1)

        def r_and_dR_dU_at_ip(
                params, U, U_prev, shapes_ip, w, dv, h, ip_set,
        ):
            R = r_at_ip(
                params, U, U_prev, shapes_ip, w, dv, h, ip_set,
            )
            dR_dU = dR_dU_at_ip(
                params, U, U_prev, shapes_ip, w, dv, h, ip_set,
            )
            return R, dR_dU

        return {
            "R": jit(r_at_ip),
            "R_and_dR_dU": jit(r_and_dR_dU_at_ip),
        }

    def _for_model_coupled(
            self,
            model: Model,
            local_newton_settings: dict[str, Any],
            print_local_convergence: bool,
    ) -> GREvaluators:
        residual_fn = self._residual_fn

        # `make_newton_solve` conflates init guess and held-in-
        # residual x_prev (uses xi_prev as both); this matches
        # xi_init = xi_prev path continuity for plasticity.
        local_newton = make_newton_solve(
            model._residual,
            **local_newton_settings,
            print_local_convergence=print_local_convergence,
        )

        # Public-closure argnums:
        #   params=0, U=1, U_prev=2, xi_prev=3,
        #   shapes_ip=4, w=5, dv=6, h=7, ip_set=8.
        def coupled_r_total(params, U, U_prev, xi_prev,
                            shapes_ip, w, dv, h, ip_set):
            U_ip = self.interpolate_global_fields_at_ip(U, shapes_ip)
            U_ip_prev = self.interpolate_global_fields_at_ip(
                U_prev, shapes_ip)
            xi = local_newton(xi_prev, params, U_ip, U_ip_prev)
            return residual_fn(
                xi, xi_prev, params, U, U_prev,
                model, GlobalResidualMode.COUPLED,
                shapes_ip, w, dv, h, ip_set,
            )

        dR_dU_total = jacfwd(coupled_r_total, argnums=1)

        def r_and_dR_dU_and_xi_at_ip(params, U, U_prev, xi_prev,
                                     shapes_ip, w, dv, h, ip_set, ip_idx=0):
            if print_local_convergence:
                debug.print(
                    "[LOCAL elem={e} ip={i}]",
                    e=axis_index("elem"), i=ip_idx,
                )
            U_ip = self.interpolate_global_fields_at_ip(U, shapes_ip)
            U_ip_prev = self.interpolate_global_fields_at_ip(
                U_prev, shapes_ip)
            xi = local_newton(xi_prev, params, U_ip, U_ip_prev)
            R = residual_fn(
                xi, xi_prev, params, U, U_prev,
                model, GlobalResidualMode.COUPLED,
                shapes_ip, w, dv, h, ip_set,
            )
            dR_dU = dR_dU_total(
                params, U, U_prev, xi_prev,
                shapes_ip, w, dv, h, ip_set,
            )
            return R, dR_dU, xi

        return {
            "R": jit(coupled_r_total),
            "R_and_dR_dU_and_xi": jit(r_and_dR_dU_and_xi_at_ip),
        }
