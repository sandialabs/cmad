"""Sensitivity-strategy dispatcher for ``cmad gradient`` / ``cmad hessian``.

Presents a uniform driver-facing surface across the two structurally
different objective families: :mod:`cmad.objectives.objective` (subclass
of ``Objective`` ABC, numpy-einsum adjoint / direct / direct-adjoint
implementations) and :class:`cmad.objectives.jvp_objective.JVPObjective`
(end-to-end JAX-traced).

The per-subcommand restriction (``cmad hessian`` requires
``direct_adjoint`` or ``jvp``) is enforced at factory time; the schema
keeps a single enum of all four strategy names.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from cmad.objectives.jvp_objective import JVPObjective
from cmad.objectives.objective import (
    AdjointObjective,
    DirectAdjointObjective,
    DirectObjective,
    Objective,
)
from cmad.qois.qoi import QoI
from cmad.solver.nonlinear_solver import make_newton_solve
from cmad.typing import GradientResult, HessianResult, PyTree, StateList


class SensitivityDriver(Protocol):
    """Uniform gradient / Hessian evaluation surface for the CLI."""

    def evaluate_grad(
        self, x: NDArray[np.floating],
    ) -> GradientResult: ...

    def evaluate_hess(
        self, x: NDArray[np.floating],
    ) -> HessianResult: ...


class _ObjectiveFamilyDriver:
    """Wraps an ``Objective`` ABC subclass into the ``SensitivityDriver`` surface.

    ``AdjointObjective`` / ``DirectObjective`` return ``GradientResult``;
    ``DirectAdjointObjective`` returns ``HessianResult``. Attempting
    ``evaluate_hess`` against a gradient-only variant is a factory-time
    error, not a runtime one — the assert here is defensive backstop.
    """

    def __init__(self, objective: Objective) -> None:
        self._obj = objective

    def evaluate_grad(
            self, x: NDArray[np.floating],
    ) -> GradientResult:
        result = self._obj.evaluate(x)
        if isinstance(result, HessianResult):
            return GradientResult(J=result.J, grad=result.grad)
        return result

    def evaluate_hess(
            self, x: NDArray[np.floating],
    ) -> HessianResult:
        result = self._obj.evaluate(x)
        assert isinstance(result, HessianResult), (
            f"evaluate_hess called on {type(self._obj).__name__}, which "
            f"produces a gradient-only result; factory should have "
            f"prevented this"
        )
        return result


class _JVPDriver:
    """Wraps :class:`JVPObjective` into the ``SensitivityDriver`` surface.

    Constructs the model-specific Newton solver helper
    (``make_newton_solve(model._residual, model._init_xi, **newton_kwargs)``)
    so the JVP-internal forward pass uses the same convergence tolerances
    as the driver-level Newton loop.
    """

    def __init__(
            self, qoi: QoI, newton_kwargs: dict[str, Any],
    ) -> None:
        model = qoi.model()
        # ``list`` invariance on the PyTree union means StateList is not
        # directly a PyTree to mypy; make_newton_solve returns the loose
        # ``Callable[..., PyTree]`` while JVPObjective declares the
        # tighter ``Callable[..., StateList]``. Both coincide at
        # runtime; casts narrow the static types.
        x0 = cast(PyTree, model._init_xi)
        update_fun = cast(
            Callable[..., StateList],
            make_newton_solve(model._residual, x0, **newton_kwargs),
        )
        self._jvp_obj = JVPObjective(qoi, update_fun)

    def evaluate_grad(
            self, x: NDArray[np.floating],
    ) -> GradientResult:
        J_jax, grad_jax = self._jvp_obj.evaluate_objective_and_grad(x)
        return GradientResult(
            J=float(np.asarray(J_jax)),
            grad=np.asarray(grad_jax, dtype=np.float64),
        )

    def evaluate_hess(
            self, x: NDArray[np.floating],
    ) -> HessianResult:
        J_jax, grad_jax = self._jvp_obj.evaluate_objective_and_grad(x)
        hess_jax = self._jvp_obj.evaluate_hessian(x)
        return HessianResult(
            J=float(np.asarray(J_jax)),
            grad=np.asarray(grad_jax, dtype=np.float64),
            hessian=np.asarray(hess_jax, dtype=np.float64),
        )


def build_sensitivity_driver(
        sensitivity_section: dict[str, Any],
        qoi: QoI,
        newton_kwargs: dict[str, Any],
        subcommand: str,
) -> SensitivityDriver:
    """Build the right driver for ``sensitivity.type`` and ``subcommand``.

    The schema already validated ``sensitivity.type`` is one of the four
    known names. This factory enforces the per-subcommand restriction:
    ``cmad hessian`` refuses ``adjoint`` / ``direct`` (they do not
    produce a Hessian), and ``cmad gradient`` accepts all four but
    warns on ``direct_adjoint`` since it computes a Hessian as a
    side effect (use ``cmad hessian`` instead to get it).
    """
    stype = sensitivity_section["type"]

    if subcommand == "hessian" and stype in ("adjoint", "direct"):
        raise ValueError(
            f"sensitivity.type: 'cmad hessian' requires "
            f"'direct_adjoint' or 'jvp'; got {stype!r}",
        )
    if subcommand == "calibrate" and stype == "direct_adjoint":
        # calibrate wires only first-order scipy methods (jac=True), so
        # direct_adjoint — which returns a Hessian — would compute work
        # nothing consumes.
        raise ValueError(
            f"sensitivity.type: 'cmad calibrate' accepts "
            f"'adjoint', 'direct', or 'jvp' (first-order only); "
            f"got {stype!r}",
        )
    if subcommand == "gradient" and stype == "direct_adjoint":
        print(
            "warning: sensitivity.type=direct_adjoint computes a Hessian "
            "as a side effect; for gradient-only work prefer 'adjoint', "
            "'direct', or 'jvp'",
            file=sys.stderr,
        )

    if stype == "adjoint":
        return _ObjectiveFamilyDriver(AdjointObjective(qoi))
    if stype == "direct":
        return _ObjectiveFamilyDriver(DirectObjective(qoi))
    if stype == "direct_adjoint":
        return _ObjectiveFamilyDriver(DirectAdjointObjective(qoi))
    if stype == "jvp":
        return _JVPDriver(qoi, newton_kwargs)
    raise ValueError(f"sensitivity.type: unknown value {stype!r}")
