"""Base class for constitutive models whose flux is stress.

Separates the mechanics-specific contract the mechanics global residual
relies on from the general :class:`cmad.models.model.Model`.
"""
from collections.abc import Callable

import numpy as np
from jax import jit
from numpy.typing import NDArray

from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.models.kinematics import gather_F
from cmad.models.model import Model
from cmad.typing import CauchyFn, JaxArray, Params, ResidualFn, Scalar, StateList


class MechanicsModel(Model):
    """Constitutive model whose flux is stress.

    The base the mechanics global residual binds to: the model maps the
    local deformation to a stress, which the residual assembles as the
    flux of the (quasi-static) momentum balance. Subclasses pass the
    Cauchy stress function (see CauchyFn in cmad.typing) beside the
    residual function to ``super().__init__()``, and the closed form
    stress function when the model has one; both are jit-compiled at
    construction and reached as ``cauchy`` and ``cauchy_closed_form``.
    On top of :class:`Model` it adds the two pieces the residual reads:

    - ``is_finite_deformation``: selects the form the GR assembles --
      finite (first Piola-Kirchhoff, ``sigma @ cof(F)``) or small strain
      (``grad_N @ sigma``). A member variable, not a ClassVar: a model
      can be built for either regime (``Elastic`` derives it from its
      stress function).
    - :meth:`deformation_gradient`: the 3x3 deformation gradient at an
      integration point, used for the finite Cauchy-to-PK1 map.

    It also declares the mixed formulation's contract, the two scale
    factors for the pressure equation, which a model with
    ``supports_mixed`` True overrides; the base raises.

    Subclasses set ``_def_type`` (and ``_oop_stretch_idx`` when they
    carry an out-of-plane stretch unknown) before ``super().__init__()``.
    """

    is_finite_deformation: bool = False

    _def_type: int
    # Index in xi of the plane stress out-of-plane stretch unknown; -1
    # when the model has no such unknown (FULL_3D, plane strain), where
    # gather_F ignores it.
    _oop_stretch_idx: int = -1

    cauchy_closed_form: Callable[..., JaxArray] | None
    _Sigma: NDArray[np.floating]

    def __init__(
            self, residual_fun: ResidualFn, cauchy_fun: CauchyFn,
            cauchy_closed_form_fun: Callable[..., JaxArray] | None = None,
    ) -> None:
        self.cauchy = jit(cauchy_fun)
        self.cauchy_closed_form = (
            jit(cauchy_closed_form_fun)
            if cauchy_closed_form_fun is not None else None
        )
        super().__init__(residual_fun)

    def evaluate_cauchy(self) -> None:
        """Evaluate the Cauchy stress at the gathered state, read by Sigma."""
        self._Sigma = np.asarray(
            self.cauchy(*self.variables()), dtype=np.float64)

    def Sigma(self) -> NDArray[np.floating]:
        return self._Sigma

    def deformation_gradient(
            self, xi: StateList, U: GlobalFieldsAtPoint,
    ) -> JaxArray:
        """3x3 deformation gradient F at an integration point.

        ``gather_F`` embeds the in-plane gradient into 3x3 per
        ``_def_type``: for plane stress it pulls ``F_33`` from the
        out-of-plane stretch unknown in xi (so the GR's ``cof(F)`` and
        ``det(F)`` carry it); for plane strain ``F_33 = 1``; for FULL_3D
        it is ``I + grad_u``.
        """
        return gather_F(xi, U, self._def_type, self._oop_stretch_idx)

    # The mixed formulation's contract; a model with supports_mixed True
    # overrides these.

    @staticmethod
    def pressure_scale_factor(params: Params) -> Scalar:
        raise NotImplementedError

    @staticmethod
    def shear_scale_factor(params: Params) -> Scalar:
        raise NotImplementedError
