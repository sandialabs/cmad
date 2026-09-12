"""Base class for constitutive models whose flux is stress.

Separates the mechanics-specific contract the mechanics global residual
relies on from the general :class:`cmad.models.model.Model`.
"""
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.models.kinematics import gather_F
from cmad.models.model import Model
from cmad.typing import JaxArray, Params, Scalar, StateList


class MechanicsModel(Model):
    """Constitutive model whose flux is stress.

    The base the mechanics global residual binds to: the model maps the
    local deformation to a stress, which the residual assembles as the
    flux of the (quasi-static) momentum balance. On top of
    :class:`Model` it adds the two pieces that residual reads:

    - ``is_finite_deformation``: selects the form the GR assembles --
      finite (first Piola-Kirchhoff, ``sigma @ cof(F)``) or small strain
      (``grad_N @ sigma``). A member variable, not a ClassVar: a model
      can be built for either regime (``Elastic`` derives it from its
      stress function).
    - :meth:`deformation_gradient`: the 3x3 deformation gradient at an
      integration point, used for the finite Cauchy-to-PK1 map.

    It also declares the mixed formulation's contract, the deviatoric and
    hydrostatic stress splits and the two scale factors, which a model
    with ``supports_mixed`` True overrides; the base raises.

    Subclasses set ``_def_type`` (and ``_oop_stretch_idx`` when they
    carry an out-of-plane stretch unknown) before ``super().__init__()``.
    """

    is_finite_deformation: bool = False

    _def_type: int
    # Index in xi of the plane stress out-of-plane stretch unknown; -1
    # when the model has no such unknown (FULL_3D, plane strain), where
    # gather_F ignores it.
    _oop_stretch_idx: int = -1

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

    def dev_cauchy(
            self,
            xi: StateList, xi_prev: StateList, params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> JaxArray:
        raise NotImplementedError

    def hydro_cauchy(
            self,
            xi: StateList, xi_prev: StateList, params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> Scalar:
        raise NotImplementedError

    @staticmethod
    def dev_cauchy_closed_form(
            params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> JaxArray:
        raise NotImplementedError

    @staticmethod
    def hydro_cauchy_closed_form(
            params: Params,
            U: GlobalFieldsAtPoint, U_prev: GlobalFieldsAtPoint,
    ) -> Scalar:
        raise NotImplementedError

    @staticmethod
    def pressure_scale_factor(params: Params) -> Scalar:
        raise NotImplementedError

    @staticmethod
    def shear_scale_factor(params: Params) -> Scalar:
        raise NotImplementedError
