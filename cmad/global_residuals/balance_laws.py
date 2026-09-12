"""Balance laws at an integration point, shared by the global residuals.

Each function is one residual block of one balance law, written on the
fields interpolated at the point and the test functions of its own block,
so a global residual passes its block's shapes and routes the result. The
closed form versus coupled dispatch to the model lives here, so a global
residual class only routes its blocks.
"""
from jax import numpy as jnp

from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.models.kinematics import cofactor
from cmad.models.mechanics_model import MechanicsModel
from cmad.typing import JaxArray, Params, Scalar, StateList


def _cauchy_by_mode(
        xi: StateList,
        xi_prev: StateList,
        params: Params,
        U_ip: GlobalFieldsAtPoint,
        U_ip_prev: GlobalFieldsAtPoint,
        model: MechanicsModel,
        mode: GlobalResidualMode,
) -> JaxArray:
    """The model's 3x3 Cauchy stress, closed form or from the local state."""
    if mode == GlobalResidualMode.CLOSED_FORM:
        assert model.cauchy_closed_form is not None
        return model.cauchy_closed_form(params, U_ip, U_ip_prev)
    return model.cauchy(xi, xi_prev, params, U_ip, U_ip_prev)


def cauchy_with_pressure(sigma: JaxArray, p: Scalar) -> JaxArray:
    """The mixed formulation's stress: the deviatoric part of the 3x3
    ``sigma`` with the pressure field as its hydrostatic part,
    ``dev(sigma) - p I``."""
    return sigma - (jnp.trace(sigma) / 3. + p) * jnp.eye(3)


def momentum_balance(
        xi: StateList,
        xi_prev: StateList,
        params: Params,
        U_ip: GlobalFieldsAtPoint,
        U_ip_prev: GlobalFieldsAtPoint,
        model: MechanicsModel,
        mode: GlobalResidualMode,
        shapes_u: ShapeFunctionsAtIP,
        w: Scalar,
        dv: Scalar,
        ndims: int,
        mixed: bool,
) -> JaxArray:
    """Internal force of the quasi static momentum balance, shape
    ``(n_basis_u, ndims)``.

    ``grad_N @ sigma`` in small strain, or ``grad_N @ P.T`` with the first
    Piola-Kirchhoff stress ``P = sigma @ cofactor(F)`` over the reference
    volume when ``model.is_finite_deformation``, times ``w dv``. ``sigma``
    is the model's Cauchy stress (closed form or from the local state per
    ``mode``) contracted to its leading ``ndims`` block; when ``mixed`` its
    hydrostatic part is replaced by the pressure field
    (:func:`cauchy_with_pressure`).
    """
    n = ndims
    sigma = _cauchy_by_mode(xi, xi_prev, params, U_ip, U_ip_prev, model, mode)
    if mixed:
        sigma = cauchy_with_pressure(sigma, U_ip.fields["p"][0])
    sigma = sigma[:n, :n]
    if model.is_finite_deformation:
        F = model.deformation_gradient(xi, U_ip)
        cof_F = cofactor(F)[:n, :n]
        P = sigma @ cof_F
        return (shapes_u.grad_N @ P.T) * w * dv
    return (shapes_u.grad_N @ sigma) * w * dv


def pressure_equation(
        xi: StateList,
        xi_prev: StateList,
        params: Params,
        U_ip: GlobalFieldsAtPoint,
        U_ip_prev: GlobalFieldsAtPoint,
        model: MechanicsModel,
        mode: GlobalResidualMode,
        shapes_p: ShapeFunctionsAtIP,
        w: Scalar,
        dv: Scalar,
        h: Scalar,
        ndims: int,
        stabilization_multiplier: float,
) -> JaxArray:
    """Stabilized pressure equation of the mixed formulation, shape
    ``(n_basis_p, 1)``.

    ``-(p + hydro) / psf N_p - tau grad_N_p . grad p`` times ``w dv``, with
    ``hydro = tr(sigma) / 3`` of the model's Cauchy stress (closed form or
    from the local state per ``mode``), ``psf = model.pressure_scale_factor``,
    and ``tau = stabilization_multiplier 0.5 h^2 / mu`` with
    ``mu = model.shear_scale_factor``; in finite deformation the
    stabilization is scaled by ``(cof_F.T @ cof_F) / det F``.
    """
    sigma = _cauchy_by_mode(xi, xi_prev, params, U_ip, U_ip_prev, model, mode)
    hydro = jnp.trace(sigma) / 3.
    p = U_ip.fields["p"][0]
    psf = model.pressure_scale_factor(params)
    mu = model.shear_scale_factor(params)
    tau = stabilization_multiplier * 0.5 * h ** 2 / mu
    grad_p = U_ip.grad_fields["p"][0]
    if model.is_finite_deformation:
        F = model.deformation_gradient(xi, U_ip)
        cof_F = cofactor(F)[:ndims, :ndims]
        stab = tau * (cof_F.T @ cof_F) / jnp.linalg.det(F)
        stab_term = shapes_p.grad_N @ (stab @ grad_p)
    else:
        stab_term = tau * (shapes_p.grad_N @ grad_p)
    R_p = (-(p + hydro) / psf * shapes_p.N - stab_term) * w * dv
    return R_p[:, None]
