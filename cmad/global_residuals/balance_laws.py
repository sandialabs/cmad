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
    volume when ``model.is_finite_deformation``, times ``w dv``. The
    stress is ``model.cauchy_closed_form`` (CLOSED_FORM) or ``model.cauchy``
    (COUPLED) contracted to its leading ``ndims`` block; when ``mixed`` the
    model supplies the deviatoric part and the pressure field the
    hydrostatic part, ``sigma = dev - p I``.
    """
    n = ndims
    if mixed:
        if mode == GlobalResidualMode.CLOSED_FORM:
            dev = model.dev_cauchy_closed_form(params, U_ip, U_ip_prev)
        else:
            dev = model.dev_cauchy(xi, xi_prev, params, U_ip, U_ip_prev)
        p = U_ip.fields["p"][0]
        sigma = dev[:n, :n] - p * jnp.eye(n)
    else:
        if mode == GlobalResidualMode.CLOSED_FORM:
            assert model.cauchy_closed_form is not None
            sigma = model.cauchy_closed_form(params, U_ip, U_ip_prev)
        else:
            sigma = model.cauchy(xi, xi_prev, params, U_ip, U_ip_prev)
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

    ``-(p + hydro) / psf N_p - tau grad_N_p . grad p`` times ``w dv``:
    ``hydro`` is ``model.hydro_cauchy_closed_form`` (CLOSED_FORM) or
    ``model.hydro_cauchy`` (COUPLED), ``psf = model.pressure_scale_factor``,
    and ``tau = stabilization_multiplier 0.5 h^2 / mu`` with
    ``mu = model.shear_scale_factor``; in finite deformation the
    stabilization is scaled by ``(cof_F.T @ cof_F) / det F``.
    """
    if mode == GlobalResidualMode.CLOSED_FORM:
        hydro = model.hydro_cauchy_closed_form(params, U_ip, U_ip_prev)
    else:
        hydro = model.hydro_cauchy(xi, xi_prev, params, U_ip, U_ip_prev)
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
