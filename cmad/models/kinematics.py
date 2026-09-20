import jax.numpy as jnp
from jax import custom_jvp, jacfwd, jvp
from jax.lax import while_loop

from cmad.models.deformation_types import DefType
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.typing import JaxArray, StateList


def gather_F(
        xi: StateList, U: GlobalFieldsAtPoint, def_type: int,
        local_var_idx: int, uniaxial_stress_idx: int = 0,
) -> JaxArray:

    grad_u = U.grad_fields["u"]

    if def_type == DefType.FULL_3D:
        return jnp.eye(3) + grad_u

    elif def_type == DefType.PLANE_STRESS:
        F_2D = jnp.eye(2) + grad_u
        F_33 = xi[local_var_idx]
        F = jnp.r_[jnp.c_[F_2D, jnp.zeros((2, 1))],
                   jnp.c_[jnp.zeros((1, 2)), F_33]]

        return F

    elif def_type == DefType.PLANE_STRAIN:
        F_2D = jnp.eye(2) + grad_u
        F = jnp.r_[jnp.c_[F_2D, jnp.zeros((2, 1))],
                   jnp.c_[jnp.zeros((1, 2)), 1.]]

        return F

    elif def_type == DefType.UNIAXIAL_STRESS:
        on_axis_idx = uniaxial_stress_idx
        F_1D = jnp.eye(1) + grad_u
        F_uniaxial = F_1D[on_axis_idx, on_axis_idx]
        stretches = xi[local_var_idx]

        if on_axis_idx == 0:
            F = jnp.diag(jnp.r_[F_uniaxial, stretches])
        elif on_axis_idx == 1:
            F = jnp.diag(jnp.r_[stretches[0], F_uniaxial, stretches[1]])
        elif on_axis_idx == 2:
            F = jnp.diag(jnp.r_[stretches, F_uniaxial])
        else:
            raise ValueError("uniaxial_stress_idx != 0, 1, or 2")

        return F

    else:
        raise NotImplementedError


def compute_invariants(A: JaxArray) -> tuple[JaxArray, JaxArray, JaxArray]:
    I1 = jnp.trace(A)
    I2 = 0.5 * (I1**2 - jnp.trace(A @ A))
    I3 = det_3x3(A)
    return I1, I2, I3


def off_axis_idx(uniaxial_stress_idx: int) -> JaxArray:
    all_idx = jnp.arange(3)
    return jnp.sort(jnp.setdiff1d(all_idx, uniaxial_stress_idx, size=2))


def cofactor(F: JaxArray) -> JaxArray:
    """Cofactor matrix ``cof(F) = det(F) * F^{-T}`` of a 3x3 matrix, row
    ``i`` the cross product of the other two rows in cyclic order, or of
    a 2x2 matrix."""
    if F.shape == (2, 2):
        return jnp.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]])
    assert F.shape == (3, 3), F.shape
    return jnp.stack([
        jnp.cross(F[1], F[2]),
        jnp.cross(F[2], F[0]),
        jnp.cross(F[0], F[1]),
    ])


def det_3x3(A: JaxArray) -> JaxArray:
    """Determinant of a 3x3 matrix, ``A[0] . cof(A)[0]``."""
    assert A.shape == (3, 3), A.shape
    return jnp.sum(A[0] * cofactor(A)[0])


def inv_3x3(A: JaxArray) -> JaxArray:
    """Inverse of a 3x3 matrix, ``cof(A)^T / det(A)``."""
    assert A.shape == (3, 3), A.shape
    cof = cofactor(A)
    return cof.T / jnp.sum(A[0] * cof[0])


_POLAR_ROTATION_TOL = 1e-14
_POLAR_ROTATION_MAX_ITERS = 30


def _skew_matrix(w: JaxArray) -> JaxArray:
    return jnp.array([[0.0, -w[2], w[1]],
                      [w[2], 0.0, -w[0]],
                      [-w[1], w[0], 0.0]])


def _polar_rotation_equations(
        w: JaxArray, R: JaxArray, F: JaxArray,
) -> JaxArray:
    """The three components of ``skew(R_wᵀ F)`` for ``R_w = R (I + skew(w))``,
    zero at ``w = 0`` when ``R`` is the polar rotation of ``F``."""
    stretch = (R @ (jnp.eye(3) + _skew_matrix(w))).T @ F
    return jnp.stack([
        stretch[0, 1] - stretch[1, 0],
        stretch[0, 2] - stretch[2, 0],
        stretch[1, 2] - stretch[2, 1],
    ])


@custom_jvp
def polar_rotation(F: JaxArray) -> JaxArray:
    """Rotation ``R`` from the right polar decomposition ``F = R U``.

    The scaled Newton iteration ``R <- (g R + R⁻ᵀ / g) / 2`` from ``R = F``
    (Higham 1986), run until the relative change is under
    ``_POLAR_ROTATION_TOL``. The derivative is ``R skew(w)``, ``w`` from the
    implicit function theorem on ``_polar_rotation_equations``.
    """
    def cond_fun(carry: tuple[JaxArray, JaxArray, JaxArray]) -> JaxArray:
        k, R, R_prev = carry
        change = jnp.linalg.norm(R - R_prev)
        return jnp.logical_and(
            k < _POLAR_ROTATION_MAX_ITERS,
            change > _POLAR_ROTATION_TOL * jnp.linalg.norm(R))

    def body_fun(
            carry: tuple[JaxArray, JaxArray, JaxArray],
    ) -> tuple[JaxArray, JaxArray, JaxArray]:
        k, R, _ = carry
        R_inv_T = inv_3x3(R).T
        g = jnp.sqrt(jnp.linalg.norm(R_inv_T) / jnp.linalg.norm(R))
        return k + 1, 0.5 * (g * R + R_inv_T / g), R

    init = (jnp.zeros((), dtype=jnp.int32), F, jnp.zeros_like(F))
    return while_loop(cond_fun, body_fun, init)[1]


@polar_rotation.defjvp
def _polar_rotation_jvp(
        primals: tuple[JaxArray], tangents: tuple[JaxArray],
) -> tuple[JaxArray, JaxArray]:
    F = primals[0]
    dF = tangents[0]
    R = polar_rotation(F)
    zero = jnp.zeros(3)
    A = jacfwd(_polar_rotation_equations)(zero, R, F)
    _, b = jvp(
        lambda F_: _polar_rotation_equations(zero, R, F_), (F,), (dF,))
    return R, -R @ _skew_matrix(inv_3x3(A) @ b)


def small_strain_increment(F: JaxArray, F_prev: JaxArray) -> JaxArray:
    """Small strain increment ``ε − ε_prev = sym(F − F_prev)``."""
    dF = F - F_prev
    return 0.5 * (dF + dF.T)


def unrotated_rate_of_deformation_increment(
        F: JaxArray, F_prev: JaxArray,
) -> JaxArray:
    """Unrotated rate of deformation times the step,
    ``D Δt = Rᵀ sym((F - F_prev) F_mid⁻¹) R``, with
    ``F_mid = (F + F_prev) / 2`` (Hughes and Winget 1980) and
    ``R = polar_rotation(F_mid)``. Its trace is replaced by
    ``ln(det F / det F_prev)``, the exact integral of ``tr D``.
    """
    F_mid = 0.5 * (F + F_prev)
    R = polar_rotation(F_mid)
    L = (F - F_prev) @ inv_3x3(F_mid)
    midpoint = R.T @ (0.5 * (L + L.T)) @ R
    deviatoric = midpoint - jnp.trace(midpoint) / 3.0 * jnp.eye(3)
    volumetric = jnp.log(det_3x3(F) / det_3x3(F_prev))
    return deviatoric + volumetric / 3.0 * jnp.eye(3)
