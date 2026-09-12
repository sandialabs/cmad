import jax.numpy as jnp
from jax.scipy.linalg import polar

from cmad.models.deformation_types import DefType
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.typing import JaxArray, Scalar, StateList


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
    ``i`` the cross product of the other two rows in cyclic order."""
    assert F.shape == (3, 3), F.shape
    return jnp.stack([
        jnp.cross(F[1], F[2]),
        jnp.cross(F[2], F[0]),
        jnp.cross(F[0], F[1]),
    ])


def det_3x3(A: JaxArray) -> JaxArray:
    """Determinant of a 3x3 matrix, ``A[0] . cof(A)[0]``."""
    return jnp.sum(A[0] * cofactor(A)[0])


def inv_3x3(A: JaxArray) -> JaxArray:
    """Inverse of a 3x3 matrix, ``cof(A)^T / det(A)``."""
    cof = cofactor(A)
    return cof.T / jnp.sum(A[0] * cof[0])


def polar_rotation(F: JaxArray) -> JaxArray:
    """Rotation ``R`` from the right polar decomposition ``F = R U``.

    ``U`` is the symmetric positive definite right stretch. Uses the QDWH
    iteration (QR + matmul, no singular vectors), which stays
    differentiable at repeated singular values.
    """
    return polar(F, side="right", method="qdwh")[0]


def unrotated_rate_of_deformation(
        F: JaxArray, F_prev: JaxArray, dt: Scalar,
) -> JaxArray:
    """Unrotated rate of deformation ``D = Rᵀ sym(Ḟ F⁻¹) R``.

    The velocity gradient ``Ḟ F⁻¹`` is taken as the backward difference
    ``(F - F_prev) F⁻¹ / dt``; ``D`` is the symmetric part pulled back to
    the unrotated frame by ``R = polar_rotation(F)``.
    """
    R = polar_rotation(F)
    L = (F - F_prev) @ inv_3x3(F) / dt
    D = 0.5 * (L + L.T)
    return R.T @ D @ R
