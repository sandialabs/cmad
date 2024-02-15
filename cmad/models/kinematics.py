import jax.numpy as jnp

from cmad.models.deformation_types import DefType


def gather_F(xi, u, def_type, local_var_idx):
    if def_type == DefType.FULL_3D:
        return u[0]

    elif def_type == DefType.PLANE_STRESS:
        F_2D = u[0]
        F_33 = xi[local_var_idx]
        F = jnp.r_[jnp.c_[F_2D, jnp.zeros((2, 1))],
                   jnp.c_[jnp.zeros((1, 2)), F_33]]

        return F

    elif def_type == DefType.PLANE_STRAIN:
        F_2D = u[0]
        F = jnp.r_[jnp.c_[F_2D, jnp.zeros((2, 1))],
                   jnp.c_[jnp.zeros((1, 2)), 1.]]

        return F

    elif def_type == DefType.UNIAXIAL_STRESS:
        F_11 = u[0][0, 0]

        return jnp.diag(jnp.r_[F_11, xi[local_var_idx]])

    else:
        raise NotImplementedError


def compute_invariants(A):
    I1 = jnp.trace(A)
    I2 = 0.5 * (I1**2 - jnp.trace(A @ A))
    I3 = jnp.linalg.det(A)
    return I1, I2, I3
