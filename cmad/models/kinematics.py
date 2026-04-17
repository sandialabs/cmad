import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.models.deformation_types import DefType
from cmad.typing import GlobalList, JaxArray, StateList


def gather_F(
        xi: StateList, u: GlobalList, def_type: int,
        local_var_idx: int, uniaxial_stress_idx: int = 0,
) -> JaxArray | NDArray[np.floating]:

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
        on_axis_idx = uniaxial_stress_idx
        F_uniaxial = u[0][on_axis_idx, on_axis_idx]
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
    I3 = jnp.linalg.det(A)
    return I1, I2, I3


def off_axis_idx(uniaxial_stress_idx: int) -> JaxArray:
    all_idx = jnp.arange(3)
    return jnp.sort(jnp.setdiff1d(all_idx, uniaxial_stress_idx, size=2))
