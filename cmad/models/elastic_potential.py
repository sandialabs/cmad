import jax.numpy as jnp

from jax import grad

from cmad.models.kinematics import compute_invariants


def compute_cauchy_from_psi_b(F, u, params, psi_b_fun):
    b = F @ F.T
    invariants_b = compute_invariants(b)
    I1, I2, I3 = invariants_b
    J = jnp.sqrt(I3)

    dpsi_dIs = grad(psi_b_fun)(invariants_b, params)
    cauchy = 2. * J**-1 \
        * (I3 * dpsi_dIs[2] * jnp.eye(3)
           + (dpsi_dIs[0] + I1 * dpsi_dIs[1]) * b
            - dpsi_dIs[1] * b @ b)

    return cauchy


def compressible_neohookean_potential(invariants, params):
    """ From Computational Plasticity by Simo and Hughes"""

    I1, I2, I3 = invariants
    J = jnp.sqrt(I3)
    Jm23 = jnp.cbrt(J)**-2

    kappa = params["elastic"]["kappa"]
    mu = params["elastic"]["mu"]

    psi_bulk = 0.5 * kappa * (0.5 * (J**2 - 1.) - jnp.log(J))
    psi_shear = 0.5 * mu * (Jm23 * I1 - 3.)

    return psi_bulk + psi_shear
