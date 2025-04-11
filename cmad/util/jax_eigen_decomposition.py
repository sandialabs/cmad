"""
Combination of Harari and Albocher -- 2023 10.1002/nme.7311
and Scherzinger and Dohrmann -- 2008 10.1016/j.cma.2008.03.031

"""

import jax.numpy as jnp

from jax.lax import cond


# much faster than the implementation in here
def jax_compute_eigenvalues(A):
    eigenvalues, _ = jnp.linalg.eigh(A)
    return eigenvalues


def diagonal_differences(A):
    d_12 = A[0, 0] - A[1, 1]
    d_23 = A[1, 1] - A[2, 2]
    d_31 = A[2, 2] - A[0, 0]

    return d_12, d_23, d_31


def diagonal_components(A):
    A_11 = A[0, 0]
    A_22 = A[1, 1]
    A_33 = A[2, 2]

    return A_11, A_22, A_33


def off_diagonal_components(A):
    A_12 = A[0, 1]
    A_23 = A[1, 2]
    A_31 = A[2, 0]

    SA_12 = A_12**2
    SA_23 = A_23**2
    SA_31 = A_31**2

    return A_12, A_23, A_31, SA_12, SA_23, SA_31


def compute_J2(d_12, d_23, d_31, SA_12, SA_23, SA_31):
    J_2d = (d_12**2 + d_23**2 + d_31**2) / 6.
    J_2o = SA_12 + SA_23 + SA_31

    return J_2d + J_2o


def compute_J3(d_12, d_23, d_31, A_12, A_23, A_31,
        SA_12, SA_23, SA_31):

    J_3d = (d_12 - d_31) * (d_23 - d_12) * (d_31 - d_23) / 27.
    J_3m = -((d_12 - d_31) * SA_23 \
           + (d_23 - d_12) * SA_31 \
           + (d_31 - d_23) * SA_12) / 3.
    J_3o = 2. * A_12 * A_23 * A_31

    return J_3d + J_3m + J_3o


def compute_discriminant(d_12, d_23, d_31, A_12, A_23, A_31,
        SA_12, SA_23, SA_31):

    h_x = d_12 * d_23 * d_31 + SA_12 * d_12 + SA_23 * d_23 + SA_31 * d_31

    h_y1 = A_23 * ((2. * SA_23 - SA_31 - SA_12) + 2. * d_12 * d_31) \
         + A_12 * A_31 * (d_12 - d_31)
    h_y2 = A_31 * ((2. * SA_31 - SA_23 - SA_12) + 2. * d_23 * d_12) \
         + A_12 * A_23 * (d_23 - d_12)
    h_y3 = A_12 * ((2. * SA_12 - SA_23 - SA_31) + 2. * d_31 * d_23) \
         + A_31 * A_23 * (d_31 - d_23)

    h_z1 = A_23 * (SA_31 - SA_12) + A_12 * A_31 * d_23
    h_z2 = A_31 * (SA_12 - SA_23) + A_23 * A_12 * d_31
    h_z3 = A_12 * (SA_23 - SA_31) + A_31 * A_23 * d_12

    Delta = h_x**2 + h_y1**2 + h_y2**2 + h_y3**2 \
          + 15. * (h_z1**2 + h_z2**2 + h_z3**2)

    return Delta


def compute_deviator_eigenvalues(A):
    d_12, d_23, d_31 = diagonal_differences(A)
    A_12, A_23, A_31, SA_12, SA_23, SA_31 = off_diagonal_components(A)
    J2 = compute_J2(d_12, d_23, d_31, SA_12, SA_23, SA_31)
    J3 = compute_J3(d_12, d_23, d_31, A_12, A_23, A_31, SA_12, SA_23, SA_31)
    Delta = compute_discriminant(d_12, d_23, d_31, A_12, A_23, A_31,
        SA_12, SA_23, SA_31)

    sqrt_J2 = jnp.sqrt(J2)
    J2_32 = jnp.sqrt(J2) * J2

    sd = jnp.sign(J3)
    C3 = (J3 / 2.) * 3.**(1.5) / J2_32
    alpha = 2. / 3. * jnp.arctan(jnp.sqrt(Delta) / (2. * J2_32 \
        + 3. * jnp.sqrt(3) * sd * J3))

    eta_1 = 2. * sd * sqrt_J2 / jnp.sqrt(3) * jnp.cos(alpha)
    shared_component_eta_23 = sd * sqrt_J2 * jnp.sin(alpha)
    eta_2 = shared_component_eta_23 - eta_1 / 2.
    eta_3 = -shared_component_eta_23 - eta_1 / 2.

    return jnp.array([eta_1, eta_2, eta_3])


def compute_eigenvalues(A):
    return compute_deviator_eigenvalues(A) + jnp.trace(A) / 3.


def diagonal_decomposition(A):
    return jnp.diag(A), jnp.eye(3)


def compute_eigen_decomposition(A):
    abs_A = jnp.abs(A)
    off_diagonal_sum = jnp.sum(abs_A) - jnp.trace(abs_A)
    tol = 0.
    return cond(off_diagonal_sum > tol, non_diagonal_decomposition,
        diagonal_decomposition, A)


def non_diagonal_decomposition(A):
    deviator_eigenvalues = compute_deviator_eigenvalues(A)
    eta_1, eta_2, eta_3 = deviator_eigenvalues

    A_spherical = jnp.trace(A) / 3.
    I = jnp.eye(3)
    A_dev = A - A_spherical * I

    R = (A_dev - eta_1 * I)
    rvec_norms = jnp.linalg.norm(R, axis=0)
    s1_idx = jnp.argmax(rvec_norms)
    s23_idx = jnp.setdiff1d(jnp.arange(3), s1_idx, size=2)
    s1 = R[:, s1_idx] / rvec_norms[s1_idx]
    r2 = R[:, s23_idx[0]]
    r3 = R[:, s23_idx[1]]
    t2 = r2 - (s1 @ r2) * s1
    t3 = r3 - (s1 @ r3) * s1
    T = jnp.vstack((t2, t3)).T
    tvec_norms = jnp.linalg.norm(T, axis=0)
    s2_idx = jnp.argmax(tvec_norms)
    s2 = T[:, s2_idx] / tvec_norms[s2_idx]

    A_dev_s1 = A_dev @ s1
    A_dev_s2 = A_dev @ s2
    s1_A_dev_s2 = s1 @ A_dev_s2
    R2 = jnp.array([[s1 @ A_dev_s1, s1_A_dev_s2],
         [s1_A_dev_s2, s2 @ A_dev_s2]]) - eta_2 * jnp.eye(2)
    wvec_norm2s = jnp.sum(R2**2, axis=0)
    w_idx = jnp.argmax(wvec_norm2s)
    w = R2[:, w_idx]

    v1 = jnp.cross(s1, s2)
    unnormalized_v2 = w[0] * s2 - w[1] * s1
    v2 = unnormalized_v2 / jnp.linalg.norm(unnormalized_v2)
    v3 = jnp.cross(v1, v2)

    eigenvalues = deviator_eigenvalues + A_spherical
    eigenvectors = jnp.vstack((v1, v2, v3)).T

    return eigenvalues, eigenvectors


def sorted_eigen_decomposition(A):
    eigenvalues, eigenvectors = compute_eigen_decomposition(A)
    sorted_idx = jnp.argsort(eigenvalues)
    return eigenvalues[sorted_idx], eigenvectors[:, sorted_idx]
