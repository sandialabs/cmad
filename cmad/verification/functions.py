import numpy as np
import jax.numpy as jnp

from cmad.util.jax_eigen_decomposition \
    import compute_eigenvalues as jax_compute_eigenvalues


def J2_yield(cauchy):
    hydro_cauchy = np.trace(cauchy) / 3.
    s = cauchy - hydro_cauchy * np.eye(3)
    snorm = np.linalg.norm(s)
    phi = np.sqrt(3. / 2.) * snorm

    return phi


def J2_yield_normal(cauchy):
    hydro_cauchy = np.trace(cauchy) / 3.
    s = cauchy - hydro_cauchy * np.eye(3)
    snorm = np.linalg.norm(s)
    normal = np.sqrt(3. / 2.) * s / snorm

    return normal


def hill_yield(cauchy, hill_params):
    F, G, H, L, M, N = hill_params

    phi = np.sqrt(F * (cauchy[1, 1] - cauchy[2, 2])**2
                  + G * (cauchy[2, 2] - cauchy[0, 0])**2
                  + H * (cauchy[0, 0] - cauchy[1, 1])**2
                  + 2. * (L * cauchy[1, 2]**2
                          + M * cauchy[0, 2]**2
                          + N * cauchy[0, 1]**2))

    return phi


def hill_yield_normal(cauchy, hill_params):
    F, G, H, L, M, N = hill_params

    n_00 = (G + H) * cauchy[0, 0] - H * cauchy[1, 1] - G * cauchy[2, 2]
    n_11 = (F + H) * cauchy[1, 1] - H * cauchy[0, 0] - F * cauchy[2, 2]
    n_22 = (G + F) * cauchy[2, 2] - G * cauchy[0, 0] - F * cauchy[1, 1]
    n_01 = N * cauchy[0, 1]
    n_02 = M * cauchy[0, 2]
    n_12 = L * cauchy[1, 2]

    normal = np.array([[n_00, n_01, n_02],
                       [n_01, n_11, n_12],
                       [n_02, n_12, n_22]]) / hill_yield(cauchy, hill_params)

    return normal


def jax_hill_yield(cauchy, hill_params):
    F, G, H, L, M, N = hill_params

    phi = jnp.sqrt(F * (cauchy[1, 1] - cauchy[2, 2])**2
                   + G * (cauchy[2, 2] - cauchy[0, 0])**2
                   + H * (cauchy[0, 0] - cauchy[1, 1])**2
                   + 2. * (L * cauchy[1, 2]**2
                           + M * cauchy[0, 2]**2
                           + N * cauchy[0, 1]**2))

    return phi


def jax_hill_yield_normal(cauchy, hill_params):
    F, G, H, L, M, N = hill_params

    n_00 = (G + H) * cauchy[0, 0] - H * cauchy[1, 1] - G * cauchy[2, 2]
    n_11 = (F + H) * cauchy[1, 1] - H * cauchy[0, 0] - F * cauchy[2, 2]
    n_22 = (G + F) * cauchy[2, 2] - G * cauchy[0, 0] - F * cauchy[1, 1]
    n_01 = N * cauchy[0, 1]
    n_02 = M * cauchy[0, 2]
    n_12 = L * cauchy[1, 2]

    normal = jnp.array([[n_00, n_01, n_02],
                       [n_01, n_11, n_12],
                       [n_02, n_12, n_22]]) \
        / jax_hill_yield(cauchy, hill_params)

    return normal


def jax_unpack_barlat_params(barlat_params):
    sp_cparams = barlat_params[:9]  # single prime c params
    dp_cparams = barlat_params[9:-1]  # double prime c params

    sp_12, sp_13, sp_21, sp_23, sp_31, sp_32, sp_44, sp_55, sp_66 \
        = sp_cparams
    dp_12, dp_13, dp_21, dp_23, dp_31, dp_32, dp_44, dp_55, dp_66 \
        = dp_cparams

    L_sp_upper_left = jnp.array([
        [sp_12 + sp_13, -2. * sp_12 + sp_13, sp_12 - 2. * sp_13],
        [-2. * sp_21 + sp_23, sp_21 + sp_23, sp_21 - 2. * sp_23],
        [-2. * sp_31 + sp_32, sp_31 - 2. * sp_32, sp_31 + sp_32]]) / 3.
    L_sp_lower_right = jnp.diag(jnp.array([sp_44, sp_55, sp_66]))
    L_sp = jnp.vstack((jnp.c_[L_sp_upper_left, jnp.zeros((3, 3))],
                       jnp.c_[jnp.zeros((3, 3)), L_sp_lower_right]))

    L_dp_upper_left = jnp.array([
        [dp_12 + dp_13, -2. * dp_12 + dp_13, dp_12 - 2. * dp_13],
        [-2. * dp_21 + dp_23, dp_21 + dp_23, dp_21 - 2. * dp_23],
        [-2. * dp_31 + dp_32, dp_31 - 2. * dp_32, dp_31 + dp_32]]) / 3.
    L_dp_lower_right = jnp.diag(jnp.array([dp_44, dp_55, dp_66]))
    L_dp = jnp.vstack((jnp.c_[L_dp_upper_left, jnp.zeros((3, 3))],
                       jnp.c_[jnp.zeros((3, 3)), L_dp_lower_right]))

    return L_sp, L_dp


def jax_flatten_stress(stress_matrix):
    flat_stress = jnp.array([stress_matrix[0, 0],
                             stress_matrix[1, 1],
                             stress_matrix[2, 2],
                             stress_matrix[0, 1],
                             stress_matrix[1, 2],
                             stress_matrix[2, 0]])

    return flat_stress


def jax_unflatten_stress(flat_stress):
    stress_matrix = jnp.array([[flat_stress[0],
                                flat_stress[3],
                                flat_stress[5]],
                               [flat_stress[3],
                                flat_stress[1],
                                flat_stress[4]],
                               [flat_stress[5],
                                flat_stress[4],
                                flat_stress[2]]])

    return stress_matrix


def jax_compute_sbar_matrices(cauchy, barlat_params):
    L_sp, L_dp = jax_unpack_barlat_params(barlat_params)
    flat_cauchy = jax_flatten_stress(cauchy)
    sbar_sp = jax_unflatten_stress(L_sp @ flat_cauchy)
    sbar_dp = jax_unflatten_stress(L_dp @ flat_cauchy)

    return sbar_sp, sbar_dp


def jax_barlat_yield(cauchy, barlat_params):
    a = barlat_params[-1]
    sbar_sp, sbar_dp = jax_compute_sbar_matrices(cauchy, barlat_params)
    sbar_sp_1, sbar_sp_2, sbar_sp_3 = jax_compute_eigenvalues(sbar_sp)
    sbar_dp_1, sbar_dp_2, sbar_dp_3 = jax_compute_eigenvalues(sbar_dp)

    phi = (0.25 * (jnp.abs(sbar_sp_1 - sbar_dp_1)**a
                   + jnp.abs(sbar_sp_1 - sbar_dp_2)**a
                   + jnp.abs(sbar_sp_1 - sbar_dp_3)**a
                   + jnp.abs(sbar_sp_2 - sbar_dp_1)**a
                   + jnp.abs(sbar_sp_2 - sbar_dp_2)**a
                   + jnp.abs(sbar_sp_2 - sbar_dp_3)**a
                   + jnp.abs(sbar_sp_3 - sbar_dp_1)**a
                   + jnp.abs(sbar_sp_3 - sbar_dp_2)**a
                   + jnp.abs(sbar_sp_3 - sbar_dp_3)**a))**(1. / a)

    return phi
