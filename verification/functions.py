import numpy as np


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
                  + 2. * ((L * cauchy[1, 2])**2
                          + (M * cauchy[0, 2])**2
                          + (N * cauchy[0, 1])**2))

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
