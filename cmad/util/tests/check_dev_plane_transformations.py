import numpy as np


def compute_forward_and_backward_matrices():
    # rotation of pi / 4 about the s_2 axis
    R1 = np.array([
        [np.sqrt(2.) / 2., 0, -np.sqrt(2.) / 2],
        [0., 1., 0.],
        [np.sqrt(2.) / 2., 0, np.sqrt(2.) / 2]
    ])

    # rotation of -arccos(sqrt(2 / 3)) about the s'_1 axis
    R2 = np.array([
        [1., 0., 0.],
        [0., np.sqrt(2. / 3.), -np.sqrt(1. / 3.)],
        [0., np.sqrt(1. / 3.), np.sqrt(2. / 3.)]
    ])

    P = np.array([
        [np.sqrt(3. / 2.), 0., 0.],
        [0., np.sqrt(3. / 2.), 0.]
    ])

    L = np.array([
        [np.sqrt(2. / 3.), 0.],
        [0., np.sqrt(2. / 3.)],
        [0., 0.]
    ])

    # forward: from dev principals to dev plane
    F = P @ R2 @ R1
    # backward: from dev plane to dev principals
    B = R1.T @ R2.T @ L

    return F, B


def compute_dev_plane_angle(pi_coords):
    pi_1 = pi_coords[0, :]
    pi_2 = pi_coords[1, :]

    return np.mod(np.arctan2(pi_2, pi_1), 2. * np.pi)


def compute_dev_plane_radius(pi_coords):
    pi_1 = pi_coords[0, :]
    pi_2 = pi_coords[1, :]

    return np.sqrt(zeta_1**2 + zeta_2**2)


def compute_dev_plane_coords_from_dev_polar(radius, angle):
    return np.c_[radius * np.cos(angle), radius * np.sin(angle)]


diff_tol = 1e-14
num_angles = 720

# start at dev principal stresses = 2 / 3 * (1., -0.5, -0.5)
# which corresponds to principal stresses = (1., 0., 0.)
dev_plane_angles = np.mod(np.linspace(0., 2. * np.pi, num_angles,
    endpoint=False) - np.pi / 6., 2. * np.pi)
dev_plane_radii = np.ones(num_angles)
dev_plane_coords = \
    compute_dev_plane_coords_from_dev_polar(
    dev_plane_radii, dev_plane_angles
)

F, B = compute_forward_and_backward_matrices()
diff_tol = 1e-14

mapped_dev_principal_stresses = (B @ dev_plane_coords.T).T
mapped_dev_plane_coords = (F @ mapped_dev_principal_stresses.T).T
check_mapping_dev_plane_coords = np.linalg.norm(
    dev_plane_coords - mapped_dev_plane_coords
)
assert check_mapping_dev_plane_coords < diff_tol

psi_angles = np.linspace(0., 2. * np.pi, num_angles, endpoint=False)
dev_principal_stresses = 2. / 3. * np.cos(np.column_stack((psi_angles,
    psi_angles - 2. * np.pi / 3., psi_angles + 2. * np.pi / 3.)))
check_mapping_dev_principal_stresses = np.linalg.norm(
    dev_principal_stresses - mapped_dev_principal_stresses
)
assert check_mapping_dev_principal_stresses < diff_tol
