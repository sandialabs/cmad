# https://doi.org/10.1007/978-3-642-38547-6
# See Section 3.2 of Plasticity by Ronaldo I. Borja

import numpy as np
import matplotlib.pyplot as plt


def compute_forward_and_backward_matrices(use_scaling=False):
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
        [1., 0., 0.],
        [0., 1., 0.]
    ])
    if use_scaling:
        P *= np.sqrt(3. / 2.)

    L = np.array([
        [1., 0.],
        [0., 1.],
        [0., 0.]
    ])
    if use_scaling:
        L *= np.sqrt(2. / 3.)

    # forward: from dev principals to dev plane
    F = P @ R2 @ R1
    # backward: from dev plane to dev principals
    B = R1.T @ R2.T @ L

    return F, B


def compute_matrix_from_projection(
        projection_values,
        projection_basis):

    assert len(projection_values) == 3
    assert projection_basis.shape == (3, 3)

    return projection_basis @ np.diag(projection_values) @ projection_basis.T


def setup_dev_plane_plot(axis_scale_factor=1.):
    axis_parametric_coords = np.array([-1., 1.]) * axis_scale_factor
    s1_axis = np.column_stack((np.sqrt(3.) / 2. * axis_parametric_coords,
        -0.5 * axis_parametric_coords))
    s2_axis = np.column_stack((0. * axis_parametric_coords,
        axis_parametric_coords))
    s3_axis = np.column_stack((-s1_axis[:, 0], s1_axis[:, 1]))

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.plot(s1_axis[:, 0], s1_axis[:, 1], color="black", zorder=0)
    plt.plot(s2_axis[:, 0], s2_axis[:, 1], color="black", zorder=0)
    plt.plot(s3_axis[:, 0], s3_axis[:, 1], color="black", zorder=0)
    ax.axis("equal")

    return fig, ax
