import numpy as np
import unittest

from cmad.util.dev_plane_transformations import (
    compute_forward_and_backward_matrices
)


class TestDevPlane(unittest.TestCase):
    def test_dev_plane_transformations(self):

        num_angles = 720
        Y = 1.
        diff_tol = 1e-12

        for use_scaling in (True, False):
            # start at deviatoric principal stresses = 2 / 3 * (1., -0.5, -0.5)
            # which corresponds to principal stresses = (1., 0., 0.)
            dev_plane_angles = np.mod(np.linspace(0., 2. * np.pi, num_angles,
                endpoint=False) - np.pi / 6., 2. * np.pi)
            dev_plane_radii = Y * np.ones(num_angles)
            dev_plane_coords = np.c_[
                dev_plane_radii * np.cos(dev_plane_angles),
                dev_plane_radii * np.sin(dev_plane_angles)
            ]
            psi_angles = np.linspace(0., 2. * np.pi, num_angles, endpoint=False)
            dev_principal_stresses = Y * 2. / 3. * np.cos(np.column_stack((
                psi_angles,
                psi_angles - 2. * np.pi / 3., psi_angles + 2. * np.pi / 3.)))

            F, B = compute_forward_and_backward_matrices(use_scaling)

            # unit circle in devatoric plane maps to
            # sqrt(3. / 2.) * dev_principal_stresses
            # for the unscaled transformation
            if not use_scaling:
                dev_principal_stresses *= np.sqrt(3. / 2.)

            mapped_dev_principal_stresses = (B @ dev_plane_coords.T).T
            mapped_dev_plane_coords = (F @ dev_principal_stresses.T).T

            check_mapping_dev_plane_coords = np.linalg.norm(
                dev_plane_coords - mapped_dev_plane_coords
            )
            assert check_mapping_dev_plane_coords / Y < diff_tol

            check_mapping_dev_principal_stresses = np.linalg.norm(
                dev_principal_stresses - mapped_dev_principal_stresses
            )
            assert check_mapping_dev_principal_stresses / Y < diff_tol


if __name__ == "__main__":
    dev_plane_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestDevPlane)
    unittest.TextTestRunner(verbosity=2).run(dev_plane_test_suite)
