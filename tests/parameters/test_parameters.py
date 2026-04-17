"""
Verify Parameters round-tripping and transform chain-rule correctness.

Uses J2AnalyticalProblem's J2_parameters (with scale_params=True), which
gives a mixed-transform parameter tree: log scale on Y, bounds on S/D.
"""
import unittest

import numpy as np

from cmad.parameters.parameters import (
    transform_from_canonical,
    transform_to_canonical,
)
from tests.support.test_problems import J2AnalyticalProblem


def fresh_parameters():
    return J2AnalyticalProblem(scale_params=True).J2_parameters


class TestParametersRoundTrip(unittest.TestCase):

    def test_round_trip_canonical(self):
        parameters = fresh_parameters()
        original = parameters.flat_active_values(return_canonical=True)

        # Inject canonical values back, read them out canonical.
        parameters.set_active_values_from_flat(original, are_canonical=True)
        round_tripped = parameters.flat_active_values(return_canonical=True)

        np.testing.assert_allclose(original, round_tripped, rtol=1e-12)

    def test_round_trip_raw(self):
        parameters = fresh_parameters()
        original = parameters.flat_active_values(return_canonical=False)

        # Inject raw values back, read them out raw.
        parameters.set_active_values_from_flat(original, are_canonical=False)
        round_tripped = parameters.flat_active_values(return_canonical=False)

        np.testing.assert_allclose(original, round_tripped, rtol=1e-12)


class TestTransformChainRule(unittest.TestCase):

    def test_transform_grad_matches_FD(self):
        """transform_grad applied to raw_grad = ones should give the
        per-axis Jacobian d(raw)/d(canonical), which equals
        first_deriv_transform(value, transform). FD-check that.
        """
        parameters = fresh_parameters()
        canonical = parameters.flat_active_values(return_canonical=True)
        transforms = parameters._flat_active_transforms
        n = parameters.num_active_params

        # Apply transform_grad to a grad of all ones.
        # In-place mutation: grad[i] = first_deriv * 1 = first_deriv.
        analytical_grad = np.ones(n)
        parameters.transform_grad(analytical_grad)

        # FD: for each i, compute d(transform_from_canonical_i)/d(canonical_i).
        h = 1e-5
        fd_grad = np.zeros(n)
        for i in range(n):
            val_plus = transform_from_canonical(
                canonical[i] + h, True, transforms[i])
            val_minus = transform_from_canonical(
                canonical[i] - h, True, transforms[i])
            fd_grad[i] = (val_plus - val_minus) / (2 * h)

        np.testing.assert_allclose(
            analytical_grad, fd_grad, rtol=1e-6, atol=1e-8)

    def test_transform_hessian_matches_FD(self):
        """transform_hessian applied to raw_hessian = I and raw_grad = ones
        should give a diagonal Hessian whose diagonal entries equal
        first_deriv**2 + second_deriv (per-axis chain rule for the
        second derivative). FD-check the diagonal; off-diagonals should
        be exactly zero.
        """
        parameters = fresh_parameters()
        canonical = parameters.flat_active_values(return_canonical=True)
        transforms = parameters._flat_active_transforms
        n = parameters.num_active_params

        raw_hessian = np.eye(n)
        raw_grad = np.ones(n)

        # In-place: hessian[i,i] = 1 * first_deriv**2 + 1 * second_deriv;
        #          hessian[i,j] = 0 * first_deriv_i * first_deriv_j = 0 (i!=j).
        analytical_hessian = raw_hessian.copy()
        analytical_grad = raw_grad.copy()
        parameters.transform_hessian(analytical_hessian, analytical_grad)

        # Off-diagonals exactly zero (raw hessian was I).
        off_diag_mask = ~np.eye(n, dtype=bool)
        np.testing.assert_array_equal(
            analytical_hessian[off_diag_mask], 0.0)

        # FD diagonal: d(first_deriv_fd)**2 + second_deriv_fd at each i,
        # where first_deriv_fd and second_deriv_fd come from FD of
        # transform_from_canonical w.r.t. canonical_i.
        h = 1e-4
        fd_diagonal = np.zeros(n)
        for i in range(n):
            val_plus = transform_from_canonical(
                canonical[i] + h, True, transforms[i])
            val_center = transform_from_canonical(
                canonical[i], True, transforms[i])
            val_minus = transform_from_canonical(
                canonical[i] - h, True, transforms[i])

            first_deriv_fd = (val_plus - val_minus) / (2 * h)
            second_deriv_fd = (val_plus - 2 * val_center + val_minus) / h**2

            fd_diagonal[i] = first_deriv_fd**2 + second_deriv_fd

        np.testing.assert_allclose(
            np.diag(analytical_hessian), fd_diagonal, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    unittest.TextTestRunner(verbosity=2).run(suite)
