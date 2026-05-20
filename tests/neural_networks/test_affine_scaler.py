"""Unit tests for AffineScaler."""

import numpy as np

from cmad.neural_networks.input_convex_neural_network import AffineScaler

# Column 0 spans [0, 4]; column 1 spans [10, 30].
SAMPLES = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])


def test_default_range_is_minus_one_to_one():
    scaler = AffineScaler().fit(SAMPLES)
    scaled = scaler.scale_ * SAMPLES + scaler.min_
    np.testing.assert_allclose(scaled.min(axis=0), [-1.0, -1.0])
    np.testing.assert_allclose(scaled.max(axis=0), [1.0, 1.0])


def test_unit_range_matches_closed_form():
    scaler = AffineScaler(feature_range=(0.0, 1.0)).fit(SAMPLES)
    np.testing.assert_allclose(scaler.scale_, [0.25, 0.05])
    np.testing.assert_allclose(scaler.min_, [0.0, -0.5])
    scaled = scaler.scale_ * SAMPLES + scaler.min_
    np.testing.assert_allclose(scaled.min(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(scaled.max(axis=0), [1.0, 1.0])


def test_constant_feature_maps_to_range_low():
    samples = np.array([[5.0], [5.0], [5.0]])
    scaler = AffineScaler(feature_range=(0.0, 1.0)).fit(samples)
    scaled = scaler.scale_ * samples + scaler.min_
    np.testing.assert_allclose(scaled, 0.0)
