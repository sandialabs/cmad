"""The closed form 3x3 helpers against ``jnp.linalg``, and the unrotated
rate of deformation on a rigid rotation and on a stretch increment."""
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jacfwd

from cmad.models.kinematics import (
    cofactor,
    det_3x3,
    inv_3x3,
    unrotated_rate_of_deformation,
)


def _matrices(dtype: type) -> list[jnp.ndarray]:
    rng = np.random.default_rng(0)
    out = []
    for _ in range(4):
        A = rng.standard_normal((3, 3))
        if dtype is complex:
            A = A + 1j * rng.standard_normal((3, 3))
        out.append(jnp.asarray(A + 3.0 * np.eye(3)))
    return out


@pytest.mark.parametrize("dtype", [float, complex])
def test_helpers_match_linalg(dtype: type) -> None:
    for A in _matrices(dtype):
        det = jnp.linalg.det(A)
        np.testing.assert_allclose(det_3x3(A), det, rtol=1e-12)
        np.testing.assert_allclose(inv_3x3(A), jnp.linalg.inv(A), rtol=1e-12)
        np.testing.assert_allclose(
            cofactor(A), det * jnp.linalg.inv(A).T, rtol=1e-12,
        )


def test_derivatives_match_linalg() -> None:
    for A in _matrices(float):
        np.testing.assert_allclose(
            jacfwd(det_3x3)(A), jacfwd(jnp.linalg.det)(A), rtol=1e-12,
        )
        np.testing.assert_allclose(
            jacfwd(inv_3x3)(A), jacfwd(jnp.linalg.inv)(A), rtol=1e-10,
        )


def test_cofactor_of_2x2() -> None:
    A = jnp.asarray([[2.0, -1.0], [0.5, 3.0]])
    np.testing.assert_allclose(
        cofactor(A), jnp.linalg.det(A) * jnp.linalg.inv(A).T, rtol=1e-14,
    )


def test_shape_is_checked() -> None:
    with pytest.raises(AssertionError):
        det_3x3(jnp.eye(2))
    with pytest.raises(AssertionError):
        cofactor(jnp.eye(4))


def _rotation(angle: float) -> np.ndarray:
    """Rotation by ``angle`` about the axis (1, 2, 3)."""
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def test_rate_of_deformation_vanishes_for_a_rigid_rotation_increment() -> None:
    rng = np.random.default_rng(1)
    F_prev = np.eye(3) + 0.1 * rng.standard_normal((3, 3))
    F = _rotation(0.3) @ F_prev
    D = unrotated_rate_of_deformation(jnp.asarray(F), jnp.asarray(F_prev), 1.0)
    np.testing.assert_allclose(D, 0.0, atol=1e-14)


def test_stretch_increment_error_falls_by_eight_when_the_increment_halves() -> None:
    def error(d: float) -> float:
        F = jnp.diag(jnp.asarray([1.0 + d, 1.0, 1.0]))
        D = unrotated_rate_of_deformation(F, jnp.eye(3), 1.0)
        return abs(float(D[0, 0]) - float(np.log1p(d)))

    ratio = error(0.04) / error(0.02)
    assert 7.0 < ratio < 8.5
