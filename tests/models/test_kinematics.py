"""The closed form 3x3 helpers against ``jnp.linalg``."""
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jacfwd

from cmad.models.kinematics import cofactor, det_3x3, inv_3x3


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
