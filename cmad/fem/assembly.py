"""Element + global FE assembly machinery."""
import jax.numpy as jnp

from cmad.typing import JaxArray


def iso_jac_at_ip(
        grad_N_ref: JaxArray, X_elem: JaxArray,
) -> tuple[JaxArray, JaxArray, JaxArray]:
    """Per-IP isoparametric Jacobian and physical-frame shape gradients.

    Args:
        grad_N_ref: ``(nnodes, ndims)`` reference-frame ∂N_a/∂ξ_j from
            the element's interpolant (:mod:`cmad.fem.interpolants`).
        X_elem: ``(nnodes, ndims)`` physical coords of element nodes.

    Returns:
        ``grad_N_phys``: ``(nnodes, ndims)`` physical-frame ∂N_a/∂x_i.

        ``iso_jac_det``: scalar ``det(∂x/∂ξ)``. **Signed** — a negative
        value indicates element inversion. Returned without ``abs(...)``
        so inversion propagates as a hard failure (Newton diverges
        visibly) rather than silent garbage. Mesh-correctness
        (positive volumes) is the mesh builder / loader's
        responsibility, not assembly's.

        ``iso_jac``: ``(ndims, ndims)`` Jacobian matrix with
        ``iso_jac[j, i] = ∂x_i/∂ξ_j`` (transpose-of-classical-Jacobian
        convention, chosen so ``grad_N_phys = grad_N_ref @ inv(iso_jac)``
        follows directly from the chain rule). Naming ``iso_jac`` /
        ``iso_jac_det`` avoids collision with continuum-mechanics
        ``J = det(F)``.

    Convention check
    ----------------
    For an isoparametric element with reference coords ξ and physical
    coords x = Σ_a N_a(ξ) X_a:

    - ``iso_jac[j, i] = Σ_a (∂N_a/∂ξ_j) X_a_i = (grad_N_ref.T @ X_elem)[j, i]``.
    - The chain rule gives
      ``∂N_a/∂x_i = Σ_j (∂N_a/∂ξ_j)(∂ξ_j/∂x_i) =
      grad_N_ref[a, j] * inv(iso_jac)[j, i] = (grad_N_ref @ inv(iso_jac))[a, i]``.
    - ``det(iso_jac) = det(∂x/∂ξ)`` is the volume scaling factor.
    """
    iso_jac = grad_N_ref.T @ X_elem
    iso_jac_det = jnp.linalg.det(iso_jac)
    inv_iso_jac = jnp.linalg.inv(iso_jac)
    grad_N_phys = grad_N_ref @ inv_iso_jac
    return grad_N_phys, iso_jac_det, iso_jac
