from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from cmad.typing import Scalar


def compute_mu(E: Scalar, nu: Scalar) -> Scalar:
    return E / (2. * (1. + nu))


def compute_kappa(E: Scalar, nu: Scalar) -> Scalar:
    return E / (3. * (1. - 2. * nu))


def compute_lambda(E: Scalar, nu: Scalar) -> Scalar:
    return E * nu / ((1. + nu) * (1. - 2. * nu))


_CONSTANT_NAMES = ("E", "nu", "mu", "kappa", "lambda")


@dataclass(frozen=True)
class ElasticConstants:
    """Isotropic linear-elastic constants, canonicalized to the Lame pair.

    Any two of the five standard constants (``E``, ``nu``, ``mu``,
    ``kappa``, ``lambda``) determine the rest. Stored as the Lame pair
    ``(lmbda, mu)`` -- the form the constitutive models consume directly
    (``sigma = lmbda*tr(eps)*I + 2*mu*eps``) -- with ``kappa`` / ``E`` /
    ``nu`` derived on demand.
    """

    lmbda: Scalar
    mu: Scalar

    @property
    def kappa(self) -> Scalar:
        return self.lmbda + 2. * self.mu / 3.

    @property
    def E(self) -> Scalar:
        return (
            self.mu * (3. * self.lmbda + 2. * self.mu)
            / (self.lmbda + self.mu)
        )

    @property
    def nu(self) -> Scalar:
        return self.lmbda / (2. * (self.lmbda + self.mu))

    @classmethod
    def from_params(cls, elastic: dict[str, Any]) -> "ElasticConstants":
        """Build from any two of ``{E, nu, mu, kappa, lambda}``.

        Routes the given pair to the Lame pair ``(lmbda, mu)``. Raises
        ``ValueError`` unless exactly two recognized constants are present.
        """
        given = tuple(n for n in _CONSTANT_NAMES if n in elastic)
        if len(given) != 2:
            raise ValueError(
                f"ElasticConstants needs exactly two of {_CONSTANT_NAMES}; "
                f"got {given}"
            )
        pair = frozenset(given)

        if pair == frozenset(("lambda", "mu")):
            lmbda, mu = elastic["lambda"], elastic["mu"]
        elif pair == frozenset(("E", "nu")):
            E, nu = elastic["E"], elastic["nu"]
            lmbda, mu = compute_lambda(E, nu), compute_mu(E, nu)
        elif pair == frozenset(("mu", "kappa")):
            mu, kappa = elastic["mu"], elastic["kappa"]
            lmbda = kappa - 2. * mu / 3.
        elif pair == frozenset(("E", "mu")):
            E, mu = elastic["E"], elastic["mu"]
            lmbda = mu * (E - 2. * mu) / (3. * mu - E)
        elif pair == frozenset(("E", "kappa")):
            E, kappa = elastic["E"], elastic["kappa"]
            mu = 3. * kappa * E / (9. * kappa - E)
            lmbda = 3. * kappa * (3. * kappa - E) / (9. * kappa - E)
        elif pair == frozenset(("mu", "nu")):
            mu, nu = elastic["mu"], elastic["nu"]
            lmbda = 2. * mu * nu / (1. - 2. * nu)
        elif pair == frozenset(("kappa", "nu")):
            kappa, nu = elastic["kappa"], elastic["nu"]
            mu = 3. * kappa * (1. - 2. * nu) / (2. * (1. + nu))
            lmbda = 3. * kappa * nu / (1. + nu)
        elif pair == frozenset(("lambda", "nu")):
            lmbda, nu = elastic["lambda"], elastic["nu"]
            mu = lmbda * (1. - 2. * nu) / (2. * nu)
        elif pair == frozenset(("lambda", "kappa")):
            lmbda, kappa = elastic["lambda"], elastic["kappa"]
            mu = 3. * (kappa - lmbda) / 2.
        elif pair == frozenset(("E", "lambda")):
            # the one pair needing a quadratic root
            E, lmbda = elastic["E"], elastic["lambda"]
            R = jnp.sqrt(E**2 + 9. * lmbda**2 + 2. * E * lmbda)
            mu = (E - 3. * lmbda + R) / 4.
        else:  # unreachable: given is filtered + length-checked above
            raise ValueError(f"unsupported elastic-constant pair: {given}")

        return cls(lmbda=lmbda, mu=mu)
