from cmad.typing import Scalar


def compute_mu(E: Scalar, nu: Scalar) -> Scalar:
    return E / (2. * (1. + nu))


def compute_kappa(E: Scalar, nu: Scalar) -> Scalar:
    return E / (3. * (1. - 2. * nu))


def compute_lambda(E: Scalar, nu: Scalar) -> Scalar:
    return E * nu / ((1. + nu) * (1. - 2. * nu))
