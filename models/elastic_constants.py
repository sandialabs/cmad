def compute_mu(E, nu):
    return E / (2. * (1. + nu))


def compute_kappa(E, nu):
    return E / (3. * (1. - 2. * nu))


def compute_lambda(E, nu):
    return E * nu / ((1. + nu) * (1. - 2. * nu))
