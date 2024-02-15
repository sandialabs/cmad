import numpy as np

from jax import jit


def compute_elastic_fields(stress_tensor_at_yield,
    min_stress_scale, max_stress_scale, elastic_params, num_steps):

    E, nu = elastic_params
    stress_scale = np.linspace(min_stress_scale, max_stress_scale, num_steps)

    # stress
    stress = np.repeat(stress_tensor_at_yield[:, :, np.newaxis], num_steps,
        axis=2)
    for idx, scale in enumerate(stress_scale):
        stress[:, :, idx] *= scale
    I = np.eye(3)
    trace_stress_I = np.array([np.trace(stress[:, :, ii])
                               for ii in range(num_steps)]) \
        * np.repeat(I[:, :, np.newaxis], num_steps, axis=2)

    # strain
    strain = (stress - nu * (trace_stress_I - stress)) / E

    return stress, strain


def compute_plastic_fields(stress_mask, yield_fun, yield_normal_fun,
    isotropic_params, max_alpha, num_steps):

    yield_normal_fun = yield_normal_fun

    E, nu, Y, S, D = isotropic_params
    alpha = np.linspace(0., max_alpha, num_steps)
    alpha_rate = np.diff(alpha)[0]

    # stress
    stress = np.repeat(stress_mask[:, :, np.newaxis], num_steps, axis=2)
    scale_factor = yield_fun(stress_mask)
    stress_values = (Y + S * (1. - np.exp(-D * alpha))) / scale_factor
    stress *= stress_values
    I = np.eye(3)
    trace_stress_I = np.array([np.trace(stress[:, :, ii])
                               for ii in range(num_steps)]) \
        * np.repeat(I[:, :, np.newaxis], num_steps, axis=2)

    # plastic strain
    pstrain = np.zeros((3, 3, num_steps))
    for ii in range(1, num_steps):
        pstrain[:, :, ii] = alpha_rate * \
            yield_normal_fun(stress[:, :, ii]) + pstrain[:, :, ii - 1]

    # strain
    strain = (stress - nu * (trace_stress_I - stress)) / E + pstrain

    return stress, strain, alpha
