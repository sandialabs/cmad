"""
support code for Al7079 calibration
"""

import numpy as np

import jax.numpy as jnp

from jax import tree_map

from cmad.parameters.parameters import Parameters


def compute_R(orig_basis):
    standard_basis = np.eye(3)
    R = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            R[i, j] = standard_basis[i, :] @ orig_basis[j, :]
    return R


def compute_rotation_A(alpha):
    basis_A = np.array([[1., 0., 0.],
        [0., -np.sin(alpha), -np.cos(alpha)],
        [0., np.cos(alpha), -np.sin(alpha)]])
    return compute_R(basis_A)


def compute_rotation_B(beta):
    basis_B = np.array([[0., np.sin(beta), np.cos(beta)],
        [1., 0., 0.],
        [0., np.cos(beta), -np.sin(beta)]])
    return compute_R(basis_B)


def compute_rotation_C(gamma):
    basis_C = np.array([[np.cos(gamma), np.sin(gamma), 0.],
        [np.sin(gamma), -np.cos(gamma), 0.],
        [0., 0., 1.]])
    return compute_R(basis_C)


def slab_data(angle_type):
    assert angle_type in ["alpha", "beta", "gamma"]
    degrees_to_rad = np.pi / 180.

    if angle_type == "alpha":
        angles = np.array([0., 15., 30., 45., 60., 75., 90.]) \
            * degrees_to_rad
        sigma_c_values = np.array([525., 512., 515., 505., 493., 511., 530.])
        ratio_c_values = np.array([0.18, 0.27, 0.75, 1.2, 1.0, 0.7, 0.91])
        rotation_fun = compute_rotation_A

    if angle_type == "beta":
        angles = np.array([45., 60., 90.]) * degrees_to_rad
        sigma_c_values = np.array([510., 544., 523.])
        ratio_c_values = np.array([2.9, 1.5, 1.1])
        rotation_fun = compute_rotation_B


    if angle_type == "gamma":
        angles = np.array([45., 60.]) * degrees_to_rad
        sigma_c_values = np.array([486., 485.])
        ratio_c_values = np.array([0.47, 0.52])
        rotation_fun = compute_rotation_C

    R_matrices = [rotation_fun(angle) for angle in angles]

    return angles, sigma_c_values, ratio_c_values, R_matrices


def calibration_weights():
    # weight_sigma, weight_ratio
    return np.array([10., 1.])


def calibrated_hill_coefficients():
    # F, G, H, L, M, N
    return np.array([0.1477, 0.6805, 0.5345, 1.7977, 1.7148, 2.1675])

def calibrated_barlat_coefficients():
    # sp_12, sp_13, sp_21, sp_23, sp_31, sp_32, sp_44, sp_55, sp_66,
    # dp_12, sp_13, sp_21, sp_23, sp_31, sp_32, sp_44, sp_55, sp_66,
    # a
    coefficients = np.array([
        0.4555, 1.0274, 0.7101, 1.3755, 0.5314, 0.8817, 1.0558, 1.1133, 0.9220,
        1.2431, 1.5438, 1.2204, 0.7632, 0.5327, 0.3015, 0.9722, 0.7399, 1.0760,
        18.2])
    return coefficients


def params_hill_voce(p_elastic, p_hill, p_voce):
    E, nu = p_elastic
    Y, F, G, H, L, M, N = p_hill
    S, D = p_voce

    elastic_params = {"E": E, "nu": nu}
    J2_effective_stress_params = {"J2": 0.}
    initial_yield_params = {"Y": Y}
    voce_params = {"S": S, "D": D}
    hardening_params = {"voce": voce_params}

    hill_coefficients = {"F": F, "G": G, "H": H, "L": L, "M": M, "N": N}
    hill_effective_stress_params = {"hill": hill_coefficients}
    hill_coefficients_bounds = np.array([0.1, 3.])

    hill_values = {
        "elastic": elastic_params,
        "plastic": {
            "effective stress": hill_effective_stress_params,
            "flow stress": {
                "initial yield": initial_yield_params,
                "hardening": hardening_params}}}

    hill_active_flags = hill_values.copy()
    hill_active_flags = tree_map(lambda a: False, hill_active_flags)
    for key in hill_coefficients:
        hill_active_flags["plastic"]["effective stress"]["hill"][key] = True

    hill_transforms = hill_values.copy()
    hill_transforms = tree_map(lambda a: None, hill_transforms)
    for key in hill_coefficients:
        hill_transforms["plastic"]["effective stress"]["hill"][key] = \
            hill_coefficients_bounds

    hill_parameters = \
        Parameters(hill_values, hill_active_flags, hill_transforms)

    return hill_parameters


def params_hybrid_hill_voce(p_elastic, p_hill, p_voce, nn_params):
    E, nu = p_elastic
    Y, F, G, H, L, M, N = p_hill
    S, D = p_voce

    elastic_params = {"E": E, "nu": nu}
    initial_yield_params = {"Y": Y}
    voce_params = {"S": S, "D": D}
    hardening_params = {"voce": voce_params}

    hill_coefficients = {"F": F, "G": G, "H": H, "L": L, "M": M, "N": N}
    hill_effective_stress_params = {"hill": hill_coefficients}
    hill_coefficients_bounds = np.array([0.1, 3.])

    hybrid_hill_values = {
        "elastic": elastic_params,
        "plastic": {
            "effective stress": {
                "hill": hill_coefficients,
                "neural network": nn_params},
            "flow stress": {
                "initial yield": initial_yield_params,
                "hardening": hardening_params}}}

    hybrid_hill_active_flags = hybrid_hill_values.copy()
    hybrid_hill_active_flags = tree_map(lambda a: False,
        hybrid_hill_active_flags)
    hybrid_hill_active_flags["plastic"]["effective stress"]["neural network"] \
        = tree_map(lambda x: True,
        hybrid_hill_active_flags["plastic"]["effective stress"]["neural network"])
    #for key in hill_coefficients:
    #    hybrid_hill_active_flags["plastic"]["effective stress"]["hill"][key] = True

    hybrid_hill_transforms = hybrid_hill_values.copy()
    hybrid_hill_transforms = tree_map(lambda a: None, hybrid_hill_transforms)
    hybrid_hill_transforms["plastic"]["effective stress"]["neural network"] \
        = tree_map(lambda x: np.array([1.]),
        hybrid_hill_transforms["plastic"]["effective stress"]["neural network"],
        is_leaf=lambda x: x is None)
    #for key in hill_coefficients:
    #    hybrid_hill_transforms["plastic"]["effective stress"]["hill"][key] = \
    #        hill_coefficients_bounds

    hybrid_hill_parameters = \
        Parameters(hybrid_hill_values, hybrid_hill_active_flags,
        hybrid_hill_transforms)

    return hybrid_hill_parameters

def params_icnn_hybrid_hill_voce(p_elastic, p_hill, p_voce, icnn_params):
    E, nu = p_elastic
    Y, F, G, H, L, M, N = p_hill
    S, D = p_voce

    elastic_params = {"E": E, "nu": nu}
    initial_yield_params = {"Y": Y}
    voce_params = {"S": S, "D": D}
    hardening_params = {"voce": voce_params}

    hill_coefficients = {"F": F, "G": G, "H": H, "L": L, "M": M, "N": N}
    hill_effective_stress_params = {"hill": hill_coefficients}
    hill_coefficients_bounds = np.array([0.1, 3.])

    hybrid_hill_values = {
        "elastic": elastic_params,
        "plastic": {
            "effective stress": {
                "hill": hill_coefficients,
                "neural network": icnn_params},
            "flow stress": {
                "initial yield": initial_yield_params,
                "hardening": hardening_params}}}

    hybrid_hill_active_flags = hybrid_hill_values.copy()
    hybrid_hill_active_flags = tree_map(lambda a: False,
        hybrid_hill_active_flags)
    hybrid_hill_active_flags["plastic"]["effective stress"]["neural network"] \
        = tree_map(lambda x: True,
        hybrid_hill_active_flags["plastic"]["effective stress"]["neural network"])
    for key in hill_coefficients:
        hybrid_hill_active_flags["plastic"]["effective stress"]["hill"][key] = True

    hybrid_hill_transforms = hybrid_hill_values.copy()
    hybrid_hill_transforms = tree_map(lambda a: None, hybrid_hill_transforms)
    hybrid_hill_transforms["plastic"]["effective stress"]["neural network"] \
        ["z params"] = tree_map(lambda x: np.array([1.]),
        hybrid_hill_transforms["plastic"]["effective stress"]["neural network"]
        ["z params"],
        is_leaf=lambda x: x is None)
    for key in hill_coefficients:
        hybrid_hill_transforms["plastic"]["effective stress"]["hill"][key] = \
            hill_coefficients_bounds

    hybrid_hill_parameters = \
        Parameters(hybrid_hill_values, hybrid_hill_active_flags,
        hybrid_hill_transforms)

    return hybrid_hill_parameters
