import numpy as np

from jax import tree_map

from cmad.parameters.parameters import Parameters
from cmad.verification.functions import J2_yield, J2_yield_normal
from cmad.verification.solutions import compute_plastic_fields


def params_J2_voce(flat_param_values):

    E, nu, Y, S, D = flat_param_values

    flat_J2_equivalent_hill_coefficients = 0.5 * np.ones(6)
    F, G, H, L, M, N = flat_J2_equivalent_hill_coefficients

    J2_equivalent_hosford_coefficient = 4.

    elastic_params = {"E": E, "nu": nu}
    J2_effective_stress_params = {"J2": 0.}
    initial_yield_params = {"Y": Y}
    voce_params = {"S": S, "D": D}
    hardening_params = {"voce": voce_params}

    Y_log_scale = np.array([200.])
    S_bounds = np.array([100., 300.])
    D_bounds = np.array([10., 30.])

    J2_values = {
        "rotation matrix": np.eye(3),
        "elastic": elastic_params,
        "plastic": {
            "effective stress": J2_effective_stress_params,
            "flow stress": {
                "initial yield": initial_yield_params,
                "hardening": hardening_params}}}

    J2_active_flags = J2_values.copy()
    J2_active_flags = tree_map(lambda a: False, J2_active_flags)
    J2_active_flags["plastic"]["flow stress"] = tree_map(
        lambda x: True, J2_active_flags["plastic"]["flow stress"])

    J2_transforms = J2_values.copy()
    J2_transforms = tree_map(lambda a: None, J2_transforms)

    J2_flow_stress_transforms = J2_transforms["plastic"]["flow stress"]
    J2_flow_stress_transforms["initial yield"]["Y"] = Y_log_scale
    J2_flow_stress_transforms["hardening"]["voce"]["S"] = S_bounds
    J2_flow_stress_transforms["hardening"]["voce"]["D"] = D_bounds

    J2_parameters = \
        Parameters(J2_values, J2_active_flags, J2_transforms)

    hill_coefficients = {"F": F, "G": G, "H": H, "L": L, "M": M, "N": N}
    hill_effective_stress_params = {"hill": hill_coefficients}

    hill_values = {
        "rotation matrix": np.eye(3),
        "elastic": elastic_params,
        "plastic": {
            "effective stress": hill_effective_stress_params,
            "flow stress": {
                "initial yield": initial_yield_params,
                "hardening": hardening_params}}}

    hill_active_flags = hill_values.copy()
    hill_active_flags = tree_map(lambda a: False, hill_active_flags)
    hill_active_flags["plastic"]["effective stress"]["hill"]["Y"] = True
    hill_active_flags["plastic"]["flow stress"] = \
        tree_map(lambda x: True, hill_active_flags["plastic"]["flow stress"])

    hill_transforms = hill_values.copy()
    hill_transforms = tree_map(lambda a: None, hill_transforms)
    hill_flow_stress_transforms = hill_transforms["plastic"]["flow stress"]
    hill_flow_stress_transforms["initial yield"]["Y"] = Y_log_scale
    hill_flow_stress_transforms["hardening"]["voce"]["S"] = S_bounds
    hill_flow_stress_transforms["hardening"]["voce"]["D"] = D_bounds

    hill_parameters = \
        Parameters(hill_values, hill_active_flags, hill_transforms)

    hosford_coefficients = {"a": J2_equivalent_hosford_coefficient}
    hosford_effective_stress_params = {"hosford": hosford_coefficients}

    hosford_values = {
        "rotation matrix": np.eye(3),
        "elastic": elastic_params,
        "plastic": {
            "effective stress": hosford_effective_stress_params,
            "flow stress": {
                "initial yield": initial_yield_params,
                "hardening": hardening_params}}}

    hosford_active_flags = hosford_values.copy()
    hosford_active_flags = tree_map(lambda a: False, hosford_active_flags)
    hosford_active_flags["plastic"]["effective stress"]["hosford"]["Y"] = True
    hosford_active_flags["plastic"]["flow stress"] = \
        tree_map(lambda x: True,
                 hosford_active_flags["plastic"]["flow stress"])

    hosford_transforms = hosford_values.copy()
    hosford_transforms = tree_map(lambda a: None, hosford_transforms)
    hosford_flow_stress_transforms = \
        hosford_transforms["plastic"]["flow stress"]
    hosford_flow_stress_transforms["initial yield"]["Y"] = Y_log_scale
    hosford_flow_stress_transforms["hardening"]["voce"]["S"] = S_bounds
    hosford_flow_stress_transforms["hardening"]["voce"]["D"] = D_bounds

    hosford_parameters = \
        Parameters(hosford_values, hosford_active_flags, hosford_transforms)

    return J2_parameters, hill_parameters, hosford_parameters


def params_hyperelastic(flat_param_values):

    kappa, mu = flat_param_values
    elastic_params = {"kappa": kappa, "mu": mu}
    elastic_params_log_scale = np.array([1.])

    hyperelastic_values = {"elastic": elastic_params}

    hyperelastic_active_flags = hyperelastic_values.copy()
    hyperelastic_active_flags["elastic"] = tree_map(
        lambda x: True, hyperelastic_active_flags["elastic"])

    hyperelastic_transforms = hyperelastic_values.copy()
    hyperelastic_transforms = tree_map(lambda a: None, hyperelastic_transforms)
    hyperelastic_transforms["elastic"] = tree_map(
        lambda x: elastic_params_log_scale,
        hyperelastic_transforms["elastic"],
        is_leaf=lambda x: x is None)

    hyperelastic_parameters = \
        Parameters(hyperelastic_values, hyperelastic_active_flags,
                   hyperelastic_transforms)

    return hyperelastic_parameters


class J2AnalyticalProblem():
    """
    Effective stress: J2 or J2 equivalent Hill
    Hardening: Voce
    """

    def __init__(self):

        # E, nu, Y, S, D
        self._flat_param_values = np.array([200e3, 0.3, 200., 200., 20.])
        self.J2_parameters, self.hill_parameters, self.hosford_parameters = \
            params_J2_voce(self._flat_param_values)

    def analytical_solution(self, stress_mask, max_alpha, num_steps):

        stress, strain, alpha = \
            compute_plastic_fields(stress_mask, J2_yield, J2_yield_normal,
                                   self._flat_param_values, max_alpha,
                                   num_steps)

        return stress, strain, alpha
