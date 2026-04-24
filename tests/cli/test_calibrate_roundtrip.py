"""End-to-end ``cmad calibrate`` recovery round-trip (plane_stress).

Truth parameters generate cauchy data under a biaxial strain path matching
``examples/noisy_calibration.py`` (ramp on xx, then hold xx + ramp on yy,
50 steps each). A deck whose three active parameters (Y, S, D, each
log-transformed) start at 1.1x truth is handed to ``cmad calibrate`` with
L-BFGS-B and tight tolerances. Assertions check that the written
``opt_params.yaml`` / ``opt_status.json`` / ``opt_history.json`` reflect
recovery of truth.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from jax.tree_util import tree_map

from cmad.cli.main import main as cmad_main
from cmad.models.deformation_types import DefType, def_type_ndims
from cmad.models.global_fields import mp_U_from_F
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.parameters.parameters import Parameters
from cmad.solver.nonlinear_solver import newton_solve


def _truth_parameters() -> Parameters:
    """Build the truth parameter tree (matches noisy_calibration's setup).

    Three active parameters (Y, S, D) with log transforms; elastic
    constants and rotation matrix inactive.
    """
    values: Any = {
        "rotation matrix": np.eye(3),
        "elastic": {"E": 70e3, "nu": 0.3},
        "plastic": {
            "effective stress": {"J2": 0.0},
            "flow stress": {
                "initial yield": {"Y": 200.0},
                "hardening": {"voce": {"S": 200.0, "D": 20.0}},
            },
        },
    }
    active: Any = tree_map(lambda _: False, values)
    active["plastic"]["flow stress"] = tree_map(
        lambda _: True, active["plastic"]["flow stress"],
    )
    transforms: Any = tree_map(lambda _: None, values)
    flow = transforms["plastic"]["flow stress"]
    flow["initial yield"]["Y"] = np.array([200.0])
    flow["hardening"]["voce"]["S"] = np.array([200.0])
    flow["hardening"]["voce"]["D"] = np.array([20.0])
    return Parameters(values, active, transforms)


def _biaxial_F(num_pts: int = 50) -> np.ndarray:
    """Plane-stress F history: ramp xx, then hold xx + ramp yy.

    Returns shape ``(2, 2, 2 * num_pts + 1)``.
    """
    strain_increment = 0.02
    init_strain = strain_increment / num_pts
    eps_xx = np.r_[
        np.zeros(1),
        np.linspace(init_strain, strain_increment, num_pts),
        np.ones(num_pts) * strain_increment,
    ]
    eps_yy = np.r_[
        np.zeros(1),
        np.zeros(num_pts),
        np.linspace(init_strain, strain_increment, num_pts),
    ]
    ndims = def_type_ndims(DefType.PLANE_STRESS)
    num_steps = 2 * num_pts
    I = np.eye(ndims)
    F = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
    F[0, 0, :] += eps_xx
    F[1, 1, :] += eps_yy
    return F


def _truth_cauchy(F: np.ndarray) -> np.ndarray:
    """Forward-solve the plane_stress truth model; return ``(3, 3, N+1)``."""
    model = SmallElasticPlastic(_truth_parameters(), DefType.PLANE_STRESS)
    num_steps = F.shape[2] - 1
    cauchy = np.zeros((3, 3, num_steps + 1))
    model.set_xi_to_init_vals()
    for step in range(1, num_steps + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step]),
            mp_U_from_F(F[:, :, step - 1]),
        )
        model.seed_xi()
        newton_solve(model)
        model.advance_xi()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
    return cauchy


def _deck(init_factor: float) -> dict[str, Any]:
    Y = 200.0 * init_factor
    S = 200.0 * init_factor
    D = 20.0 * init_factor
    return {
        "problem": {"type": "material_point"},
        "model": {
            "name": "small_elastic_plastic",
            "def_type": "plane_stress",
            "effective_stress": "J2",
        },
        "parameters": {
            "rotation matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "elastic": {"E": 70_000.0, "nu": 0.3},
            "plastic": {
                "effective stress": {"J2": 0.0},
                "flow stress": {
                    "initial yield": {
                        "Y": {
                            "value": Y, "active": True,
                            "transform": {"log": 200.0},
                        },
                    },
                    "hardening": {
                        "voce": {
                            "S": {
                                "value": S, "active": True,
                                "transform": {"log": 200.0},
                            },
                            "D": {
                                "value": D, "active": True,
                                "transform": {"log": 20.0},
                            },
                        },
                    },
                },
            },
        },
        "deformation": {"history_file": "F.npy"},
        "qoi": {
            "name": "calibration",
            "data_file": "cauchy_data.npy",
            "weight": [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
        },
        "sensitivity": {"type": "adjoint"},
        "optimizer": {
            "algorithm": "L-BFGS-B",
            "options": {"ftol": 1e-14, "gtol": 1e-10, "maxiter": 200},
        },
        "output": {"path": "out"},
    }


class TestCalibrateRoundTrip(unittest.TestCase):
    def test_plane_stress_recovers_truth_params(self) -> None:
        F = _biaxial_F()
        cauchy_data = _truth_cauchy(F)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", F)
            np.save(tmp / "cauchy_data.npy", cauchy_data)

            deck = _deck(init_factor=1.1)
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            exit_code = cmad_main(["calibrate", str(deck_path)])
            self.assertEqual(exit_code, 0)

            out = tmp / "out"

            with (out / "opt_status.json").open("r") as f:
                status = json.load(f)
            self.assertTrue(status["success"])
            self.assertLess(status["fun"], 1e-10)

            with (out / "opt_params.yaml").open("r") as f:
                opt_params = yaml.safe_load(f)
            flow = opt_params["parameters"]["plastic"]["flow stress"]
            Y = flow["initial yield"]["Y"]["value"]
            S = flow["hardening"]["voce"]["S"]["value"]
            D = flow["hardening"]["voce"]["D"]["value"]
            np.testing.assert_allclose(Y, 200.0, rtol=1e-5)
            np.testing.assert_allclose(S, 200.0, rtol=1e-5)
            np.testing.assert_allclose(D, 20.0, rtol=1e-5)

            with (out / "opt_history.json").open("r") as f:
                hist = json.load(f)
            self.assertEqual(len(hist["active_param_paths"]), 3)
            self.assertGreater(len(hist["history"]), 0)
            self.assertLess(hist["history"][-1]["grad_norm"], 1e-6)


if __name__ == "__main__":
    unittest.main()
