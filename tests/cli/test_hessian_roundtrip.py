"""Cross-strategy ``cmad hessian`` round-trip.

Builds Calibration data from the truth model, then evaluates
``cmad hessian`` against a deck away from truth for each of the two
Hessian-producing strategies (``direct_adjoint``, ``jvp``), and asserts
J, grad, and Hessian agree. Matches
:mod:`tests.objectives.test_jvp_vs_original` at the CLI level.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.cli.main import main as cmad_main
from cmad.models.deformation_types import DefType
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.solver.nonlinear_solver import newton_solve
from tests.support.test_problems import J2AnalyticalProblem


def _truth_cauchy(F: np.ndarray) -> np.ndarray:
    problem = J2AnalyticalProblem(scale_params=True)
    model = SmallElasticPlastic(problem.J2_parameters, DefType.FULL_3D)
    num_steps = F.shape[2] - 1
    cauchy = np.zeros((3, 3, num_steps + 1))
    model.set_xi_to_init_vals()
    for step in range(1, num_steps + 1):
        model.gather_global([F[:, :, step]], [F[:, :, step - 1]])
        newton_solve(model)
        model.advance_xi()
        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
    return cauchy


def _deck_with_sensitivity(stype: str, out_subdir: str) -> dict[str, Any]:
    return {
        "problem": {"type": "material_point"},
        "model": {
            "name": "small_elastic_plastic",
            "def_type": "full_3d",
            "effective_stress": "J2",
        },
        "parameters": {
            "rotation matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "elastic": {"E": 200_000.0, "nu": 0.3},
            "plastic": {
                "effective stress": {"J2": 0.0},
                "flow stress": {
                    "initial yield": {
                        "Y": {
                            "value": 220.0, "active": True,
                            "transform": {"log": 200.0},
                        },
                    },
                    "hardening": {
                        "voce": {
                            "S": {
                                "value": 220.0, "active": True,
                                "transform": {"log": 200.0},
                            },
                            "D": {
                                "value": 22.0, "active": True,
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
            "weight": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "sensitivity": {"type": stype},
        "output": {"path": out_subdir},
    }


class TestHessianRoundTrip(unittest.TestCase):
    def test_direct_adjoint_vs_jvp_agreement(self) -> None:
        num_steps = 30
        max_alpha = 0.5

        stress_mask = np.zeros((3, 3))
        stress_mask[0, 0] = 1.0

        problem = J2AnalyticalProblem()
        _, strain_ref, _ = problem.analytical_solution(
            stress_mask, max_alpha, num_steps,
        )

        I = np.eye(3)
        F_history = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
        F_history[:, :, 1:] += strain_ref
        cauchy_data = _truth_cauchy(F_history)

        strategies = ["direct_adjoint", "jvp"]
        results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", F_history)
            np.save(tmp / "cauchy_data.npy", cauchy_data)

            for stype in strategies:
                deck = _deck_with_sensitivity(stype, f"out_{stype}")
                deck_path = tmp / f"deck_{stype}.yaml"
                deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

                exit_code = cmad_main(["hessian", str(deck_path)])
                self.assertEqual(exit_code, 0)

                out_dir = tmp / f"out_{stype}"
                with (out_dir / "J.json").open("r") as f:
                    J_val = json.load(f)["J"]
                grad = np.load(out_dir / "grad.npy")
                hess = np.load(out_dir / "hess.npy")
                results[stype] = (J_val, grad, hess)

        ref_J, ref_grad, ref_hess = results["direct_adjoint"]
        jvp_J, jvp_grad, jvp_hess = results["jvp"]
        np.testing.assert_allclose(jvp_J, ref_J, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(jvp_grad, ref_grad, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(jvp_hess, ref_hess, rtol=1e-7, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
