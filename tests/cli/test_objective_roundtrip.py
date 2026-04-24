"""End-to-end ``cmad objective`` round-trip: J == 0 at the truth parameters.

Generates Calibration data by forward-solving :class:`SmallElasticPlastic`
at the ``J2AnalyticalProblem`` truth parameters, then runs
``cmad objective`` against a deck whose parameters match those same
truth values. J read back from ``J.json`` should be zero to float
precision.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from cmad.cli.main import main as cmad_main
from cmad.models.deformation_types import DefType
from cmad.models.global_fields import mp_U_from_F
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.solver.nonlinear_solver import newton_solve
from tests.support.test_problems import J2AnalyticalProblem


def _truth_cauchy(F: np.ndarray) -> np.ndarray:
    """Forward-solve the truth model and return the ``(3, 3, N+1)`` cauchy path."""
    problem = J2AnalyticalProblem(scale_params=True)
    model = SmallElasticPlastic(problem.J2_parameters, DefType.FULL_3D)
    num_steps = F.shape[2] - 1
    cauchy = np.zeros((3, 3, num_steps + 1))
    model.set_xi_to_init_vals()
    for step in range(1, num_steps + 1):
        model.gather_global(
            mp_U_from_F(F[:, :, step]),
            mp_U_from_F(F[:, :, step - 1]),
        )
        newton_solve(model)
        model.advance_xi()
        model.seed_none()
        model.evaluate_cauchy()
        cauchy[:, :, step] = model.Sigma().copy()
    return cauchy


class TestObjectiveRoundTrip(unittest.TestCase):
    def test_j2_voce_full_3d_J_zero_at_truth(self) -> None:
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

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", F_history)
            np.save(tmp / "cauchy_data.npy", cauchy_data)

            deck = {
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
                            "initial yield": {"Y": 200.0},
                            "hardening": {"voce": {"S": 200.0, "D": 20.0}},
                        },
                    },
                },
                "deformation": {"history_file": "F.npy"},
                "qoi": {
                    "name": "calibration",
                    "data_file": "cauchy_data.npy",
                    "weight": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
                "output": {"path": "out"},
            }
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            exit_code = cmad_main(["objective", str(deck_path)])
            self.assertEqual(exit_code, 0)

            with (tmp / "out" / "J.json").open("r") as f:
                J_out = json.load(f)["J"]

            self.assertAlmostEqual(J_out, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
