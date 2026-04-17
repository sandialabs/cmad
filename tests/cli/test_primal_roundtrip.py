"""End-to-end ``cmad primal`` round-trip against J2AnalyticalProblem."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from cmad.cli.main import main as cmad_main
from tests.support.test_problems import J2AnalyticalProblem


class TestPrimalRoundTrip(unittest.TestCase):
    def test_j2_voce_uniaxial_stress(self) -> None:
        num_steps = 30
        max_alpha = 0.5

        stress_mask = np.zeros((3, 3))
        stress_mask[0, 0] = 1.0

        problem = J2AnalyticalProblem()
        stress_ref, strain_ref, _alpha_ref = problem.analytical_solution(
            stress_mask, max_alpha, num_steps,
        )

        I = np.eye(3)
        F_history = np.repeat(I[:, :, np.newaxis], num_steps + 1, axis=2)
        F_history[:, :, 1:] += strain_ref

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", F_history)

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
                "output": {"path": "out"},
            }
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            exit_code = cmad_main(["primal", str(deck_path)])
            self.assertEqual(exit_code, 0)

            cauchy = np.load(tmp / "out" / "cauchy.npy")
            np.testing.assert_allclose(
                cauchy[:, :, 1:], stress_ref, rtol=1e-6, atol=1e-8,
            )


if __name__ == "__main__":
    unittest.main()
