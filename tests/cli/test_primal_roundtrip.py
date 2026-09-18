"""End-to-end ``cmad primal`` round-trip against J2AnalyticalProblem, and
against the scalar return map for a rate-dependent material whose input
file gives non-uniform step times.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from cmad.cli.main import main as cmad_main
from tests.models.test_rate_dependent_uniaxial import (
    JOHNSON_COOK,
    NUM_STEPS,
    PEAK_STRAIN,
    STRAIN_RATES,
    johnson_cook_flow_stress,
    return_map,
    stress_bound,
)
from tests.support.test_problems import J2AnalyticalProblem


def two_rate_schedule():
    """Uniform strain steps, the first half at the slow rate and the
    second at the fast one, so every step has its own ``dt``.
    """
    strains = np.linspace(0.0, PEAK_STRAIN, NUM_STEPS + 1)
    rates = np.where(
        np.arange(NUM_STEPS) < NUM_STEPS // 2, STRAIN_RATES[0], STRAIN_RATES[1])
    times = np.concatenate([[0.0], np.cumsum(np.diff(strains) / rates)])
    return strains, times


def johnson_cook_deck(tmp: Path, times) -> dict:
    """A uniaxial stress material point input file on the Johnson-Cook
    material of the point references; ``times`` is the entry or ``None``.
    """
    deck = {
        "problem": {"type": "material_point"},
        "model": {
            "name": "small_elastic_plastic",
            "def_type": "uniaxial_stress",
            "effective_stress": "J2",
        },
        "parameters": {
            "rotation matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "elastic": {"E": 200_000.0, "nu": 0.3},
            "plastic": {
                "effective stress": {"J2": 0.0},
                "flow stress": JOHNSON_COOK,
            },
        },
        "deformation": {"history_file": str(tmp / "F.npy")},
        "output": {"path": str(tmp / "out")},
    }
    if times is not None:
        deck["deformation"]["times"] = [float(t) for t in times]
    return deck


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
                "deformation": {"history_file": str(tmp / "F.npy")},
                "output": {"path": str(tmp / "out")},
            }
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            exit_code = cmad_main(["primal", str(deck_path)])
            self.assertEqual(exit_code, 0)

            cauchy = np.load(tmp / "out" / "cauchy.npy")
            np.testing.assert_allclose(
                cauchy[:, :, 1:], stress_ref, rtol=1e-6, atol=stress_bound(),
            )

    def test_johnson_cook_two_rates_from_the_input_file(self) -> None:
        strains, times = two_rate_schedule()
        sigma_ref, _, _ = return_map(johnson_cook_flow_stress, strains, times)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", (1.0 + strains)[None, None, :])
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(
                yaml.safe_dump(johnson_cook_deck(tmp, times), sort_keys=False))

            self.assertEqual(cmad_main(["primal", str(deck_path)]), 0)

            cauchy = np.load(tmp / "out" / "cauchy.npy")
            self.assertLess(
                np.abs(cauchy[0, 0, :] - sigma_ref).max(), stress_bound())


if __name__ == "__main__":
    unittest.main()
