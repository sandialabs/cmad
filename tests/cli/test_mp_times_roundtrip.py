"""Deck-level checks for the material point time history.

A material point deck spells its times inside ``deformation``, the three
ways the FE ``discretization`` section spells its own. These cover the
deck surface end to end: the spellings agreeing with each other, the
times actually reaching a rate dependent residual, a calibration
recovering a viscosity through the CLI, and the refusal to run a rate
dependent model against a deck that names no times at all.
"""

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.cli.main import main as cmad_main

_NUM_STEPS = 12
_ETA = 5e3


def _F_history() -> np.ndarray:
    """Monotone, nearly isochoric uniaxial stretch, shape (3, 3, N)."""
    s = np.linspace(0.0, 1.0, _NUM_STEPS + 1)
    lambda_axial = 1.0 + 0.06 * s
    lambda_lateral = 1.0 / np.sqrt(lambda_axial)
    F = np.zeros((3, 3, _NUM_STEPS + 1))
    F[0, 0, :] = lambda_axial
    F[1, 1, :] = lambda_lateral
    F[2, 2, :] = lambda_lateral
    return F


def _deck(
        tmp: Path, deformation_extra: dict[str, Any],
        eta: float = _ETA, rate_dependent: bool = True,
) -> dict[str, Any]:
    flow_stress: dict[str, Any] = {
        "initial yield": {"Y": 200.0},
        "hardening": {"voce": {"S": 200.0, "D": 20.0}},
    }
    if rate_dependent:
        flow_stress["rate_dependence"] = {"perzyna": {"eta": eta}}

    return {
        "problem": {"type": "material_point"},
        "model": {
            "name": "be_bar_elastic_plastic",
            "def_type": "full_3d",
        },
        "parameters": {
            "elastic": {"E": 200_000.0, "nu": 0.3},
            "plastic": {
                "effective stress": {"J2": {}},
                "flow stress": flow_stress,
            },
        },
        "deformation": {
            "history_file": str(tmp / "F.npy"),
            **deformation_extra,
        },
        "output": {"path": str(tmp / "out")},
    }


def _run_primal(tmp: Path, deck: dict[str, Any]) -> np.ndarray:
    deck_path = tmp / "deck.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
    exit_code = cmad_main(["primal", str(deck_path)])
    assert exit_code == 0
    return np.load(tmp / "out" / "cauchy.npy")


class TestMPTimesRoundTrip(unittest.TestCase):

    def test_step_size_and_inline_times_agree(self) -> None:
        """``num steps`` + ``step size`` is the inline list it expands to."""
        dt = 0.05
        times = (np.arange(_NUM_STEPS + 1) * dt).tolist()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", _F_history())

            swept = _run_primal(tmp, _deck(
                tmp, {"num steps": _NUM_STEPS, "step size": dt},
            ))
            inline = _run_primal(tmp, _deck(tmp, {"times": times}))

        np.testing.assert_allclose(swept, inline, rtol=0.0, atol=1e-12)

    def test_times_file_agrees_with_inline(self) -> None:
        rng = np.random.default_rng(3)
        dts = 1.0 + rng.uniform(0.0, 1.5, _NUM_STEPS)
        times = np.concatenate([[0.0], np.cumsum(dts)]) * 1e-2

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", _F_history())
            np.savetxt(tmp / "t.txt", times)

            from_file = _run_primal(tmp, _deck(
                tmp, {"times file": str(tmp / "t.txt")},
            ))
            inline = _run_primal(tmp, _deck(
                tmp, {"times": times.tolist()},
            ))

        np.testing.assert_allclose(from_file, inline, rtol=0.0, atol=1e-12)

    def test_step_size_changes_a_rate_dependent_response(self) -> None:
        """The deck's step sizes reach the viscoplastic residual.

        Perzyna's overstress grows with the plastic strain rate, so the
        same stretch path driven ten times faster carries more stress.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", _F_history())

            fast = _run_primal(tmp, _deck(
                tmp, {"num steps": _NUM_STEPS, "step size": 1e-3},
            ))
            slow = _run_primal(tmp, _deck(
                tmp, {"num steps": _NUM_STEPS, "step size": 1e-2},
            ))

        self.assertGreater(fast[0, 0, -1], slow[0, 0, -1])

    def test_rate_independent_deck_ignores_the_times(self) -> None:
        """Without a rate law, the deck's times must not change the answer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", _F_history())

            no_times = _run_primal(tmp, _deck(
                tmp, {}, rate_dependent=False,
            ))
            with_times = _run_primal(tmp, _deck(
                tmp, {"num steps": _NUM_STEPS, "step size": 1e-3},
                rate_dependent=False,
            ))

        np.testing.assert_allclose(no_times, with_times, rtol=0.0, atol=1e-10)

    def test_rate_dependent_deck_without_times_is_refused(self) -> None:
        """A placeholder dt = 1 would silently absorb the real step size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", _F_history())

            deck_path = tmp / "deck.yaml"
            deck_path.write_text(
                yaml.safe_dump(_deck(tmp, {}), sort_keys=False),
            )
            with self.assertRaises(ValueError) as cm:
                cmad_main(["primal", str(deck_path)])

        message = str(cm.exception)
        self.assertIn("be_bar_elastic_plastic", message)
        self.assertIn("rate dependent", message)
        self.assertIn("times file", message)

    def test_two_time_sources_are_refused_by_the_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", _F_history())

            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(_deck(tmp, {
                "times": (np.arange(_NUM_STEPS + 1) * 0.05).tolist(),
                "num steps": _NUM_STEPS,
                "step size": 0.05,
            }), sort_keys=False))

            with self.assertRaises(ValueError) as cm:
                cmad_main(["primal", str(deck_path)])
        self.assertIn("deformation", str(cm.exception))


def _varying_rate_times() -> np.ndarray:
    """Times whose step sizes change sharply partway through the history.

    At a single constant strain rate the Perzyna overstress is very nearly
    a constant shift of the initial yield, so ``Y`` and ``eta`` are
    degenerate and neither is identifiable on its own. Driving the same
    stretch path at two different rates separates them: the shift moves
    with the rate, the yield does not.
    """
    fast, slow = 2e-4, 4e-3
    dts = np.r_[np.full(_NUM_STEPS // 2, fast),
                np.full(_NUM_STEPS - _NUM_STEPS // 2, slow)]
    return np.concatenate([[0.0], np.cumsum(dts)])


class TestMPSensitivityTypesOverRealTimes(unittest.TestCase):
    """All four ``sensitivity.type`` strategies, driven over varying dt.

    The three MPObjective strategies and the traced JVP one thread the
    deck's time history through entirely separately, so the CLI is where
    a divergence between them would actually bite. be_bar also pins the
    JVP driver's initial guess: its return map needs the elastic
    predictor to be viable, and a traced solve started from ``xi_prev``
    diverges to NaN rather than failing loudly.
    """

    def _driven_deck(self, tmp: Path, stype: str) -> dict[str, Any]:
        deck = _deck(tmp, {"times file": str(tmp / "t.npy")})
        flow_stress = deck["parameters"]["plastic"]["flow stress"]
        flow_stress["initial yield"] = {
            "Y": {"value": 180.0, "active": True},
        }
        flow_stress["rate_dependence"] = {
            "perzyna": {"eta": {"value": 1.3 * _ETA, "active": True}},
        }
        deck["qoi"] = {
            "name": "calibration",
            "data_file": str(tmp / "cauchy_data.npy"),
            "weight": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        }
        deck["sensitivity"] = {"type": stype}
        deck["output"] = {"path": str(tmp / f"out_{stype}")}
        return deck

    def test_all_four_strategies_agree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", _F_history())
            np.save(tmp / "t.npy", _varying_rate_times())

            cauchy = _run_primal(tmp, _deck(
                tmp, {"times file": str(tmp / "t.npy")},
            ))
            np.save(tmp / "cauchy_data.npy", cauchy)

            grads: dict[str, np.ndarray] = {}
            hessians: dict[str, np.ndarray] = {}
            for stype in ("adjoint", "direct", "jvp", "direct_adjoint"):
                deck_path = tmp / f"{stype}.yaml"
                deck_path.write_text(yaml.safe_dump(
                    self._driven_deck(tmp, stype), sort_keys=False,
                ))
                takes_hessian = stype in ("jvp", "direct_adjoint")
                subcommand = "hessian" if takes_hessian else "gradient"
                self.assertEqual(
                    cmad_main([subcommand, str(deck_path)]), 0,
                )
                out = tmp / f"out_{stype}"
                grads[stype] = np.load(out / "grad.npy")
                if takes_hessian:
                    hessians[stype] = np.load(out / "hess.npy")

        reference = grads["adjoint"]
        self.assertTrue(np.all(np.isfinite(reference)))
        for stype, grad in grads.items():
            self.assertTrue(
                np.all(np.isfinite(grad)), f"{stype} gradient is not finite",
            )
            np.testing.assert_allclose(
                grad, reference, rtol=1e-6, atol=1e-6,
                err_msg=f"{stype} disagrees with adjoint",
            )

        np.testing.assert_allclose(
            hessians["jvp"], hessians["direct_adjoint"],
            rtol=1e-4,
            atol=1e-4 * np.abs(hessians["direct_adjoint"]).max(),
        )


class TestMPRateDependentCalibrate(unittest.TestCase):
    """``cmad calibrate`` recovers a yield and a viscosity over real times."""

    def test_recovers_Y_and_eta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", _F_history())
            np.save(tmp / "t.npy", _varying_rate_times())
            times_section = {"times file": str(tmp / "t.npy")}

            cauchy = _run_primal(tmp, _deck(tmp, times_section))
            np.save(tmp / "cauchy_data.npy", cauchy)

            # Start well off the truth in both parameters.
            deck = _deck(tmp, times_section)
            flow_stress = deck["parameters"]["plastic"]["flow stress"]
            flow_stress["initial yield"] = {
                "Y": {
                    "value": 0.7 * 200.0, "active": True,
                    "transform": {"log": 200.0},
                },
            }
            flow_stress["rate_dependence"] = {
                "perzyna": {
                    "eta": {
                        "value": 2.5 * _ETA, "active": True,
                        "transform": {"log": _ETA},
                    },
                },
            }
            deck["qoi"] = {
                "name": "calibration",
                "data_file": str(tmp / "cauchy_data.npy"),
                "weight": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
            }
            deck["sensitivity"] = {"type": "adjoint"}
            deck["optimizer"] = {
                "algorithm": "L-BFGS-B",
                "options": {"ftol": 1e-15, "gtol": 1e-12, "maxiter": 400},
            }

            deck_path = tmp / "calibrate.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
            self.assertEqual(cmad_main(["calibrate", str(deck_path)]), 0)

            opt = yaml.safe_load(
                (tmp / "out" / "opt_params.yaml").read_text(),
            )["parameters"]

        flow_stress = opt["plastic"]["flow stress"]
        Y = float(flow_stress["initial yield"]["Y"]["value"])
        eta = float(flow_stress["rate_dependence"]["perzyna"]["eta"]["value"])
        self.assertAlmostEqual(Y / 200.0, 1.0, places=3)
        self.assertAlmostEqual(eta / _ETA, 1.0, places=3)

    def test_recovers_eta_alone(self) -> None:
        """A viscosity on its own, the whole deck's only active parameter.

        One active parameter makes the objectives' accumulated gradient
        row ``(1, 1)``, the shape that used to flatten to a 0-d array and
        take ``transform_grad`` down with it before the optimizer ever
        saw a step.
        """
        dt = 2e-3
        times_section = {"num steps": _NUM_STEPS, "step size": dt}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "F.npy", _F_history())

            cauchy = _run_primal(tmp, _deck(tmp, times_section))
            np.save(tmp / "cauchy_data.npy", cauchy)

            deck = _deck(tmp, times_section)
            deck["parameters"]["plastic"]["flow stress"]["rate_dependence"] = {
                "perzyna": {
                    "eta": {
                        "value": 3.0 * _ETA, "active": True,
                        "transform": {"log": _ETA},
                    },
                },
            }
            deck["qoi"] = {
                "name": "calibration",
                "data_file": str(tmp / "cauchy_data.npy"),
                "weight": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
            }
            deck["sensitivity"] = {"type": "adjoint"}
            deck["optimizer"] = {
                "algorithm": "L-BFGS-B",
                "options": {"ftol": 1e-15, "gtol": 1e-12, "maxiter": 300},
            }

            deck_path = tmp / "calibrate_eta.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
            self.assertEqual(cmad_main(["calibrate", str(deck_path)]), 0)

            opt = yaml.safe_load(
                (tmp / "out" / "opt_params.yaml").read_text(),
            )["parameters"]

        eta = float(
            opt["plastic"]["flow stress"]
            ["rate_dependence"]["perzyna"]["eta"]["value"],
        )
        self.assertAlmostEqual(eta / _ETA, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
