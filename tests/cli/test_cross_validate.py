"""Leave-one-out cross validation over three small elastic cubes.

Each cube is stretched by its own displacement ramp and has its own truth
field from ``cmad primal``. Every fold calibrates on two cubes and scores
the third; the held out values equal an independent evaluation of that
cube at the fold's parameters, the score is their mean, and the
``cross_validate`` subcommand writes the summary and the fold outputs.
"""
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.calibration import build_objective, cross_validate, summarize
from cmad.cli.common import load_fe_input
from cmad.cli.main import main as cmad_main
from cmad.fem.mesh import StructuredHexMesh
from cmad.io.exodus import ExodusWriter

_CUBES = {
    "a": ((2, 2, 2), "0.04 * t"),
    "b": ((3, 2, 2), "0.05 * t"),
    "c": ((2, 3, 2), "0.06 * t"),
}
_START = {
    "kappa": {"value": 150.0, "active": True, "transform": {"log": 100.0}},
    "mu": {"value": 40.0, "active": True, "transform": {"log": 50.0}},
}


def _shared(elastic: dict[str, Any], out: Path) -> dict[str, Any]:
    return {
        "problem": {"type": "fe"},
        "residuals": {
            "global residual": {"type": "mechanics", "def_type": "full_3d"},
            "local residual": {
                "type": "elastic",
                "materials": {"all": {"elastic": elastic}},
            },
        },
        "optimizer": {
            "algorithm": "L-BFGS-B",
            "options": {"maxiter": 2},
        },
        "output": {"path": str(out)},
    }


def _specimen(mesh: Path, ramp: str, truth: Path | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "discretization": {
            "mesh file": str(mesh), "num steps": 3, "step size": 0.25,
        },
        "dirichlet bcs": {
            "expression": {
                "fix_x": ["equilibrium", 0, "xmin_sides", "0.0"],
                "fix_y": ["equilibrium", 1, "ymin_sides", "0.0"],
                "fix_z": ["equilibrium", 2, "zmin_sides", "0.0"],
                "ramp_x": ["equilibrium", 0, "xmax_sides", ramp],
            },
        },
    }
    if truth is not None:
        entry["qoi"] = {
            "name": "fe_displacement_match", "data_file": str(truth),
        }
    return entry


def _write(path: Path, deck: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(deck, sort_keys=False))
    return path


class TestCrossValidate(unittest.TestCase):

    def test_folds_score_the_held_out_cube(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            entries: dict[str, dict[str, Any]] = {}
            for tag, (cells, ramp) in _CUBES.items():
                mesh = tmp / f"{tag}.exo"
                with ExodusWriter(str(mesh), StructuredHexMesh((1.0, 1.0, 1.0), cells)):
                    pass
                primal_out = tmp / f"primal_{tag}"
                primal_deck = {
                    **_shared({"kappa": 100.0, "mu": 50.0}, primal_out),
                    **_specimen(mesh, ramp, None),
                }
                del primal_deck["optimizer"]
                primal_deck["output"]["exodus filename"] = "truth.exo"
                primal = _write(tmp / f"primal_{tag}.yaml", primal_deck)
                self.assertEqual(cmad_main(["primal", str(primal)]), 0)
                entries[tag] = _specimen(mesh, ramp, primal_out / "truth.exo")

            out = tmp / "out"
            joint = _write(tmp / "joint.yaml", {
                **_shared(_START, out), "specimens": entries,
            })
            self.assertEqual(cmad_main(["cross_validate", str(joint)]), 0)

            cv_dir = out / "cross_validation"
            with (cv_dir / "summary.yaml").open() as f:
                summary = yaml.safe_load(f)
            self.assertEqual(
                list(summary),
                ["held out a", "held out b", "held out c", "cv score"],
            )
            for tag in _CUBES:
                fold_dir = cv_dir / f"held_out_{tag}"
                for name in (
                    "opt_history.json", "opt_params.yaml",
                    "active_params.json", "opt_status.json",
                ):
                    self.assertTrue((fold_dir / name).exists(), name)
                entry = summary[f"held out {tag}"]
                self.assertEqual(
                    entry["trained on"], [t for t in _CUBES if t != tag],
                )
                self.assertEqual(
                    set(entry["training"]), set(entry["trained on"]),
                )

            # Through the API: the held out values against independent
            # evaluations at the folds' parameters.
            resolved = load_fe_input(joint, "cross_validate")
            folds = cross_validate(resolved)
            held_out = []
            for fold in folds:
                single = build_objective(load_fe_input(_write(
                    tmp / f"single_{fold.held_out}.yaml",
                    {**_shared(_START, out), **entries[fold.held_out]},
                ), "calibrate"))
                single.value(fold.result.x)
                independent = single.history[-1]["accumulated_qois"]
                assert fold.held_out_qois is not None
                self.assertEqual(
                    list(fold.held_out_qois), ["fe_displacement_match"],
                )
                np.testing.assert_allclose(
                    fold.held_out_qois["fe_displacement_match"],
                    independent["fe_displacement_match"], rtol=1e-13,
                )
                self.assertFalse(np.array_equal(fold.result.x, single.x0))
                held_out.append(fold.held_out_qois["fe_displacement_match"])
            score = summarize(folds)["cv score"]
            self.assertEqual(score["folds"], 3)
            self.assertNotIn("failed", score)
            np.testing.assert_allclose(
                score["fe_displacement_match"], np.mean(held_out), rtol=1e-14,
            )
            print("held out displacement mismatches:", held_out)

            resolved["cross validation"] = {"hold out": ["b"]}
            (one_fold,) = cross_validate(resolved)
            self.assertEqual(one_fold.held_out, "b")
            self.assertEqual(one_fold.trained_on, ["a", "c"])


if __name__ == "__main__":
    unittest.main()
