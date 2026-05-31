"""End-to-end FE load-matching round-trip (small-disp elastic).

A displacement-controlled cube (``u_x`` ramp on ``xmax_sides``, zero-normal
DBCs on the min faces) is solved at truth; the ``fe_load_match`` QoI in write
mode emits the per-step net ``x``-reaction on ``xmax_sides`` to a CSV. That
CSV is then the load data for match mode: ``cmad objective`` is ``~0`` at
truth and positive when ``kappa`` is perturbed, and ``cmad calibrate``
recovers ``kappa`` from the load history with ``mu`` held fixed.
"""
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.cli.main import main as cmad_main
from cmad.fem.mesh import StructuredHexMesh
from cmad.io.exodus import ExodusWriter

_SIDESET = "xmax_sides"


def _write_hex_cube_mesh(path: Path) -> None:
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
    with ExodusWriter(str(path), mesh):
        pass


def _make_deck(
        mesh_path: Path,
        output_path: Path,
        elastic: dict[str, Any],
        *,
        qoi: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deck: dict[str, Any] = {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": str(mesh_path),
            "num steps": 5,
            "step size": 0.2,
        },
        "residuals": {
            "global residual": {
                "type": "small_disp_equilibrium",
                "def_type": "full_3d",
            },
            "local residual": {
                "type": "elastic",
                "materials": {"all": {"elastic": elastic}},
            },
        },
        "dirichlet bcs": {
            "expression": {
                "fix_x": ["equilibrium", 0, "xmin_sides", "0.0"],
                "fix_y": ["equilibrium", 1, "ymin_sides", "0.0"],
                "fix_z": ["equilibrium", 2, "zmin_sides", "0.0"],
                "ramp_x": ["equilibrium", 0, _SIDESET, "0.05 * t"],
            },
        },
        "output": {"path": str(output_path)},
    }
    if qoi is not None:
        deck["qoi"] = qoi
    if optimizer is not None:
        deck["optimizer"] = optimizer
    return deck


def _match_qoi(load_csv: Path) -> dict[str, Any]:
    return {
        "name": "fe_load_match",
        "data_file": str(load_csv),
        "sideset": _SIDESET,
        "components": [0],
    }


class TestLoadMatchRoundTrip(unittest.TestCase):
    KAPPA_TRUTH = 100.0
    MU_TRUTH = 50.0

    def _objective(
            self, tmp: Path, mesh_path: Path,
            elastic: dict[str, Any], load_csv: Path, tag: str,
    ) -> float:
        out = tmp / f"out_{tag}"
        deck = _make_deck(mesh_path, out, elastic, qoi=_match_qoi(load_csv))
        deck_path = tmp / f"{tag}.yaml"
        deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
        self.assertEqual(cmad_main(["objective", str(deck_path)]), 0)
        with (out / "J.json").open() as f:
            return float(json.load(f)["J"])

    def test_objective_and_calibrate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_path = tmp / "mesh.exo"
            _write_hex_cube_mesh(mesh_path)
            load_csv = tmp / "load.csv"

            # Truth solve, write mode -> the reaction series CSV.
            truth_deck_path = tmp / "truth.yaml"
            truth_deck_path.write_text(yaml.safe_dump(
                _make_deck(
                    mesh_path, tmp / "out_primal",
                    {"kappa": self.KAPPA_TRUTH, "mu": self.MU_TRUTH},
                    qoi={
                        "name": "fe_load_match",
                        "output_file": str(load_csv),
                        "sideset": _SIDESET,
                        "components": [0],
                    },
                ),
                sort_keys=False,
            ))
            self.assertEqual(cmad_main(["primal", str(truth_deck_path)]), 0)
            self.assertTrue(load_csv.exists())

            # Objective: ~0 at truth, positive when kappa is perturbed.
            J_truth = self._objective(
                tmp, mesh_path,
                {"kappa": self.KAPPA_TRUTH, "mu": self.MU_TRUTH},
                load_csv, "truth",
            )
            self.assertLess(J_truth, 1e-8)
            J_perturbed = self._objective(
                tmp, mesh_path,
                {"kappa": 1.5 * self.KAPPA_TRUTH, "mu": self.MU_TRUTH},
                load_csv, "perturbed",
            )
            self.assertGreater(J_perturbed, 1e-8)

            # Calibrate kappa (mu fixed) from the load history.
            cal_out = tmp / "out_cal"
            cal_deck_path = tmp / "cal.yaml"
            cal_deck_path.write_text(yaml.safe_dump(
                _make_deck(
                    mesh_path, cal_out,
                    {
                        "kappa": {
                            "value": 1.5 * self.KAPPA_TRUTH, "active": True,
                            "transform": {"log": self.KAPPA_TRUTH},
                        },
                        "mu": self.MU_TRUTH,
                    },
                    qoi=_match_qoi(load_csv),
                    optimizer={
                        "algorithm": "L-BFGS-B",
                        "options": {
                            "ftol": 1e-14, "gtol": 1e-10, "maxiter": 200,
                        },
                    },
                ),
                sort_keys=False,
            ))
            self.assertEqual(cmad_main(["calibrate", str(cal_deck_path)]), 0)

            with (cal_out / "opt_status.json").open() as f:
                self.assertTrue(json.load(f)["success"])
            with (cal_out / "active_params.json").open() as f:
                active = json.load(f)
            self.assertEqual(set(active), {"all.elastic.kappa"})
            np.testing.assert_allclose(
                active["all.elastic.kappa"], self.KAPPA_TRUTH, rtol=1e-3,
            )


if __name__ == "__main__":
    unittest.main()
