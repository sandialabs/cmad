"""End-to-end FE weighted-sum calibration (displacement + load).

A displacement-controlled cube is solved at truth; one ``cmad primal`` run
writes both ``truth.exo`` (the nodal displacement field) and, via a write-mode
``fe_load_match``, ``load.csv`` (the reaction series). ``cmad calibrate`` then
minimizes an ``fe_weighted_sum`` of a displacement match and a load match,
recovering both ``kappa`` and ``mu``.
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
        exodus_filename: str | None = None,
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
    if exodus_filename is not None:
        deck["output"]["exodus filename"] = exodus_filename
    if qoi is not None:
        deck["qoi"] = qoi
    if optimizer is not None:
        deck["optimizer"] = optimizer
    return deck


class TestWeightedSumRoundTrip(unittest.TestCase):
    KAPPA_TRUTH = 100.0
    MU_TRUTH = 50.0

    def test_recovers_kappa_and_mu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_path = tmp / "mesh.exo"
            _write_hex_cube_mesh(mesh_path)
            load_csv = tmp / "load.csv"
            primal_out = tmp / "out_primal"

            # One truth solve writes the displacement field (truth.exo) and
            # the reaction series (load.csv, write mode).
            truth_deck_path = tmp / "truth.yaml"
            truth_deck_path.write_text(yaml.safe_dump(
                _make_deck(
                    mesh_path, primal_out,
                    {"kappa": self.KAPPA_TRUTH, "mu": self.MU_TRUTH},
                    exodus_filename="truth.exo",
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
            truth_exo = primal_out / "truth.exo"
            self.assertTrue(truth_exo.exists())
            self.assertTrue(load_csv.exists())

            # Calibrate both moduli against the combined objective.
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
                        "mu": {
                            "value": 0.8 * self.MU_TRUTH, "active": True,
                            "transform": {"log": self.MU_TRUTH},
                        },
                    },
                    qoi={
                        "name": "fe_weighted_sum",
                        "terms": [
                            {
                                "name": "fe_displacement_match",
                                "data_file": str(truth_exo),
                            },
                            {
                                "name": "fe_load_match",
                                "data_file": str(load_csv),
                                "sideset": _SIDESET,
                                "components": [0],
                            },
                        ],
                    },
                    optimizer={
                        "algorithm": "L-BFGS-B",
                        "options": {
                            "ftol": 1e-15, "gtol": 1e-12, "maxiter": 500,
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
            self.assertEqual(
                set(active), {"all.elastic.kappa", "all.elastic.mu"},
            )
            np.testing.assert_allclose(
                active["all.elastic.kappa"], self.KAPPA_TRUTH, rtol=1e-2,
            )
            np.testing.assert_allclose(
                active["all.elastic.mu"], self.MU_TRUTH, rtol=1e-2,
            )


if __name__ == "__main__":
    unittest.main()
