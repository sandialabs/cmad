"""End-to-end FE ``cmad calibrate`` recovery round-trip (small-disp elastic).

``cmad primal`` at truth (``kappa=100, mu=50``) under an axial traction on
``xmax_sides`` plus zero-normal-displacement DBCs on the three "min" faces
generates ``truth.exo``. ``cmad calibrate`` then starts from a perturbed,
log-transformed ``(kappa, mu)`` and matches that displacement field via the
``fe_displacement_match`` QoI.

The traction (not a displacement ramp) is what makes ``kappa`` and ``mu``
separately identifiable: under pure displacement control the field depends
only on ``nu``, leaving the absolute stiffness -- hence the ``(kappa, mu)``
scale -- unrecoverable. The asymmetric init (``kappa`` high, ``mu`` low)
forces both to move.

Assertions cover the three calibrate outputs: ``opt_status.json`` (converged,
``fun ~ 0``), ``opt_params.yaml`` (re-loadable per-block materials) and
``active_params.json`` (flat active-only table), plus the ``opt_history.json``
native-parameter trace.
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

_BCS: dict[str, Any] = {
    "dirichlet bcs": {
        "expression": {
            "fix_x": ["equilibrium", 0, "xmin_sides", "0.0"],
            "fix_y": ["equilibrium", 1, "ymin_sides", "0.0"],
            "fix_z": ["equilibrium", 2, "zmin_sides", "0.0"],
        },
    },
    "surface flux bcs": {
        "expression": {
            "pull_x": ["equilibrium", "xmax_sides", "10.0 * t", "0.0", "0.0"],
        },
    },
}


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
        **_BCS,
        "output": {"path": str(output_path)},
    }
    if exodus_filename is not None:
        deck["output"]["exodus filename"] = exodus_filename
    if qoi is not None:
        deck["qoi"] = qoi
    if optimizer is not None:
        deck["optimizer"] = optimizer
    return deck


class TestCalibrateFERoundTrip(unittest.TestCase):
    KAPPA_TRUTH = 100.0
    MU_TRUTH = 50.0

    def test_recovers_kappa_and_mu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_path = tmp / "mesh.exo"
            _write_hex_cube_mesh(mesh_path)

            # Truth primal -> truth.exo (nodal "u" by default).
            primal_out = tmp / "out_primal"
            primal_deck = _make_deck(
                mesh_path, primal_out,
                {"kappa": self.KAPPA_TRUTH, "mu": self.MU_TRUTH},
                exodus_filename="truth.exo",
            )
            (tmp / "primal.yaml").write_text(
                yaml.safe_dump(primal_deck, sort_keys=False),
            )
            self.assertEqual(cmad_main(["primal", str(tmp / "primal.yaml")]), 0)
            truth_exo = primal_out / "truth.exo"
            self.assertTrue(truth_exo.exists())

            # Calibrate from a perturbed, log-transformed (kappa, mu).
            cal_out = tmp / "out_cal"
            cal_deck = _make_deck(
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
                    "name": "fe_displacement_match",
                    "data_file": str(truth_exo),
                },
                optimizer={
                    "algorithm": "L-BFGS-B",
                    "log_params": True,
                    "options": {"ftol": 1e-14, "gtol": 1e-10, "maxiter": 200},
                },
            )
            (tmp / "cal.yaml").write_text(
                yaml.safe_dump(cal_deck, sort_keys=False),
            )
            self.assertEqual(cmad_main(["calibrate", str(tmp / "cal.yaml")]), 0)

            with (cal_out / "opt_status.json").open() as f:
                status = json.load(f)
            self.assertTrue(status["success"])
            self.assertLess(status["fun"], 1e-10)

            # opt_params.yaml: re-loadable per-block materials, envelopes kept.
            with (cal_out / "opt_params.yaml").open() as f:
                opt_params = yaml.safe_load(f)
            elastic = opt_params["materials"]["all"]["elastic"]
            np.testing.assert_allclose(
                elastic["kappa"]["value"], self.KAPPA_TRUTH, rtol=1e-3,
            )
            np.testing.assert_allclose(
                elastic["mu"]["value"], self.MU_TRUTH, rtol=1e-3,
            )
            self.assertTrue(elastic["kappa"]["active"])

            # active_params.json: flat active-only table.
            with (cal_out / "active_params.json").open() as f:
                active = json.load(f)
            self.assertEqual(set(active), {"all.elastic.kappa", "all.elastic.mu"})
            np.testing.assert_allclose(
                active["all.elastic.kappa"], self.KAPPA_TRUTH, rtol=1e-3,
            )
            np.testing.assert_allclose(
                active["all.elastic.mu"], self.MU_TRUTH, rtol=1e-3,
            )

            # opt_history.json: native-parameter trace + labels.
            with (cal_out / "opt_history.json").open() as f:
                hist = json.load(f)
            self.assertEqual(
                hist["active_param_paths"],
                ["all.elastic.kappa", "all.elastic.mu"],
            )
            self.assertEqual(len(hist["history"][0]["params"]), 2)
            self.assertLess(
                min(h["grad_norm"] for h in hist["history"]), 1e-5,
            )


if __name__ == "__main__":
    unittest.main()
