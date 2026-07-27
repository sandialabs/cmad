"""End-to-end DIC ``cmad calibrate`` recovery round-trip (small-disp elastic).

``cmad primal`` at truth (``kappa=100, mu=50``) under an axial traction on
``xmax_sides`` samples the solved displacement at scattered measurement
points on that face and writes a synthetic DIC point cloud (no Exodus).
``cmad calibrate`` then GMLS-remaps that cloud back onto the sideset through
the ``fe_dic_match`` QoI and recovers ``(kappa, mu)`` from a perturbed,
log-transformed start.

The traction (not a displacement ramp) makes ``kappa`` and ``mu`` separately
identifiable. The displacement field is uniform-strain, hence linear in
space, which GMLS reproduces exactly -- so remapping the scattered cloud
onto the nodes returns the true nodal field and the recovered parameters go
back to truth. This exercises the synthetic-DIC ``primal`` output path
(point location in the sideset's quad facets, FE-field sampling) joined to
the DIC-match calibration path.
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
from cmad.io.point_cloud import PointCloud, write_point_cloud

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


def _write_measurement_points(path: Path, n_points: int = 300) -> None:
    """Scattered measurement coordinates on the x = 1 face (no field)."""
    rng = np.random.default_rng(0)
    yz = rng.uniform(0.0, 1.0, (n_points, 2))
    coords = np.column_stack([np.ones(n_points), yz])
    cloud = PointCloud(
        coords=coords, times=np.array([0.0]), fields={},
    )
    write_point_cloud(path, cloud, write_xdmf=False)


def _base_deck(mesh_path: Path, out_path: Path, elastic: dict[str, Any],
               ) -> dict[str, Any]:
    return {
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
        "output": {"path": str(out_path)},
    }


class TestCalibrateDICRoundTrip(unittest.TestCase):
    KAPPA_TRUTH = 100.0
    MU_TRUTH = 50.0

    def test_recovers_kappa_and_mu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_path = tmp / "mesh.exo"
            _write_hex_cube_mesh(mesh_path)
            points_file = tmp / "points.h5"
            _write_measurement_points(points_file)

            # Truth primal -> synthetic DIC cloud (no Exodus).
            primal_out = tmp / "out_primal"
            primal_deck = _base_deck(
                mesh_path, primal_out,
                {"kappa": self.KAPPA_TRUTH, "mu": self.MU_TRUTH},
            )
            primal_deck["output"]["write exodus"] = False
            primal_deck["output"]["dic cloud"] = {
                "points file": str(points_file),
                "sideset": "xmax_sides",
                "output file": "synthetic_dic.h5",
            }
            (tmp / "primal.yaml").write_text(
                yaml.safe_dump(primal_deck, sort_keys=False),
            )
            self.assertEqual(cmad_main(["primal", str(tmp / "primal.yaml")]), 0)
            dic_file = primal_out / "synthetic_dic.h5"
            self.assertTrue(dic_file.exists())

            # Calibrate from a perturbed, log-transformed (kappa, mu).
            cal_out = tmp / "out_cal"
            cal_deck = _base_deck(
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
            )
            cal_deck["qoi"] = {
                "name": "fe_dic_match",
                "dic_file": str(dic_file),
                "sideset": "xmax_sides",
            }
            cal_deck["optimizer"] = {
                "algorithm": "L-BFGS-B",
                "log_params": True,
                "options": {"ftol": 1e-14, "gtol": 1e-10, "maxiter": 200},
            }
            (tmp / "cal.yaml").write_text(
                yaml.safe_dump(cal_deck, sort_keys=False),
            )
            self.assertEqual(cmad_main(["calibrate", str(tmp / "cal.yaml")]), 0)

            with (cal_out / "opt_status.json").open() as f:
                status = json.load(f)
            self.assertTrue(status["success"])
            self.assertLess(status["fun"], 1e-8)

            with (cal_out / "active_params.json").open() as f:
                active = json.load(f)
            np.testing.assert_allclose(
                active["all.elastic.kappa"], self.KAPPA_TRUTH, rtol=1e-3,
            )
            np.testing.assert_allclose(
                active["all.elastic.mu"], self.MU_TRUTH, rtol=1e-3,
            )


if __name__ == "__main__":
    unittest.main()
