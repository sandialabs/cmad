"""End-to-end ``cmad hessian`` round-trip on ``problem.type=fe``.

Builds a small uniaxial-stretch FE deck with the
``fe_displacement_l2`` QoI and elastic kappa / mu marked active,
runs ``cmad hessian``, and verifies the wiring: ``J.json``
contains a finite positive scalar; ``hess.npy`` is a 2x2 finite
symmetric array; ``deck.resolved.yaml`` is written; ``grad.npy``
is NOT written (run ``cmad gradient`` separately for grad output).
AD-vs-FD numerical correctness is covered by
``tests/fem/test_fem_fd_checks.py``; this is a CLI-wiring check.
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


def _write_hex_cube_mesh(
        path: Path,
        divisions: tuple[int, int, int] = (1, 1, 1),
) -> None:
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), divisions)
    with ExodusWriter(str(path), mesh):
        pass


def _make_fe_hessian_deck(mesh_filename: str) -> dict[str, Any]:
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": mesh_filename,
            "num steps": 5,
            "step size": 0.2,
        },
        "residuals": {
            "global residual": {"type": "small_disp_equilibrium"},
            "local residual": {
                "type": "elastic",
                "def_type": "full_3d",
                "materials": {
                    "all": {
                        "elastic": {
                            "kappa": {"value": 100.0, "active": True},
                            "mu": {"value": 50.0, "active": True},
                        },
                    },
                },
            },
        },
        "dirichlet bcs": {
            "expression": {
                "pin_x": ["displacement", 0, "xmin_sides", "0.0"],
                "pin_y": ["displacement", 1, "ymin_sides", "0.0"],
                "pin_z": ["displacement", 2, "zmin_sides", "0.0"],
                "ramp_x": ["displacement", 0, "xmax_sides", "0.05 * t"],
            },
        },
        "qoi": {"name": "fe_displacement_l2"},
        "output": {"path": "out", "format": "npy"},
    }


class TestHessianFeRoundTrip(unittest.TestCase):
    def test_writes_J_hess_and_resolved_deck(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_hex_cube_mesh(tmp / "mesh.exo")
            deck = _make_fe_hessian_deck(mesh_filename="mesh.exo")
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            self.assertEqual(
                cmad_main(["hessian", str(deck_path)]), 0,
            )

            out_dir = tmp / "out"
            with (out_dir / "J.json").open("r") as f:
                J = json.load(f)["J"]
            self.assertTrue(np.isfinite(J))
            self.assertGreater(J, 0.0)

            hess = np.load(out_dir / "hess.npy")
            self.assertEqual(hess.shape, (2, 2))
            self.assertTrue(np.all(np.isfinite(hess)))
            np.testing.assert_allclose(hess, hess.T, rtol=0, atol=1e-12)

            self.assertTrue((out_dir / "deck.resolved.yaml").exists())
            self.assertFalse((out_dir / "primal.exo").exists())
            self.assertFalse((out_dir / "grad.npy").exists())


if __name__ == "__main__":
    unittest.main()
