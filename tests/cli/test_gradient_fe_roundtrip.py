"""End-to-end ``cmad gradient`` round-trip on ``problem.type=fe``.

Builds a small uniaxial-stretch FE deck with the
``fe_displacement_l2`` QoI and elastic kappa / mu marked active,
runs ``cmad gradient``, and verifies the wiring: ``grad.npy`` is a
length-2 finite array (one component per active param) and
``deck.resolved.yaml`` is written. ``J.json`` is not produced by
``cmad gradient`` — run ``cmad objective`` separately on the same
deck for J. AD-vs-FD numerical correctness is covered by
``tests/fem/test_fem_fd_checks.py``; this is a CLI-wiring check.
"""
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


def _make_fe_gradient_deck(mesh_filename: str) -> dict[str, Any]:
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
                "pin_x": ["equilibrium", 0, "xmin_sides", "0.0"],
                "pin_y": ["equilibrium", 1, "ymin_sides", "0.0"],
                "pin_z": ["equilibrium", 2, "zmin_sides", "0.0"],
                "ramp_x": ["equilibrium", 0, "xmax_sides", "0.05 * t"],
            },
        },
        "qoi": {"name": "fe_displacement_l2"},
        "output": {"path": "out", "format": "npy"},
    }


class TestGradientFeRoundTrip(unittest.TestCase):
    def test_writes_grad_and_resolved_deck(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_hex_cube_mesh(tmp / "mesh.exo")
            deck = _make_fe_gradient_deck(mesh_filename="mesh.exo")
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            self.assertEqual(
                cmad_main(["gradient", str(deck_path)]), 0,
            )

            out_dir = tmp / "out"
            grad = np.load(out_dir / "grad.npy")
            self.assertEqual(grad.shape, (2,))
            self.assertTrue(np.all(np.isfinite(grad)))
            # Poisson contraction makes |u|² depend on kappa / mu via
            # the equilibrium; at least one component must be non-zero.
            self.assertTrue(np.any(grad != 0.0))

            self.assertTrue((out_dir / "deck.resolved.yaml").exists())
            self.assertFalse((out_dir / "J.json").exists())
            self.assertFalse(any(out_dir.glob("*.exo")))


if __name__ == "__main__":
    unittest.main()
