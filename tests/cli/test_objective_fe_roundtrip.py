"""End-to-end ``cmad objective`` round-trip on ``problem.type=fe``.

Builds a small uniaxial-stretch FE deck with the
``fe_displacement_l2`` QoI, runs ``cmad objective``, and verifies
the wiring: ``J.json`` contains a finite positive scalar and
``deck.resolved.yaml`` is written. FE state-trajectory output is
not produced by ``cmad objective``; only ``cmad primal`` writes
Exodus II files. Numerical correctness of the QoI value is
validated directly against the closure in
``tests/qois/test_fe_displacement_l2.py``; this is a CLI-wiring
check.
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


def _make_fe_objective_deck(
        mesh_filename: str, output_section: dict[str, Any],
) -> dict[str, Any]:
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
                    "all": {"elastic": {"kappa": 100.0, "mu": 50.0}},
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
        "output": output_section,
    }


class TestObjectiveFeRoundTrip(unittest.TestCase):
    def test_writes_J_and_resolved_deck(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_hex_cube_mesh(tmp / "mesh.exo")
            deck = _make_fe_objective_deck(
                mesh_filename="mesh.exo",
                output_section={"path": "out"},
            )
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            self.assertEqual(
                cmad_main(["objective", str(deck_path)]), 0,
            )

            out_dir = tmp / "out"
            with (out_dir / "J.json").open("r") as f:
                J = json.load(f)["J"]
            self.assertTrue(np.isfinite(J))
            self.assertGreater(J, 0.0)
            self.assertTrue((out_dir / "deck.resolved.yaml").exists())
            # No Exodus II output from cmad objective; the writer is
            # only called by cmad primal.
            self.assertFalse(any(out_dir.glob("*.exo")))


if __name__ == "__main__":
    unittest.main()
