"""End-to-end ``cmad objective`` round-trip on ``problem.type=fe``.

Builds a small uniaxial-stretch FE deck with the
``fe_displacement_l2`` QoI, runs ``cmad objective``, and verifies
the wiring: ``J.json`` contains a finite positive scalar,
``deck.resolved.yaml`` is written, and the deck-named Exodus file
exists. Numerical correctness of the QoI value is validated
directly against the closure in
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
    def test_writes_J_resolved_deck_and_exodus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_hex_cube_mesh(tmp / "mesh.exo")
            deck = _make_fe_objective_deck(
                mesh_filename="mesh.exo",
                output_section={
                    "path": "out",
                    "format": "exodus",
                    "exodus filename": "objective.exo",
                    "nodal fields": [
                        {"name": "displacement", "var_type": "vector"},
                    ],
                    "element fields by block": {
                        "all": [
                            {"name": "cauchy", "var_type": "sym_tensor"},
                        ],
                    },
                },
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
            self.assertTrue((out_dir / "objective.exo").exists())

    def test_skips_exodus_for_non_exodus_format(self) -> None:
        # When format != exodus, J.json + resolved deck still written;
        # FE Exodus output is skipped. exodus filename schema field
        # is only required when format == exodus.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_hex_cube_mesh(tmp / "mesh.exo")
            deck = _make_fe_objective_deck(
                mesh_filename="mesh.exo",
                output_section={"path": "out", "format": "npy"},
            )
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            self.assertEqual(
                cmad_main(["objective", str(deck_path)]), 0,
            )

            out_dir = tmp / "out"
            with (out_dir / "J.json").open("r") as f:
                J = json.load(f)["J"]
            self.assertGreater(J, 0.0)
            self.assertTrue((out_dir / "deck.resolved.yaml").exists())
            self.assertFalse((out_dir / "objective.exo").exists())


if __name__ == "__main__":
    unittest.main()
