"""``cmad primal`` round trip on a heat transfer problem: a cube held at
two temperatures on opposite faces, whose linear profile a P1 mesh
reproduces to round off, read back from the Exodus file."""
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.cli.common import build_fe_problem_from_deck
from cmad.cli.main import main as cmad_main
from cmad.fem.mesh import StructuredHexMesh
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType


def _make_deck(mesh_filename: str, out_dir: str) -> dict[str, Any]:
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": mesh_filename,
            "num steps": 1,
            "step size": 1.0,
        },
        "residuals": {
            "global residual": {"type": "heat_transfer"},
            "local residual": {
                "type": "conduction",
                "materials": {
                    "all": {"thermal": {"conductivity": 16.0}},
                },
            },
        },
        "dirichlet bcs": {
            "expression": {
                "hot": ["energy balance", 0, "xmin_sides", "400.0"],
                "cold": ["energy balance", 0, "xmax_sides", "300.0"],
            },
        },
        "output": {
            "path": out_dir,
            "exodus filename": "primal.exo",
            "global residual": ["T"],
        },
    }


class TestPrimalFeHeatRoundTrip(unittest.TestCase):
    def test_linear_profile_between_two_faces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
            with ExodusWriter(str(tmp / "mesh.exo"), mesh):
                pass
            deck = _make_deck(str(tmp / "mesh.exo"), str(tmp / "out"))
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            self.assertEqual(cmad_main(["primal", str(deck_path)]), 0)

            results = read_results(
                tmp / "out" / "primal.exo",
                nodal_field_specs=[FieldSpec("T", VarType.SCALAR)],
            )
            nodes = build_fe_problem_from_deck(
                deck_path, "primal",
            ).fe_problem.mesh.nodes
            T = results.nodal["T"][-1].reshape(-1)
            np.testing.assert_allclose(T, 400.0 - 100.0 * nodes[:, 0], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
