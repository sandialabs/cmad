"""Tests for the ``initial conditions`` section: the expression values
reach the field's basis coefficients and the state the FE drivers start
from."""
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.cli.common import build_fe_problem_from_deck
from cmad.cli.main import main as cmad_main
from cmad.fem.dof import dof_physical_coords
from cmad.fem.mesh import StructuredHexMesh
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType


def _write_hex_cube_mesh(path: Path) -> None:
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
    with ExodusWriter(str(path), mesh):
        pass


def _make_deck(
        mesh_filename: str, out_dir: str, initial_conditions: Any,
) -> dict[str, Any]:
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": mesh_filename,
            "num steps": 2,
            "step size": 0.5,
        },
        "residuals": {
            "global residual": {"type": "mechanics", "def_type": "full_3d"},
            "local residual": {
                "type": "elastic",
                "materials": {
                    "all": {"elastic": {"kappa": 100.0, "mu": 50.0}},
                },
            },
        },
        "dirichlet bcs": {
            "expression": {
                "sym_x": ["equilibrium", 0, "xmin_sides", "0.0"],
                "sym_y": ["equilibrium", 1, "ymin_sides", "0.0"],
                "sym_z": ["equilibrium", 2, "zmin_sides", "0.0"],
                "load_x": ["equilibrium", 0, "xmax_sides", "0.05 * t"],
            },
        },
        "initial conditions": initial_conditions,
        "output": {
            "path": out_dir,
            "exodus filename": "primal.exo",
            "global residual": ["u"],
        },
    }


def _write_deck(tmp: Path, initial_conditions: Any) -> Path:
    _write_hex_cube_mesh(tmp / "mesh.exo")
    deck = _make_deck(
        str(tmp / "mesh.exo"), str(tmp / "out"), initial_conditions,
    )
    deck_path = tmp / "deck.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
    return deck_path


class TestInitialConditions(unittest.TestCase):
    def test_expression_sets_the_field_coefficients(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            deck_path = _write_deck(
                Path(tmpdir), {"u": ["0.1 * x", 0.0, 0.0]},
            )
            bundle = build_fe_problem_from_deck(deck_path, "primal")
            fe_problem = bundle.fe_problem
            coords, eq = dof_physical_coords(
                fe_problem.mesh, fe_problem.dof_map, "u",
            )
            U_init = bundle.U_init
            assert U_init is not None
            np.testing.assert_allclose(U_init[eq[:, 0]], 0.1 * coords[:, 0])
            np.testing.assert_array_equal(U_init[eq[:, 1:]], 0.0)

    def test_primal_starts_from_the_initial_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            deck_path = _write_deck(tmp, {"u": ["0.1 * x", 0.0, 0.0]})
            self.assertEqual(cmad_main(["primal", str(deck_path)]), 0)
            results = read_results(
                tmp / "out" / "primal.exo",
                nodal_field_specs=[FieldSpec("u", VarType.VECTOR)],
            )
            nodes = build_fe_problem_from_deck(
                deck_path, "primal",
            ).fe_problem.mesh.nodes
            u_initial = results.nodal["u"][0]
            np.testing.assert_allclose(
                u_initial[:, 0], 0.1 * nodes[:, 0], rtol=1e-6,
            )
            np.testing.assert_array_equal(u_initial[:, 1:], 0.0)

    def test_unknown_field_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            deck_path = _write_deck(Path(tmpdir), {"T": 293.0})
            with self.assertRaises(ValueError) as ctx:
                build_fe_problem_from_deck(deck_path, "primal")
            self.assertIn("initial conditions.T", str(ctx.exception))

    def test_wrong_component_count_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            deck_path = _write_deck(Path(tmpdir), {"u": [0.0, 0.0]})
            with self.assertRaises(ValueError) as ctx:
                build_fe_problem_from_deck(deck_path, "primal")
            self.assertIn("initial conditions.u", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
