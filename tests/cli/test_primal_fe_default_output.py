"""``cmad primal --fe`` default-output behavior.

When the deck omits ``output.{nodal fields, element fields by block}``
the writer falls back to the GR's ``default_output_fields()`` surface,
replicated across every mesh element block. For
``SmallDispEquilibrium`` the surface is ``displacement`` (nodal) +
``cauchy`` (per-block element).
"""
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from cmad.cli.main import main as cmad_main
from cmad.fem.mesh import StructuredHexMesh
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType


def _write_hex_cube_mesh(path: Path) -> None:
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
    with ExodusWriter(str(path), mesh):
        pass


def _make_fe_primal_deck_no_output_specs(
        mesh_filename: str,
) -> dict[str, Any]:
    """Schema-valid FE deck with output's field-spec keys omitted."""
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
                "ramp_x": [
                    "displacement", 0, "xmax_sides", "0.05 * t",
                ],
            },
        },
        "output": {"path": "out", "format": "exodus"},
    }


class TestPrimalFeDefaultOutput(unittest.TestCase):
    def test_omitted_specs_fall_back_to_gr_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_hex_cube_mesh(tmp / "mesh.exo")
            deck = _make_fe_primal_deck_no_output_specs(
                mesh_filename="mesh.exo",
            )
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            self.assertEqual(
                cmad_main(["primal", str(deck_path)]), 0,
            )

            # GR-default surface for SmallDispEquilibrium:
            #   nodal=[displacement (vector)],
            #   element=[cauchy (sym_tensor)].
            # Successful read against these specs proves the fallback
            # fired and the writer produced the right variables.
            results = read_results(
                tmp / "out" / "primal.exo",
                nodal_field_specs=[
                    FieldSpec("displacement", VarType.VECTOR),
                ],
                element_field_specs={
                    "all": [FieldSpec("cauchy", VarType.SYM_TENSOR)],
                },
            )
            self.assertEqual(results.time.shape, (6,))
            # 1-element hex cube -> 8 corner nodes x 3 components.
            self.assertEqual(
                results.nodal["displacement"][-1].shape, (8, 3),
            )
            self.assertEqual(
                results.element["all"]["cauchy"][-1].shape, (1, 6),
            )


if __name__ == "__main__":
    unittest.main()
