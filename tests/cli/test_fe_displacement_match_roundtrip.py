"""End-to-end FEDisplacementMatch QoI round-trip on ``problem.type=fe``.

Generates synthetic data by running ``cmad primal`` at truth, then
verifies:

- ``cmad objective`` with ``qoi = fe_displacement_match`` and
  ``data_file = truth.exo`` at the SAME params -> ``J ≈ 0``.
- ``cmad objective`` with ``kappa`` perturbed -> ``J > 0``.
- ``.npy`` round-trip: read ``truth.exo``'s nodal ``"u"``, ``np.save``
  it, use that file as ``data_file`` at truth -> ``J ≈ 0``.

``cmad gradient`` is not exercised here: AD over the same ``J`` is
covered by ``tests/cli/test_gradient_fe_roundtrip.py``.
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
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType


def _write_hex_cube_mesh(path: Path) -> None:
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
    with ExodusWriter(str(path), mesh):
        pass


def _make_deck(
        mesh_path: Path,
        output_path: Path,
        kappa: float,
        mu: float,
        qoi: dict[str, Any] | None = None,
        exodus_filename: str | None = None,
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
                "materials": {
                    "all": {"elastic": {"kappa": kappa, "mu": mu}},
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
        "output": {"path": str(output_path)},
    }
    if exodus_filename is not None:
        deck["output"]["exodus filename"] = exodus_filename
    if qoi is not None:
        deck["qoi"] = qoi
    return deck


class TestFEDisplacementMatchRoundTrip(unittest.TestCase):
    KAPPA_TRUTH = 100.0
    MU_TRUTH = 50.0

    def _run_objective_and_read_J(
            self, deck: dict[str, Any], deck_path: Path,
    ) -> float:
        deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
        self.assertEqual(cmad_main(["objective", str(deck_path)]), 0)
        out_dir = Path(deck["output"]["path"])
        with (out_dir / "J.json").open("r") as f:
            return float(json.load(f)["J"])

    def test_J_zero_at_truth_positive_perturbed_and_npy_roundtrip(
            self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_path = tmp / "mesh.exo"
            _write_hex_cube_mesh(mesh_path)

            # cmad primal at truth -> truth.exo (nodal "u" by default).
            primal_out = tmp / "out_primal"
            primal_deck = _make_deck(
                mesh_path, primal_out,
                self.KAPPA_TRUTH, self.MU_TRUTH,
                exodus_filename="primal.exo",
            )
            primal_deck_path = tmp / "primal.yaml"
            primal_deck_path.write_text(
                yaml.safe_dump(primal_deck, sort_keys=False),
            )
            self.assertEqual(
                cmad_main(["primal", str(primal_deck_path)]), 0,
            )
            truth_exo = primal_out / "primal.exo"
            self.assertTrue(truth_exo.exists())

            # cmad objective, data=truth.exo, SAME params -> J ≈ 0.
            J_at_truth = self._run_objective_and_read_J(
                _make_deck(
                    mesh_path, tmp / "out_obj_truth",
                    self.KAPPA_TRUTH, self.MU_TRUTH,
                    qoi={
                        "name": "fe_displacement_match",
                        "data_file": str(truth_exo),
                    },
                ),
                tmp / "obj_truth.yaml",
            )
            self.assertAlmostEqual(J_at_truth, 0.0, places=8)

            # Perturbed kappa -> J > 0.
            J_perturbed = self._run_objective_and_read_J(
                _make_deck(
                    mesh_path, tmp / "out_obj_perturbed",
                    self.KAPPA_TRUTH * 1.5, self.MU_TRUTH,
                    qoi={
                        "name": "fe_displacement_match",
                        "data_file": str(truth_exo),
                    },
                ),
                tmp / "obj_perturbed.yaml",
            )
            self.assertGreater(J_perturbed, 1e-8)

            # .npy round-trip at truth -> J ≈ 0.
            u_data = read_results(
                truth_exo,
                nodal_field_specs=[FieldSpec("u", VarType.VECTOR)],
            ).nodal["u"]
            u_npy = tmp / "u_data.npy"
            np.save(u_npy, u_data)
            J_npy_at_truth = self._run_objective_and_read_J(
                _make_deck(
                    mesh_path, tmp / "out_obj_npy",
                    self.KAPPA_TRUTH, self.MU_TRUTH,
                    qoi={
                        "name": "fe_displacement_match",
                        "data_file": str(u_npy),
                    },
                ),
                tmp / "obj_npy.yaml",
            )
            self.assertAlmostEqual(J_npy_at_truth, 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
