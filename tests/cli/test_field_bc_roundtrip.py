"""Round trip for field Dirichlet BCs on ``problem.type=fe``.

Runs ``cmad primal`` with an expression driving the loaded face, then
reruns the same problem with that face driven by the recorded nodal
field, and requires the two displacement histories to agree.

That agreement is what verifies the vertex ordering: the field data is
laid out against :func:`cmad.fem.dof.sideset_basis_fns`, while the solver
scatters through the ordering ``build_dof_map`` resolves. Those are the
same walk today, and a divergence would scramble the boundary rather than
fail loudly, so it is checked against a solved field rather than by
inspection. Both accepted data sources are exercised, since the ``.npy``
and Exodus readers reach the array by different paths.

Gradient agreement is included because a field value that broke the
derivative path would still produce a finite gradient; only the
comparison against the expression path rules that out.
"""
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

KAPPA_TRUTH = 100.0
MU_TRUTH = 50.0
NUM_STEPS = 5
STEP_SIZE = 0.2
U_SPEC = [FieldSpec("u", VarType.VECTOR)]


def _write_cube(path: Path) -> StructuredHexMesh:
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
    with ExodusWriter(str(path), mesh):
        pass
    return mesh


def _deck(
        mesh_path: Path,
        out_dir: Path,
        exodus_filename: str,
        *,
        loaded_face: list[Any],
        kappa: float = KAPPA_TRUTH,
        qoi: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A cube held on the three min faces and pulled on ``xmax_sides``.

    ``loaded_face`` is the entry driving that face; it lands under
    ``expression`` or ``field`` according to its value slot.
    """
    is_field = str(loaded_face[3]).endswith((".npy", ".exo"))
    elastic: dict[str, Any] = (
        {
            "kappa": {"value": kappa, "active": True},
            "mu": {"value": MU_TRUTH, "active": False},
        }
        if qoi is not None
        else {"kappa": kappa, "mu": MU_TRUTH}
    )
    dbcs: dict[str, Any] = {
        "expression": {
            "sym_x": ["equilibrium", 0, "xmin_sides", "0.0"],
            "sym_y": ["equilibrium", 1, "ymin_sides", "0.0"],
            "sym_z": ["equilibrium", 2, "zmin_sides", "0.0"],
        },
    }
    if is_field:
        dbcs["field"] = {"load_x": loaded_face}
    else:
        dbcs["expression"]["load_x"] = loaded_face

    deck: dict[str, Any] = {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": str(mesh_path),
            "num steps": NUM_STEPS,
            "step size": STEP_SIZE,
        },
        "residuals": {
            "global residual": {"type": "mechanics", "def_type": "full_3d"},
            "local residual": {
                "type": "elastic",
                "materials": {"all": {"elastic": elastic}},
            },
        },
        "dirichlet bcs": dbcs,
        "output": {"path": str(out_dir), "exodus filename": exodus_filename},
    }
    if qoi is not None:
        deck["qoi"] = qoi
    return deck


class TestFieldBCRoundTrip(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.mesh_path = self.dir / "cube.exo"
        self.mesh = _write_cube(self.mesh_path)
        self.expression_face = ["equilibrium", 0, "xmax_sides", "0.05 * t"]
        self.truth_exo = self.dir / "truth.exo"
        self.truth_u = self._run_primal(self.expression_face, "truth.exo")
        self.npy_path = self.dir / "u.npy"
        np.save(self.npy_path, self.truth_u)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_primal(
            self, loaded_face: list[Any], exodus_filename: str,
    ) -> np.ndarray:
        deck = _deck(
            self.mesh_path, self.dir, exodus_filename,
            loaded_face=loaded_face,
        )
        path = self.dir / f"{Path(exodus_filename).stem}.yaml"
        path.write_text(yaml.safe_dump(deck, sort_keys=False))
        self.assertEqual(cmad_main(["primal", str(path)]), 0)
        results = read_results(
            self.dir / exodus_filename, nodal_field_specs=U_SPEC,
        )
        return np.asarray(results.nodal["u"])

    def _run_gradient(self, loaded_face: list[Any], stem: str) -> np.ndarray:
        deck = _deck(
            self.mesh_path, self.dir, f"{stem}.exo",
            loaded_face=loaded_face,
            kappa=120.0,  # away from truth, so the gradient is not zero
            qoi={
                "name": "fe_displacement_match",
                "data_file": str(self.truth_exo),
            },
        )
        path = self.dir / f"{stem}.yaml"
        path.write_text(yaml.safe_dump(deck, sort_keys=False))
        grad_path = self.dir / "grad.npy"
        grad_path.unlink(missing_ok=True)
        self.assertEqual(cmad_main(["gradient", str(path)]), 0)
        return np.asarray(np.load(grad_path))

    def test_primal_output_carries_one_frame_per_schedule_entry(self):
        # A field BC requires this alignment, so a cmad primal output is
        # usable as field data with no reindexing.
        self.assertEqual(
            self.truth_u.shape,
            (NUM_STEPS + 1, self.mesh.nodes.shape[0], 3),
        )

    def test_npy_field_reproduces_the_expression_solution(self):
        u = self._run_primal(
            ["equilibrium", 0, "xmax_sides", str(self.npy_path)], "npy.exo",
        )
        np.testing.assert_allclose(u, self.truth_u, atol=1e-12)

    def test_exodus_field_reproduces_the_expression_solution(self):
        u = self._run_primal(
            ["equilibrium", 0, "xmax_sides", str(self.truth_exo)], "exo.exo",
        )
        np.testing.assert_allclose(u, self.truth_u, atol=1e-12)

    def test_gradient_matches_the_expression_path(self):
        from_field = self._run_gradient(
            ["equilibrium", 0, "xmax_sides", str(self.npy_path)], "g_field",
        )
        from_expression = self._run_gradient(self.expression_face, "g_expr")
        self.assertTrue(np.all(np.isfinite(from_field)))
        np.testing.assert_allclose(
            from_field, from_expression, rtol=1e-9, atol=1e-14,
        )


class TestFieldBCValidation(unittest.TestCase):
    """Data that cannot line up with the mesh or schedule is rejected.

    These fail while the problem is built, so no solve runs.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.mesh_path = self.dir / "cube.exo"
        self.mesh = _write_cube(self.mesh_path)
        self.u = np.zeros(
            (NUM_STEPS + 1, self.mesh.nodes.shape[0], 3), dtype=float,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build(self, loaded_face: list[Any]) -> int:
        deck = _deck(
            self.mesh_path, self.dir, "out.exo", loaded_face=loaded_face,
        )
        path = self.dir / "deck.yaml"
        path.write_text(yaml.safe_dump(deck, sort_keys=False))
        return cmad_main(["primal", str(path)])

    def _save(self, name: str, array: np.ndarray) -> str:
        path = self.dir / name
        np.save(path, array)
        return str(path)

    def test_frame_count_mismatch_raises(self):
        data = self._save("short.npy", self.u[:3])
        with self.assertRaisesRegex(ValueError, "3 frames but the time"):
            self._build(["equilibrium", 0, "xmax_sides", data])

    def test_node_count_mismatch_raises(self):
        data = self._save("fewnodes.npy", self.u[:, :5, :])
        with self.assertRaisesRegex(ValueError, "covers 5 nodes"):
            self._build(["equilibrium", 0, "xmax_sides", data])

    def test_unknown_sideset_raises(self):
        data = self._save("u.npy", self.u)
        with self.assertRaisesRegex(KeyError, "unknown sideset"):
            self._build(["equilibrium", 0, "not_a_sideset", data])

    def test_component_out_of_range_raises(self):
        data = self._save("u.npy", self.u)
        with self.assertRaisesRegex(ValueError, "eq 7 out of range"):
            self._build(["equilibrium", 7, "xmax_sides", data])


if __name__ == "__main__":
    unittest.main()
