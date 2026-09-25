"""The objective of a two specimen input file against its parts.

Two small elastic cubes on different meshes under different tractions,
each with its own truth displacement field from ``cmad primal``. The
joint objective's value and gradient are the weighted sums of the two
single specimen objectives', its history entry carries each specimen's
own value, and its gradient agrees with a directional finite difference.
"""
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.calibration import Objective, build_objective
from cmad.cli.common import load_fe_input
from cmad.cli.main import main as cmad_main
from cmad.fem.mesh import StructuredHexMesh
from cmad.io.exodus import ExodusWriter

_PULL = {
    "a": ("xmax_sides", ["10.0 * t", "0.0", "0.0"]),
    "b": ("ymax_sides", ["0.0", "10.0 * t", "0.0"]),
}
_CELLS = {"a": (2, 2, 2), "b": (3, 2, 2)}
_WEIGHT_B = 2.0


def _shared(elastic: dict[str, Any], out: Path) -> dict[str, Any]:
    return {
        "problem": {"type": "fe"},
        "residuals": {
            "global residual": {"type": "mechanics", "def_type": "full_3d"},
            "local residual": {
                "type": "elastic",
                "materials": {"all": {"elastic": elastic}},
            },
        },
        "output": {"path": str(out)},
    }


def _specimen(tag: str, mesh: Path, truth: Path | None) -> dict[str, Any]:
    sideset, traction = _PULL[tag]
    entry: dict[str, Any] = {
        "discretization": {
            "mesh file": str(mesh), "num steps": 5, "step size": 0.2,
        },
        "dirichlet bcs": {
            "expression": {
                "fix_x": ["equilibrium", 0, "xmin_sides", "0.0"],
                "fix_y": ["equilibrium", 1, "ymin_sides", "0.0"],
                "fix_z": ["equilibrium", 2, "zmin_sides", "0.0"],
            },
        },
        "surface flux bcs": {
            "expression": {"pull": ["equilibrium", sideset, *traction]},
        },
    }
    if truth is not None:
        entry["qoi"] = {
            "name": "fe_displacement_match", "data_file": str(truth),
        }
    return entry


def _write(path: Path, deck: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(deck, sort_keys=False))
    return path


class TestMultispecimenObjective(unittest.TestCase):
    KAPPA_TRUTH = 100.0
    MU_TRUTH = 50.0

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmpdir.name)
        truth = {"kappa": cls.KAPPA_TRUTH, "mu": cls.MU_TRUTH}
        start = {
            "kappa": {
                "value": 1.5 * cls.KAPPA_TRUTH, "active": True,
                "transform": {"log": cls.KAPPA_TRUTH},
            },
            "mu": {
                "value": 0.8 * cls.MU_TRUTH, "active": True,
                "transform": {"log": cls.MU_TRUTH},
            },
        }
        entries: dict[str, dict[str, Any]] = {}
        cls.single: dict[str, Objective] = {}
        for tag in ("a", "b"):
            mesh = tmp / f"{tag}.exo"
            with ExodusWriter(str(mesh), StructuredHexMesh((1.0, 1.0, 1.0), _CELLS[tag])):
                pass
            primal_out = tmp / f"primal_{tag}"
            primal = _write(tmp / f"primal_{tag}.yaml", {
                **_shared(truth, primal_out),
                **_specimen(tag, mesh, None),
            })
            primal_deck = yaml.safe_load(primal.read_text())
            primal_deck["output"]["exodus filename"] = "truth.exo"
            _write(primal, primal_deck)
            assert cmad_main(["primal", str(primal)]) == 0
            entries[tag] = _specimen(tag, mesh, primal_out / "truth.exo")
            single = _write(tmp / f"single_{tag}.yaml", {
                **_shared(start, tmp / f"out_{tag}"), **entries[tag],
            })
            cls.single[tag] = build_objective(
                load_fe_input(single, "gradient"),
            )
        joint = _write(tmp / "joint.yaml", {
            **_shared(start, tmp / "out_joint"),
            "specimens": {
                "a": entries["a"],
                "b": {**entries["b"], "weight": _WEIGHT_B},
            },
        })
        cls.joint = build_objective(load_fe_input(joint, "gradient"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_joint_is_the_weighted_sum(self) -> None:
        x = self.joint.x0
        np.testing.assert_array_equal(x, self.single["a"].x0)
        J_a, grad_a = self.single["a"].evaluate(x)
        J_b, grad_b = self.single["b"].evaluate(x)
        J, grad = self.joint.evaluate(x)
        np.testing.assert_allclose(J, J_a + _WEIGHT_B * J_b, rtol=1e-13)
        np.testing.assert_allclose(
            grad, grad_a + _WEIGHT_B * grad_b, rtol=1e-13,
        )
        entry = self.joint.history[-1]
        self.assertEqual(set(entry["specimens"]), {"a", "b"})
        self.assertEqual(entry["specimens"]["a"], {"J": J_a})
        self.assertEqual(entry["specimens"]["b"], {"J": J_b})
        self.assertEqual(
            self.joint.param_paths, ["all.elastic.kappa", "all.elastic.mu"],
        )

    def test_joint_gradient_against_finite_differences(self) -> None:
        x = self.joint.x0
        _J, grad = self.joint.evaluate(x)
        direction = np.array([0.6, -0.8])
        exact = float(grad @ direction)
        hs = np.logspace(-2, -8, 7)
        errors = []
        for h in hs:
            J_plus = self.joint.value(x + h * direction)
            J_minus = self.joint.value(x - h * direction)
            errors.append(abs((J_plus - J_minus) / (2.0 * h) - exact))
        rel_errors = np.asarray(errors) / abs(exact)
        print("directional FD relative errors:", rel_errors)
        log10_drop = float(np.log10(rel_errors.max() / rel_errors.min()))
        self.assertGreater(log10_drop, 4.0)
        self.assertLess(rel_errors.min(), 1e-6)


if __name__ == "__main__":
    unittest.main()
