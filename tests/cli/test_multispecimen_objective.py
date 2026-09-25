"""The objective of a two specimen input file against its parts.

Two small elastic cubes on different meshes under different tractions,
each with its own truth displacement field from ``cmad primal``. The
joint objective's value and gradient are the weighted sums of the two
single specimen objectives', its history entry carries each specimen's
own value, and its gradient agrees with a directional finite difference.
On a third, displacement controlled cube (a traction driven one has a
reaction that equals the applied load whatever the moduli are), a
weighted sum of a displacement match and a load match reports its
accumulated QoIs, which with their weights add up to its value, and a
log sum of the same terms is the weighted sum of their logarithms, with
a gradient that also agrees with finite differences.
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
_CELLS = {"a": (2, 2, 2), "b": (3, 2, 2), "c": (2, 2, 2)}
_WEIGHT_B = 2.0
# Specimen c is stretched by a prescribed displacement on xmax_sides and
# its load match reads the reaction there.
_LOAD = {"sideset": "xmax_sides", "components": [0]}
_LOAD_WEIGHT = 3.0


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
    }
    if tag in _PULL:
        sideset, traction = _PULL[tag]
        entry["surface flux bcs"] = {
            "expression": {"pull": ["equilibrium", sideset, *traction]},
        }
    else:
        entry["dirichlet bcs"]["expression"]["ramp_x"] = [
            "equilibrium", 0, "xmax_sides", "0.05 * t",
        ]
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
        for tag in ("a", "b", "c"):
            mesh = tmp / f"{tag}.exo"
            with ExodusWriter(str(mesh), StructuredHexMesh((1.0, 1.0, 1.0), _CELLS[tag])):
                pass
            primal_out = tmp / f"primal_{tag}"
            primal_deck = {
                **_shared(truth, primal_out), **_specimen(tag, mesh, None),
            }
            primal_deck["output"]["exodus filename"] = "truth.exo"
            if tag == "c":
                primal_deck["qoi"] = {
                    "name": "fe_load_match", **_LOAD,
                    "output_file": str(tmp / "load.csv"),
                }
            primal = _write(tmp / f"primal_{tag}.yaml", primal_deck)
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
        terms = [
            entries["c"]["qoi"],
            {
                "name": "fe_load_match", **_LOAD,
                "data_file": str(tmp / "load.csv"),
                "weight": _LOAD_WEIGHT,
            },
        ]
        cls.combined: dict[str, Objective] = {}
        for name in ("fe_weighted_sum", "fe_log_sum"):
            path = _write(tmp / f"{name}.yaml", {
                **_shared(start, tmp / f"out_{name}"),
                **entries["c"],
                "qoi": {"name": name, "terms": terms},
            })
            cls.combined[name] = build_objective(
                load_fe_input(path, "gradient"),
            )

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
        for tag, J_tag in (("a", J_a), ("b", J_b)):
            self.assertEqual(entry["specimens"][tag], {
                "J": J_tag,
                "accumulated_qois": {"fe_displacement_match": J_tag},
            })
        self.assertEqual(
            self.joint.param_paths, ["all.elastic.kappa", "all.elastic.mu"],
        )

    def test_weighted_sum_reports_its_accumulated_qois(self) -> None:
        J_c, _grad = self.single["c"].evaluate(self.single["c"].x0)
        weighted = self.combined["fe_weighted_sum"]
        J, _grad = weighted.evaluate(weighted.x0)
        accumulated = weighted.history[-1]["accumulated_qois"]
        self.assertEqual(
            list(accumulated), ["fe_displacement_match", "fe_load_match"],
        )
        np.testing.assert_allclose(
            accumulated["fe_displacement_match"], J_c, rtol=1e-13,
        )
        # A mismatch, not a round off residual.
        self.assertGreater(accumulated["fe_load_match"], 1e-10)
        np.testing.assert_allclose(
            J,
            accumulated["fe_displacement_match"]
            + _LOAD_WEIGHT * accumulated["fe_load_match"],
            rtol=1e-14,
        )

    def test_log_sum_is_the_weighted_sum_of_logarithms(self) -> None:
        log_sum = self.combined["fe_log_sum"]
        J, _grad = log_sum.evaluate(log_sum.x0)
        accumulated = log_sum.history[-1]["accumulated_qois"]
        np.testing.assert_allclose(
            J,
            np.log(accumulated["fe_displacement_match"])
            + _LOAD_WEIGHT * np.log(accumulated["fe_load_match"]),
            rtol=1e-13,
        )
        self._check_gradient_against_finite_differences(log_sum, "log sum")

    def test_joint_gradient_against_finite_differences(self) -> None:
        self._check_gradient_against_finite_differences(self.joint, "joint")

    def _check_gradient_against_finite_differences(
            self, objective: Objective, label: str,
    ) -> None:
        x = objective.x0
        _J, grad = objective.evaluate(x)
        direction = np.array([0.6, -0.8])
        exact = float(grad @ direction)
        hs = np.logspace(-2, -8, 7)
        errors = []
        for h in hs:
            J_plus = objective.value(x + h * direction)
            J_minus = objective.value(x - h * direction)
            errors.append(abs((J_plus - J_minus) / (2.0 * h) - exact))
        rel_errors = np.asarray(errors) / abs(exact)
        print(f"{label} directional FD relative errors:", rel_errors)
        log10_drop = float(np.log10(rel_errors.max() / rel_errors.min()))
        self.assertGreater(log10_drop, 4.0)
        self.assertLess(rel_errors.min(), 1e-6)


if __name__ == "__main__":
    unittest.main()
