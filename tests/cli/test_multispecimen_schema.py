"""Validation of an FE input file with a ``specimens`` section.

The shared sections sit at the top level and each specimen entry holds
its own discretization, boundary conditions, and qoi; the entries are
checked against the same fragments a single specimen file uses, and a
file that mixes the two forms is rejected.
"""
import copy
import unittest
from typing import Any

from cmad.io.schema import validate_deck


def _specimen(mesh: str, sideset: str) -> dict[str, Any]:
    return {
        "discretization": {
            "mesh file": mesh, "num steps": 5, "step size": 0.2,
        },
        "dirichlet bcs": {
            "expression": {
                "fix_x": ["equilibrium", 0, "xmin_sides", "0.0"],
                "ramp_x": ["equilibrium", 0, sideset, "0.05 * t"],
            },
        },
        "qoi": {"name": "fe_displacement_match", "data_file": "truth.exo"},
    }


def _joint_deck() -> dict[str, Any]:
    return {
        "problem": {"type": "fe", "name": "joint"},
        "residuals": {
            "global residual": {"type": "mechanics", "def_type": "full_3d"},
            "local residual": {
                "type": "elastic",
                "materials": {"all": {"elastic": {
                    "kappa": {"value": 100.0, "active": True},
                    "mu": 50.0,
                }}},
            },
        },
        "optimizer": {"algorithm": "L-BFGS-B"},
        "output": {"path": "out"},
        "specimens": {
            "a": _specimen("a.exo", "xmax_sides"),
            "b": {**_specimen("b.exo", "ymax_sides"), "weight": 2.0},
        },
    }


class TestMultispecimenSchema(unittest.TestCase):

    def test_two_specimens_validate(self) -> None:
        deck = _joint_deck()
        validate_deck(deck, "calibrate")
        del deck["optimizer"]
        for subcommand in ("objective", "gradient", "hessian"):
            validate_deck(deck, subcommand)

    def test_specimen_section_at_the_top_level_is_rejected(self) -> None:
        deck = _joint_deck()
        deck["discretization"] = copy.deepcopy(
            deck["specimens"]["a"]["discretization"],
        )
        with self.assertRaisesRegex(ValueError, "specimens.*discretization"):
            validate_deck(deck, "calibrate")

    def test_errors_name_the_specimen(self) -> None:
        deck = _joint_deck()
        del deck["specimens"]["b"]["qoi"]
        with self.assertRaisesRegex(ValueError, r"specimens\.b: 'qoi'"):
            validate_deck(deck, "calibrate")

        deck = _joint_deck()
        deck["specimens"]["a"]["discretization"]["mesh_file"] = "typo"
        with self.assertRaisesRegex(
                ValueError, r"specimens\.a\.discretization: .*mesh_file",
        ):
            validate_deck(deck, "calibrate")

        deck = _joint_deck()
        deck["specimens"]["b"]["weight"] = 0.0
        with self.assertRaisesRegex(ValueError, r"specimens\.b\.weight"):
            validate_deck(deck, "calibrate")

    def test_primal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "specimens: cmad primal"):
            validate_deck(_joint_deck(), "primal")


if __name__ == "__main__":
    unittest.main()
