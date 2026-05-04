"""Tests for FE-deck schema dispatch, validation, and normalization."""

import unittest
import warnings
from typing import Any

from cmad.io.deck import (
    apply_deck_defaults,
    maybe_unwrap_top_level,
    strip_calibr8_only,
)
from cmad.io.schema import validate_deck


def _minimal_fe_deck() -> dict[str, Any]:
    """Schema-valid FE primal deck with only the required sections."""
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": "cube.exo",
            "num steps": 5,
            "step size": 0.1,
        },
        "residuals": {
            "global residual": {"type": "small_disp_equilibrium"},
            "local residual": {
                "type": "elastic",
                "def_type": "full_3d",
                "materials": {
                    "body": {
                        "elastic": {"kappa": 100.0, "mu": 50.0},
                    },
                },
            },
        },
        "output": {"path": "out/"},
    }


def _minimal_mp_deck() -> dict[str, Any]:
    """Schema-valid MP primal deck with only the required sections."""
    return {
        "problem": {"type": "material_point"},
        "model": {
            "name": "elastic",
            "def_type": "full_3d",
        },
        "parameters": {
            "elastic": {"kappa": 100.0, "mu": 50.0},
        },
        "deformation": {
            "history_file": "F.npy",
        },
        "output": {"path": "out/"},
    }


class TestFEDeckSchema(unittest.TestCase):
    def test_minimal_fe_deck_validates(self) -> None:
        deck = _minimal_fe_deck()
        resolved = apply_deck_defaults(deck)
        validate_deck(resolved, "primal")
        self.assertEqual(resolved["output"]["format"], "exodus")
        self.assertIn(
            "nonlinear max iters",
            resolved["residuals"]["global residual"],
        )

    def test_missing_required_section_errors(self) -> None:
        deck = _minimal_fe_deck()
        del deck["discretization"]
        resolved = apply_deck_defaults(deck)
        with self.assertRaises(ValueError) as ctx:
            validate_deck(resolved, "primal")
        self.assertIn("discretization", str(ctx.exception))

    def test_unknown_problem_subcommand_pair_raises(self) -> None:
        deck = _minimal_fe_deck()
        with self.assertRaises(ValueError) as ctx:
            validate_deck(apply_deck_defaults(deck), "objective")
        msg = str(ctx.exception)
        self.assertIn("fe", msg)
        self.assertIn("objective", msg)

    def test_optional_sections_are_optional(self) -> None:
        deck = _minimal_fe_deck()
        self.assertNotIn("dirichlet bcs", deck)
        self.assertNotIn("surface flux bcs", deck)
        self.assertNotIn("body forces", deck)
        validate_deck(apply_deck_defaults(deck), "primal")

    def test_dbc_well_formed_validates(self) -> None:
        deck = _minimal_fe_deck()
        deck["dirichlet bcs"] = {
            "expression": {
                "bc 1": ["displacement", 0, "xmin_sides", 0.0],
                "bc 2": ["displacement", 1, "ymin_sides", "0.01 * t"],
            },
        }
        validate_deck(apply_deck_defaults(deck), "primal")

    def test_dbc_wrong_length_errors(self) -> None:
        deck = _minimal_fe_deck()
        deck["dirichlet bcs"] = {
            "expression": {
                "bc 1": ["displacement", 0, "xmin_sides"],
            },
        }
        with self.assertRaises(ValueError):
            validate_deck(apply_deck_defaults(deck), "primal")

    def test_dbc_wrong_type_errors(self) -> None:
        deck = _minimal_fe_deck()
        deck["dirichlet bcs"] = {
            "expression": {
                "bc 1": ["displacement", -1, "xmin_sides", 0.0],
            },
        }
        with self.assertRaises(ValueError):
            validate_deck(apply_deck_defaults(deck), "primal")

    def test_unknown_top_level_section_errors(self) -> None:
        deck = _minimal_fe_deck()
        deck["spurious_section"] = {"foo": "bar"}
        with self.assertRaises(ValueError) as ctx:
            validate_deck(apply_deck_defaults(deck), "primal")
        self.assertIn("spurious_section", str(ctx.exception))

    def test_residuals_missing_required_slot_errors(self) -> None:
        deck = _minimal_fe_deck()
        del deck["residuals"]["global residual"]
        with self.assertRaises(ValueError):
            validate_deck(apply_deck_defaults(deck), "primal")

    def test_unregistered_gr_errors(self) -> None:
        deck = _minimal_fe_deck()
        deck["residuals"]["global residual"]["type"] = "not_a_real_gr"
        with self.assertRaises(ValueError) as ctx:
            validate_deck(apply_deck_defaults(deck), "primal")
        msg = str(ctx.exception)
        self.assertIn("not_a_real_gr", msg)
        self.assertIn("global residual", msg)

    def test_unregistered_model_errors(self) -> None:
        deck = _minimal_fe_deck()
        deck["residuals"]["local residual"]["type"] = "not_a_real_model"
        with self.assertRaises(ValueError) as ctx:
            validate_deck(apply_deck_defaults(deck), "primal")
        self.assertIn("not_a_real_model", str(ctx.exception))


class TestDiscretizationTimeSpecs(unittest.TestCase):
    def test_num_steps_and_step_size_form(self) -> None:
        deck = _minimal_fe_deck()
        validate_deck(apply_deck_defaults(deck), "primal")

    def test_times_file_form(self) -> None:
        deck = _minimal_fe_deck()
        deck["discretization"] = {
            "mesh file": "cube.exo",
            "times file": "t.npy",
        }
        validate_deck(apply_deck_defaults(deck), "primal")

    def test_inline_times_form(self) -> None:
        deck = _minimal_fe_deck()
        deck["discretization"] = {
            "mesh file": "cube.exo",
            "times": [0.0, 0.1, 0.5, 1.0],
        }
        validate_deck(apply_deck_defaults(deck), "primal")

    def test_two_time_specs_simultaneously_error(self) -> None:
        deck = _minimal_fe_deck()
        deck["discretization"]["times"] = [0.0, 1.0]
        with self.assertRaises(ValueError):
            validate_deck(apply_deck_defaults(deck), "primal")

    def test_no_time_spec_errors(self) -> None:
        deck = _minimal_fe_deck()
        deck["discretization"] = {"mesh file": "cube.exo"}
        with self.assertRaises(ValueError):
            validate_deck(apply_deck_defaults(deck), "primal")


class TestTopLevelWrapper(unittest.TestCase):
    def test_wrapped_deck_validates(self) -> None:
        wrapped = {"cube_test": _minimal_fe_deck()}
        resolved = apply_deck_defaults(wrapped)
        self.assertIn("problem", resolved)
        self.assertNotIn("cube_test", resolved)
        validate_deck(resolved, "primal")

    def test_unwrap_helper_idempotent(self) -> None:
        deck = _minimal_fe_deck()
        self.assertIs(maybe_unwrap_top_level(deck), deck)
        wrapped = {"x": deck}
        self.assertIs(maybe_unwrap_top_level(wrapped), deck)
        self.assertIs(maybe_unwrap_top_level(maybe_unwrap_top_level(wrapped)), deck)

    def test_single_key_without_problem_not_unwrapped(self) -> None:
        # A deck with a single top-level key whose value lacks `problem`
        # is NOT auto-unwrapped (e.g. malformed deck).
        not_wrapped = {"only_key": {"unrelated": 1}}
        self.assertIs(maybe_unwrap_top_level(not_wrapped), not_wrapped)


class TestCalibr8OnlySections(unittest.TestCase):
    def test_linear_algebra_section_stripped_with_warning(self) -> None:
        deck = _minimal_fe_deck()
        deck["linear algebra"] = {"Linear Solver Type": "Belos"}
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            resolved = apply_deck_defaults(deck)
        self.assertNotIn("linear algebra", resolved)
        self.assertTrue(
            any("linear algebra" in str(w.message) for w in record)
        )
        validate_deck(resolved, "primal")

    def test_regression_section_stripped_with_warning(self) -> None:
        deck = _minimal_fe_deck()
        deck["regression"] = {"QoI": 1.0e-3, "relative error tol": 1.0e-6}
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            resolved = apply_deck_defaults(deck)
        self.assertNotIn("regression", resolved)
        self.assertTrue(
            any("regression" in str(w.message) for w in record)
        )
        validate_deck(resolved, "primal")

    def test_strip_idempotent_no_warning_on_clean_deck(self) -> None:
        deck = _minimal_fe_deck()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            stripped = strip_calibr8_only(deck)
        self.assertIs(stripped, deck)
        self.assertEqual(len(record), 0)


class TestMPDeckUnchanged(unittest.TestCase):
    """Regression coverage that MP-deck handling is unchanged."""

    def test_minimal_mp_primal_validates(self) -> None:
        resolved = apply_deck_defaults(_minimal_mp_deck())
        self.assertIn("newton", resolved["solver"])
        self.assertEqual(resolved["output"]["format"], "npy")
        validate_deck(resolved, "primal")


if __name__ == "__main__":
    unittest.main()
