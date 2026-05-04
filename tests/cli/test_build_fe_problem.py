"""Tests for cmad.cli.common.build_fe_problem_from_deck."""

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import yaml
from jax import jit

from cmad.cli.common import (
    FEProblemBundle,
    build_fe_problem_from_deck,
)
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import (
    _DEFAULT_ASSEMBLY_QUADRATURE,
    _DEFAULT_SIDE_QUADRATURE,
)
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import Mesh, StructuredHexMesh
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium


def _minimal_fe_deck(mesh_filename: str = "cube.exo") -> dict[str, Any]:
    """Schema-valid FE primal deck with only the required sections."""
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": mesh_filename,
            "num steps": 5,
            "step size": 0.1,
        },
        "residuals": {
            "global residual": {"type": "small_disp_equilibrium"},
            "local residual": {
                "type": "elastic",
                "def_type": "full_3d",
                "materials": {
                    "all": {
                        "elastic": {"kappa": 100.0, "mu": 50.0},
                    },
                },
            },
        },
        "output": {"path": "out/"},
    }


def _hex_cube_mesh(
        divisions: tuple[int, int, int] = (2, 2, 2),
) -> Mesh:
    return StructuredHexMesh((1.0, 1.0, 1.0), divisions)


def _two_block_mesh() -> Mesh:
    """Hex mesh with two element blocks (4 elements split 2-2)."""
    base = StructuredHexMesh((2.0, 1.0, 1.0), (2, 1, 1))
    n_elems = base.connectivity.shape[0]
    half = n_elems // 2
    return Mesh(
        nodes=base.nodes,
        connectivity=base.connectivity,
        element_family=base.element_family,
        element_blocks={
            "left": np.arange(0, half, dtype=np.intp),
            "right": np.arange(half, n_elems, dtype=np.intp),
        },
        node_sets=base.node_sets,
        side_sets=base.side_sets,
    )


def _build_bundle(
        deck: dict[str, Any], mesh: Mesh, tmp: Path,
) -> FEProblemBundle:
    """Write the deck and build, with ``read_mesh`` patched to return ``mesh``.

    Patches the symbol where the builder imports it
    (``cmad.cli.common.read_mesh``) so disk I/O for the mesh file is
    skipped — Exodus reader correctness is covered by tests in
    ``tests/io/test_exodus.py``.
    """
    deck_path = tmp / "deck.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
    with patch("cmad.cli.common.read_mesh", return_value=mesh):
        return build_fe_problem_from_deck(deck_path, "primal")


class TestMinimalBuild(unittest.TestCase):
    def test_minimal_deck_returns_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _build_bundle(
                _minimal_fe_deck(), _hex_cube_mesh(), Path(tmpdir),
            )
        self.assertIsInstance(bundle, FEProblemBundle)
        self.assertIsInstance(bundle.fe_problem.gr, SmallDispEquilibrium)
        self.assertEqual(bundle.t_schedule.shape, (6,))
        self.assertGreater(bundle.fe_problem.dof_map.num_total_dofs, 0)

    def test_resolved_deck_preserves_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _build_bundle(
                _minimal_fe_deck(), _hex_cube_mesh(), Path(tmpdir),
            )
        for key in ("problem", "discretization", "residuals", "output"):
            self.assertIn(key, bundle.resolved)


class TestTimeSchedule(unittest.TestCase):
    def test_num_steps_step_size_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _build_bundle(
                _minimal_fe_deck(), _hex_cube_mesh(), Path(tmpdir),
            )
        self.assertEqual(bundle.t_schedule.shape, (6,))
        np.testing.assert_allclose(bundle.t_schedule, np.arange(6) * 0.1)

    def test_inline_times_form(self) -> None:
        deck = _minimal_fe_deck()
        deck["discretization"] = {
            "mesh file": "cube.exo",
            "times": [0.0, 0.5, 1.0],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _build_bundle(deck, _hex_cube_mesh(), Path(tmpdir))
        np.testing.assert_allclose(bundle.t_schedule, [0.0, 0.5, 1.0])

    def test_times_file_npy(self) -> None:
        expected = np.array([0.0, 0.25, 0.75, 1.5])
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "t.npy", expected)
            deck = _minimal_fe_deck()
            deck["discretization"] = {
                "mesh file": "cube.exo",
                "times file": "t.npy",
            }
            bundle = _build_bundle(deck, _hex_cube_mesh(), tmp)
        np.testing.assert_allclose(bundle.t_schedule, expected)

    def test_times_file_txt(self) -> None:
        expected = np.array([0.0, 0.5, 1.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.savetxt(tmp / "t.txt", expected)
            deck = _minimal_fe_deck()
            deck["discretization"] = {
                "mesh file": "cube.exo",
                "times file": "t.txt",
            }
            bundle = _build_bundle(deck, _hex_cube_mesh(), tmp)
        np.testing.assert_allclose(bundle.t_schedule, expected)

    def test_times_file_unsupported_extension_raises(self) -> None:
        deck = _minimal_fe_deck()
        deck["discretization"] = {
            "mesh file": "cube.exo",
            "times file": "t.xml",
        }
        with tempfile.TemporaryDirectory() as tmpdir, \
                self.assertRaises(ValueError) as ctx:
            _build_bundle(deck, _hex_cube_mesh(), Path(tmpdir))
        self.assertIn(".xml", str(ctx.exception))


class TestBoundaryConditions(unittest.TestCase):
    def _build(self, extras: dict[str, Any]) -> FEProblemBundle:
        deck = _minimal_fe_deck()
        deck.update(extras)
        with tempfile.TemporaryDirectory() as tmpdir:
            return _build_bundle(deck, _hex_cube_mesh(), Path(tmpdir))

    def test_dbc_constant_value(self) -> None:
        bundle = self._build({
            "dirichlet bcs": {"expression": {
                "bc 1": ["displacement", 0, "xmin_sides", 0.0],
            }},
        })
        vals = bundle.fe_problem.dof_map.evaluate_prescribed_values(t=1.0)
        np.testing.assert_allclose(vals, 0.0)

    def test_dbc_string_expression_time_ramp(self) -> None:
        bundle = self._build({
            "dirichlet bcs": {"expression": {
                "bc 1": ["displacement", 0, "xmax_sides", "0.01 * t"],
            }},
        })
        v0 = bundle.fe_problem.dof_map.evaluate_prescribed_values(t=0.0)
        v1 = bundle.fe_problem.dof_map.evaluate_prescribed_values(t=1.0)
        np.testing.assert_allclose(v0, 0.0)
        np.testing.assert_allclose(v1, 0.01)

    def test_nbc_constant_components(self) -> None:
        bundle = self._build({
            "surface flux bcs": {"expression": {
                "flux 1": ["displacement", "xmax_sides", 0.0, 0.0, 1.0],
            }},
        })
        rbcs = bundle.fe_problem.resolved_neumann_bcs
        self.assertEqual(len(rbcs), 1)
        rbc = rbcs[0]
        self.assertEqual(rbc.num_components, 3)
        self.assertTrue(callable(rbc.values))
        out = np.asarray(rbc.values(jnp.zeros((4, 3)), 0.5))
        self.assertEqual(out.shape, (4, 3))
        np.testing.assert_allclose(out, [[0.0, 0.0, 1.0]] * 4)

    def test_body_force_string_components(self) -> None:
        bundle = self._build({
            "body forces": {"expression": {
                "bf 1": ["displacement", "0.0", "0.0", "-9.81"],
            }},
        })
        forcing = bundle.fe_problem.forcing_fns_by_block_idx
        self.assertIsNotNone(forcing)
        out = np.asarray(forcing[0](jnp.array([0.5, 0.5, 0.5]), 0.0))
        self.assertEqual(out.shape, (3,))
        np.testing.assert_allclose(out, [0.0, 0.0, -9.81])

    def test_unknown_resid_name_in_dbc_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build({
                "dirichlet bcs": {"expression": {
                    "bc 1": ["nonsense", 0, "xmin_sides", 0.0],
                }},
            })
        msg = str(ctx.exception)
        self.assertIn("nonsense", msg)
        self.assertIn("displacement", msg)

    def test_dbc_eq_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build({
                "dirichlet bcs": {"expression": {
                    "bc 1": ["displacement", 5, "xmin_sides", 0.0],
                }},
            })
        self.assertIn("5", str(ctx.exception))

    def test_duplicate_body_force_for_resid_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build({
                "body forces": {"expression": {
                    "bf 1": ["displacement", "0.0", "0.0", "-9.81"],
                    "bf 2": ["displacement", "1.0", "0.0", "0.0"],
                }},
            })
        msg = str(ctx.exception)
        self.assertIn("displacement", msg)
        self.assertIn("body", msg.lower())


class TestFiniteElements(unittest.TestCase):
    def _build(
            self, fe_overrides: dict[str, str] | None,
    ) -> FEProblemBundle:
        deck = _minimal_fe_deck()
        if fe_overrides is not None:
            deck["discretization"]["finite elements"] = fe_overrides
        with tempfile.TemporaryDirectory() as tmpdir:
            return _build_bundle(deck, _hex_cube_mesh(), Path(tmpdir))

    def test_default_fe_for_hex_mesh(self) -> None:
        bundle = self._build(None)
        self.assertIs(
            bundle.fe_problem.dof_map.field_layouts[0].finite_element,
            Q1_HEX,
        )

    def test_explicit_fe_override(self) -> None:
        bundle = self._build({"u": "Q1"})
        self.assertIs(
            bundle.fe_problem.dof_map.field_layouts[0].finite_element,
            Q1_HEX,
        )

    def test_unknown_var_name_in_finite_elements_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build({"p": "Q1"})
        msg = str(ctx.exception)
        self.assertIn("p", msg)
        self.assertIn("var_name", msg)

    def test_family_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build({"u": "P1"})
        msg = str(ctx.exception)
        self.assertIn("HEX", msg)
        self.assertIn("TET", msg)


class TestQuadrature(unittest.TestCase):
    def _build(
            self, quad_overrides: dict[str, int] | None,
    ) -> FEProblemBundle:
        deck = _minimal_fe_deck()
        if quad_overrides is not None:
            deck["discretization"]["quadrature"] = quad_overrides
        with tempfile.TemporaryDirectory() as tmpdir:
            return _build_bundle(deck, _hex_cube_mesh(), Path(tmpdir))

    def test_default_quadrature_when_omitted(self) -> None:
        bundle = self._build(None)
        rule = bundle.fe_problem.assembly_quadrature[
            ElementFamily.HEX_LINEAR
        ]
        default_rule = _DEFAULT_ASSEMBLY_QUADRATURE[
            ElementFamily.HEX_LINEAR
        ]
        self.assertEqual(rule.xi.shape, default_rule.xi.shape)

    def test_volume_degree_override(self) -> None:
        # degree=4 → ceil(5/2) = 3 pts/axis → 27 IPs (vs default 8).
        bundle = self._build({"volume degree": 4})
        rule = bundle.fe_problem.assembly_quadrature[
            ElementFamily.HEX_LINEAR
        ]
        default_rule = _DEFAULT_ASSEMBLY_QUADRATURE[
            ElementFamily.HEX_LINEAR
        ]
        self.assertGreater(rule.xi.shape[0], default_rule.xi.shape[0])

    def test_surface_degree_override(self) -> None:
        # degree=4 → 3 pts/axis on the quad face → 9 IPs (vs default 4).
        bundle = self._build({"surface degree": 4})
        rule = bundle.fe_problem.side_quadrature[
            ElementFamily.HEX_LINEAR
        ]
        default_rule = _DEFAULT_SIDE_QUADRATURE[
            ElementFamily.HEX_LINEAR
        ]
        self.assertGreater(rule.xi.shape[0], default_rule.xi.shape[0])


class TestModelBlockMatching(unittest.TestCase):
    def test_block_name_mismatch_raises(self) -> None:
        deck = _minimal_fe_deck()
        deck["residuals"]["local residual"]["materials"] = {
            "wrong_block_name": {
                "elastic": {"kappa": 100.0, "mu": 50.0},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir, \
                self.assertRaises(ValueError) as ctx:
            _build_bundle(deck, _hex_cube_mesh(), Path(tmpdir))
        msg = str(ctx.exception)
        self.assertIn("wrong_block_name", msg)
        self.assertIn("all", msg)

    def test_two_block_deck_builds(self) -> None:
        deck = _minimal_fe_deck()
        deck["residuals"]["local residual"]["materials"] = {
            "left": {"elastic": {"kappa": 100.0, "mu": 50.0}},
            "right": {"elastic": {"kappa": 200.0, "mu": 80.0}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _build_bundle(deck, _two_block_mesh(), Path(tmpdir))
        self.assertEqual(
            set(bundle.fe_problem.models_by_block.keys()),
            {"left", "right"},
        )
        for block in ("left", "right"):
            self.assertEqual(
                bundle.fe_problem.modes_by_block[block],
                GlobalResidualMode.CLOSED_FORM,
            )

    def test_closed_form_mode_for_elastic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _build_bundle(
                _minimal_fe_deck(), _hex_cube_mesh(), Path(tmpdir),
            )
        self.assertEqual(
            bundle.fe_problem.modes_by_block["all"],
            GlobalResidualMode.CLOSED_FORM,
        )


class TestJitTraceability(unittest.TestCase):
    def test_forcing_fn_jit_traces(self) -> None:
        deck = _minimal_fe_deck()
        deck["body forces"] = {"expression": {
            "bf 1": ["displacement", "x + t", "0.0", "y * z"],
        }}
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _build_bundle(deck, _hex_cube_mesh(), Path(tmpdir))
        assert bundle.fe_problem.forcing_fns_by_block_idx is not None
        fn = bundle.fe_problem.forcing_fns_by_block_idx[0]
        jitted = jit(fn)
        out = np.asarray(jitted(jnp.array([0.5, 0.5, 0.5]), 1.0))
        self.assertEqual(out.shape, (3,))
        np.testing.assert_allclose(out, [1.5, 0.0, 0.25])

    def test_nbc_value_callable_jit_traces(self) -> None:
        deck = _minimal_fe_deck()
        deck["surface flux bcs"] = {"expression": {
            "flux 1": [
                "displacement", "xmax_sides",
                "0.0", "0.0", "0.5 * t",
            ],
        }}
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = _build_bundle(deck, _hex_cube_mesh(), Path(tmpdir))
        rbc = bundle.fe_problem.resolved_neumann_bcs[0]
        self.assertTrue(callable(rbc.values))
        jitted = jit(rbc.values)
        out = np.asarray(jitted(jnp.zeros((4, 3)), 0.5))
        self.assertEqual(out.shape, (4, 3))
        np.testing.assert_allclose(out[:, 2], 0.25)


if __name__ == "__main__":
    unittest.main()
