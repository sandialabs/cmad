"""End-to-end ``cmad primal`` round-trip on FE problem types.

Two classes covering the dispatch arms reachable from
``problem.type: fe``:

- ``TestPrimalFeClosedFormRoundTrip`` — `Elastic` material under a
  ramped uniaxial-stress DBC, tight analytic check
  ``sigma_xx = E * eps_x``.
- ``TestPrimalFeCoupledRoundTrip`` — `SmallElasticPlastic` (J2 +
  Voce hardening) under a strain ramp into the plastic regime, loose
  plumbing check (yielded + uniaxial-stress); pointwise return-map
  correctness is covered at the driver/model layer.
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


def _write_hex_cube_mesh(
        path: Path,
        divisions: tuple[int, int, int] = (1, 1, 1),
) -> None:
    """Write a ``[0, 1]^3`` Q1 hex mesh-only Exodus file at ``path``."""
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), divisions)
    with ExodusWriter(str(path), mesh):
        pass


def _make_fe_primal_deck_elastic(
        mesh_filename: str, output_section: dict[str, Any],
) -> dict[str, Any]:
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
                "ramp_x": ["displacement", 0, "xmax_sides", "0.05 * t"],
            },
        },
        "output": output_section,
    }


def _make_fe_primal_deck_coupled(
        mesh_filename: str, output_section: dict[str, Any],
        ramp_expression: str = "0.01 * t",
) -> dict[str, Any]:
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
                "type": "small_elastic_plastic",
                "def_type": "full_3d",
                "effective_stress": "J2",
                "materials": {
                    "all": {
                        "rotation matrix": [
                            [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        ],
                        "elastic": {"E": 200_000.0, "nu": 0.3},
                        "plastic": {
                            "effective stress": {"J2": 0.0},
                            "flow stress": {
                                "initial yield": {"Y": 200.0},
                                "hardening": {
                                    "voce": {"S": 200.0, "D": 20.0},
                                },
                            },
                        },
                    },
                },
            },
        },
        "dirichlet bcs": {
            "expression": {
                "pin_x": ["displacement", 0, "xmin_sides", "0.0"],
                "pin_y": ["displacement", 1, "ymin_sides", "0.0"],
                "pin_z": ["displacement", 2, "zmin_sides", "0.0"],
                "ramp_x": [
                    "displacement", 0, "xmax_sides", ramp_expression,
                ],
            },
        },
        "output": output_section,
    }


class TestPrimalFeClosedFormRoundTrip(unittest.TestCase):
    """Linear-elastic `Elastic` (CLOSED_FORM) FE primal round-trip
    against the analytic uniaxial-stress isotropic-elastic answer."""

    def test_uniaxial_stress_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_hex_cube_mesh(tmp / "mesh.exo")
            deck = _make_fe_primal_deck_elastic(
                mesh_filename="mesh.exo",
                output_section={
                    "path": "out",
                    "format": "exodus",
                    "nodal fields": [
                        {"name": "displacement", "var_type": "vector"},
                    ],
                    "element fields by block": {
                        "all": [
                            {"name": "cauchy", "var_type": "sym_tensor"},
                        ],
                    },
                },
            )
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            self.assertEqual(
                cmad_main(["primal", str(deck_path)]), 0,
            )

            out_dir = tmp / "out"
            self.assertTrue((out_dir / "primal.exo").exists())
            self.assertTrue((out_dir / "solver.json").exists())
            self.assertTrue((out_dir / "deck.resolved.yaml").exists())

            results = read_results(
                out_dir / "primal.exo",
                nodal_field_specs=[
                    FieldSpec("displacement", VarType.VECTOR),
                ],
                element_field_specs={
                    "all": [FieldSpec("cauchy", VarType.SYM_TENSOR)],
                },
            )
            self.assertEqual(results.time.shape, (6,))
            self.assertAlmostEqual(float(results.time[-1]), 1.0)

            # Uniaxial-stress isotropic linear elastic:
            # E = 9 * kappa * mu / (3 * kappa + mu);
            # sigma_xx = E * eps_x; lateral / shear ~ 0.
            kappa, mu = 100.0, 50.0
            E = 9.0 * kappa * mu / (3.0 * kappa + mu)
            eps_x = 0.05
            sigma_xx_analytic = E * eps_x

            cauchy_terminal = results.element["all"]["cauchy"][-1]
            np.testing.assert_allclose(
                cauchy_terminal[:, 0], sigma_xx_analytic, rtol=1e-6,
            )
            np.testing.assert_allclose(
                cauchy_terminal[:, 1:], 0.0, atol=1e-6,
            )


class TestPrimalFeCoupledRoundTrip(unittest.TestCase):
    """SmallElasticPlastic (J2 + Voce, COUPLED local Newton) FE
    primal round-trip — confirms the deck-driven COUPLED dispatch +
    local-Newton kwargs + xi flow + writer chain wire correctly.
    Yielded-and-uniaxial loose check; pointwise return-map
    correctness lives in the driver/model tests."""

    def test_j2_voce_uniaxial_yield_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_hex_cube_mesh(tmp / "mesh.exo")
            deck = _make_fe_primal_deck_coupled(
                mesh_filename="mesh.exo",
                output_section={
                    "path": "out",
                    "format": "exodus",
                    "nodal fields": [
                        {"name": "displacement", "var_type": "vector"},
                    ],
                    "element fields by block": {
                        "all": [
                            {"name": "cauchy", "var_type": "sym_tensor"},
                        ],
                    },
                },
                ramp_expression="0.01 * t",
            )
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            self.assertEqual(
                cmad_main(["primal", str(deck_path)]), 0,
            )

            out_dir = tmp / "out"
            self.assertTrue((out_dir / "primal.exo").exists())
            self.assertTrue((out_dir / "solver.json").exists())
            self.assertTrue((out_dir / "deck.resolved.yaml").exists())

            results = read_results(
                out_dir / "primal.exo",
                element_field_specs={
                    "all": [FieldSpec("cauchy", VarType.SYM_TENSOR)],
                },
            )
            cauchy_terminal = results.element["all"]["cauchy"][-1]
            # Yielded: sigma_xx exceeds initial yield Y=200 (peak eps_x =
            # 0.01, eps_yield = Y/E = 1e-3 -> ~10x yield strain at terminal).
            self.assertTrue(np.all(cauchy_terminal[:, 0] > 200.0))
            # Uniaxial stress: lateral / shear components ≈ 0.
            np.testing.assert_allclose(
                cauchy_terminal[:, 1:], 0.0, atol=1.0,
            )


if __name__ == "__main__":
    unittest.main()
