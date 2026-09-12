"""``cmad primal`` round trip on a heat transfer problem: a cube held at
two temperatures on opposite faces, whose linear profile a P1 mesh
reproduces to round off, read back from the Exodus file."""
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.cli.common import build_fe_problem_from_deck
from cmad.cli.main import main as cmad_main
from cmad.fem.mesh import StructuredHexMesh
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType


def _make_deck(mesh_filename: str, out_dir: str) -> dict[str, Any]:
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": mesh_filename,
            "num steps": 1,
            "step size": 1.0,
        },
        "residuals": {
            "global residual": {"type": "heat_transfer"},
            "local residual": {
                "type": "conduction",
                "materials": {
                    "all": {"thermal": {"conductivity": 16.0}},
                },
            },
        },
        "dirichlet bcs": {
            "expression": {
                "hot": ["energy balance", 0, "xmin_sides", "400.0"],
                "cold": ["energy balance", 0, "xmax_sides", "300.0"],
            },
        },
        "output": {
            "path": out_dir,
            "exodus filename": "primal.exo",
            "global residual": ["T"],
            "local residual": {"all": ["heat flux"]},
        },
    }


def _make_bar_deck(
        mesh_filename: str, out_dir: str, sections: dict[str, Any],
) -> dict[str, Any]:
    """A bar of 8 hexes with ``T = 1000`` at ``x = 0``, ``k = 10``, plus
    the given surface condition sections."""
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": mesh_filename,
            "num steps": 1,
            "step size": 1.0,
        },
        "residuals": {
            "global residual": {"type": "heat_transfer"},
            "local residual": {
                "type": "conduction",
                "materials": {
                    "all": {"thermal": {"conductivity": 10.0}},
                },
            },
        },
        "dirichlet bcs": {
            "expression": {
                "hot": ["energy balance", 0, "xmin_sides", 1000.0],
            },
        },
        **sections,
        "output": {
            "path": out_dir,
            "exodus filename": "primal.exo",
            "global residual": ["T"],
        },
    }


def _end_temperature(
        tmp: Path, sections: dict[str, Any],
) -> float:
    """The temperature at ``x = 1`` after ``cmad primal`` on the bar."""
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (8, 1, 1))
    with ExodusWriter(str(tmp / "mesh.exo"), mesh):
        pass
    deck_path = tmp / "deck.yaml"
    deck_path.write_text(yaml.safe_dump(
        _make_bar_deck(str(tmp / "mesh.exo"), str(tmp / "out"), sections),
        sort_keys=False,
    ))
    assert cmad_main(["primal", str(deck_path)]) == 0
    results = read_results(
        tmp / "out" / "primal.exo",
        nodal_field_specs=[FieldSpec("T", VarType.SCALAR)],
    )
    T = results.nodal["T"][-1].reshape(-1)
    at_end = T[np.isclose(mesh.nodes[:, 0], 1.0)]
    np.testing.assert_allclose(at_end, at_end[0], rtol=1e-10)
    return float(at_end[0])


class TestPrimalFeHeatRoundTrip(unittest.TestCase):
    def test_convection_input_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            T_end = _end_temperature(Path(tmpdir), {
                "convection bcs": {"expression": {
                    "cooled": ["energy balance", "xmax_sides", 25.0, 300.0],
                }},
            })
        expected = (10.0 * 1000.0 + 25.0 * 300.0) / (10.0 + 25.0)
        self.assertAlmostEqual(T_end, expected, places=8)

    def test_radiation_input_file(self) -> None:
        sigma_B = 5.67e-8
        with tempfile.TemporaryDirectory() as tmpdir:
            T_end = _end_temperature(Path(tmpdir), {
                "radiation bcs": {
                    "stefan boltzmann constant": sigma_B,
                    "expression": {
                        "glowing": ["energy balance", "xmax_sides", 0.8, 300.0],
                    },
                },
            })
        # k (T0 - T_L) = eps sigma_B (T_L^4 - T_inf^4) at the end.
        balance = 10.0 * (1000.0 - T_end) - 0.8 * sigma_B * (
            T_end ** 4 - 300.0 ** 4
        )
        self.assertLess(abs(balance) / 10.0 / 1000.0, 1e-8)

    def test_linear_profile_between_two_faces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
            with ExodusWriter(str(tmp / "mesh.exo"), mesh):
                pass
            deck = _make_deck(str(tmp / "mesh.exo"), str(tmp / "out"))
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

            self.assertEqual(cmad_main(["primal", str(deck_path)]), 0)

            results = read_results(
                tmp / "out" / "primal.exo",
                nodal_field_specs=[FieldSpec("T", VarType.SCALAR)],
                element_field_specs={
                    "all": [FieldSpec("heat flux", VarType.VECTOR)],
                },
            )
            nodes = build_fe_problem_from_deck(
                deck_path, "primal",
            ).fe_problem.mesh.nodes
            T = results.nodal["T"][-1].reshape(-1)
            np.testing.assert_allclose(T, 400.0 - 100.0 * nodes[:, 0], rtol=1e-6)
            # q = -k dT/dx = -16 (-100) along x, zero across.
            q = results.element["all"]["heat flux"][-1]
            np.testing.assert_allclose(q[:, 0], 1600.0, rtol=1e-6)
            np.testing.assert_allclose(q[:, 1:], 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
