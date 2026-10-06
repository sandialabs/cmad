"""``cmad primal`` round trips for the face convection of a plate modeled
in its plane, on the heat transfer residual.

- The fin: a strip held at one end, insulated at the other, losing heat
  through its faces, against the fin solution with an insulated tip,
  ``T - T_inf = (T_hot - T_inf) cosh(m (L - x)) / cosh(m L)``, on two
  meshes.
- The same strip as a 3D plate with convection on its two faces, its mid
  plane against the 2D strip at two thicknesses: the gap is the through
  thickness variation and shrinks with the thickness.
- The builder refuses a face convection on a 3D mesh, without a
  thickness, under plane strain, and with a wrong key set.
"""
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.cli.common import build_fe_problem_from_deck
from cmad.cli.main import main as cmad_main
from cmad.fem.mesh import StructuredHexMesh, StructuredQuadMesh
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType

_K, _H = 10.0, 1.0
_T_HOT, _T_INF = 400.0, 300.0
_LENGTH, _WIDTH = 1.0, 0.1
_FACE_CONVECTION = {"h": _H, "T_inf": _T_INF}


def _thermal(face_convection: dict[str, Any] | None) -> dict[str, Any]:
    thermal: dict[str, Any] = {"conductivity": _K}
    if face_convection is not None:
        thermal["face convection"] = face_convection
    return thermal


def _heat_deck(
        mesh_filename: str, out_dir: str, thermal: dict[str, Any],
        thickness: float | None = None,
        sections: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A heat transfer input file held at ``_T_HOT`` on ``x = 0``, the
    other boundaries insulated unless ``sections`` adds a condition."""
    discretization: dict[str, Any] = {
        "mesh file": mesh_filename, "num steps": 1, "step size": 1.0,
    }
    if thickness is not None:
        discretization["thickness"] = thickness
    return {
        "problem": {"type": "fe"},
        "discretization": discretization,
        "residuals": {
            "global residual": {"type": "heat_transfer"},
            "local residual": {
                "type": "conduction",
                "materials": {"all": {"thermal": thermal}},
            },
        },
        "dirichlet bcs": {"expression": {
            "hot": ["energy balance", 0, "xmin_sides", _T_HOT],
        }},
        **(sections or {}),
        "output": {
            "path": out_dir,
            "exodus filename": "primal.exo",
            "global residual": ["T"],
        },
    }


def _thermomechanics_deck(
        mesh_filename: str, out_dir: str, thermal: dict[str, Any],
        thickness: float, def_type: str,
) -> dict[str, Any]:
    """An elastic thermomechanics input file with the given def_type."""
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": mesh_filename, "thickness": thickness,
            "num steps": 1, "step size": 1.0,
        },
        "residuals": {
            "global residual": {"type": "thermomechanics", "def_type": def_type},
            "local residual": {
                "type": "elastic",
                "materials": {"all": {
                    "elastic": {"E": 200.0e3, "nu": 0.3}, "thermal": thermal,
                }},
            },
        },
        "dirichlet bcs": {"expression": {
            "hot": ["energy balance", 0, "xmin_sides", _T_HOT],
            "sym_x": ["equilibrium", 0, "xmin_sides", 0.0],
        }},
    }


def _write(tmp: Path, mesh: Any, deck: dict[str, Any]) -> Path:
    with ExodusWriter(str(tmp / "mesh.exo"), mesh):
        pass
    deck["discretization"]["mesh file"] = str(tmp / "mesh.exo")
    deck_path = tmp / "deck.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
    return deck_path


def _run(tmp: Path, mesh: Any, deck: dict[str, Any]) -> np.ndarray:
    """Run ``cmad primal`` and return the nodal temperatures."""
    deck_path = _write(tmp, mesh, deck)
    assert cmad_main(["primal", str(deck_path)]) == 0
    results = read_results(
        tmp / "out" / "primal.exo",
        nodal_field_specs=[FieldSpec("T", VarType.SCALAR)])
    return results.nodal["T"][-1].reshape(-1)


def _fin(x: np.ndarray, thickness: float) -> np.ndarray:
    """The fin with an insulated tip: ``T - T_inf = (T_hot - T_inf) cosh(m
    (L - x)) / cosh(m L)``, ``m^2 = 2 h / (k t)``."""
    m = np.sqrt(2.0 * _H / (_K * thickness))
    return _T_INF + (_T_HOT - _T_INF) * np.cosh(m * (_LENGTH - x)) \
        / np.cosh(m * _LENGTH)


def _strip(tmp: Path, nx: int, thickness: float) -> tuple[np.ndarray, np.ndarray]:
    """The 2D strip's nodal ``x`` and ``T`` along ``y = 0``, in ``x`` order."""
    mesh = StructuredQuadMesh((_LENGTH, _WIDTH), (nx, 1))
    T = _run(tmp, mesh, _heat_deck(
        "", str(tmp / "out"), _thermal(_FACE_CONVECTION), thickness))
    on_edge = np.isclose(mesh.nodes[:, 1], 0.0)
    order = np.argsort(mesh.nodes[on_edge, 0])
    return mesh.nodes[on_edge, 0][order], T[on_edge][order]


def _plate_mid_plane(
        tmp: Path, nx: int, thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The 3D plate's nodal ``x`` and ``T`` on its mid plane along ``y =
    0``, in ``x`` order, with convection on its two faces."""
    mesh = StructuredHexMesh((_LENGTH, _WIDTH, thickness), (nx, 1, 4))
    convection = {"convection bcs": {"expression": {
        "top": ["energy balance", "zmax_sides", _H, _T_INF],
        "bottom": ["energy balance", "zmin_sides", _H, _T_INF],
    }}}
    T = _run(tmp, mesh, _heat_deck(
        "", str(tmp / "out"), _thermal(None), sections=convection))
    mid = np.isclose(mesh.nodes[:, 2], thickness / 2.0) \
        & np.isclose(mesh.nodes[:, 1], 0.0)
    order = np.argsort(mesh.nodes[mid, 0])
    return mesh.nodes[mid, 0][order], T[mid][order]


class TestFin(unittest.TestCase):

    def test_converges_to_the_exact_solution(self) -> None:
        # The face loss makes the temperature a cosh, which is not linear
        # between the nodes, so the nodal error is second order in the
        # mesh size: the ratio between the two meshes says the computed
        # temperature converges to the fin, and the fine error is under a
        # tenth of a kelvin on a 100 K scale.
        thickness = 0.1
        errors = []
        for nx in (8, 16):
            with tempfile.TemporaryDirectory() as tmpdir:
                x, T = _strip(Path(tmpdir), nx, thickness)
            errors.append(float(np.max(np.abs(T - _fin(x, thickness)))))
        coarse, fine = errors
        self.assertGreaterEqual(coarse / fine, 3.5, f"errors {errors}")
        self.assertLess(fine, 0.1, f"fine error {fine}")


class TestPlate(unittest.TestCase):

    def test_mid_plane_approaches_the_strip_as_the_thickness_shrinks(
            self) -> None:
        # The strip has no through thickness variation; the plate's mid
        # plane sits above its faces by half the Biot number h (t / 2) / k
        # times the excess temperature, so the gap between the two is
        # under the Biot number times that excess and halves with the
        # thickness.
        nx = 16
        gaps: dict[float, float] = {}
        for thickness in (0.1, 0.05):
            with tempfile.TemporaryDirectory() as tmpdir:
                x_2d, T_2d = _strip(Path(tmpdir), nx, thickness)
            with tempfile.TemporaryDirectory() as tmpdir:
                x_3d, T_3d = _plate_mid_plane(Path(tmpdir), nx, thickness)
            np.testing.assert_allclose(x_3d, x_2d)
            gaps[thickness] = float(np.max(np.abs(T_3d - T_2d)))
            biot = _H * (thickness / 2.0) / _K
            self.assertLess(
                gaps[thickness], biot * (_T_HOT - _T_INF), f"gaps {gaps}")
        self.assertLess(gaps[0.05] / gaps[0.1], 0.6, f"gaps {gaps}")


class TestBuilder(unittest.TestCase):

    def _build(self, mesh: Any, deck: dict[str, Any]) -> Any:
        with tempfile.TemporaryDirectory() as tmpdir:
            return build_fe_problem_from_deck(
                _write(Path(tmpdir), mesh, deck), "primal")

    def test_a_3d_mesh_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "2D mesh"):
            self._build(
                StructuredHexMesh((_LENGTH, _WIDTH, 0.1), (2, 1, 1)),
                _heat_deck("", "out", _thermal(_FACE_CONVECTION)))

    def test_no_thickness_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "thickness"):
            self._build(
                StructuredQuadMesh((_LENGTH, _WIDTH), (2, 1)),
                _heat_deck("", "out", _thermal(_FACE_CONVECTION)))

    def test_plane_strain_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "plane_stress"):
            self._build(
                StructuredQuadMesh((_LENGTH, _WIDTH), (2, 1)),
                _thermomechanics_deck(
                    "", "out", _thermal(_FACE_CONVECTION), 0.1, "plane_strain"))

    def test_wrong_keys_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "face convection"):
            self._build(
                StructuredQuadMesh((_LENGTH, _WIDTH), (2, 1)),
                _heat_deck("", "out", _thermal({"h": _H}), 0.1))


if __name__ == "__main__":
    unittest.main()
