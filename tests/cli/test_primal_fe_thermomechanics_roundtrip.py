"""``cmad primal`` round trips on the thermomechanics global residual.

Each problem has a uniform solution, so a linear basis reproduces it to
round off.

- A bar fixed at both ends and heated: the stress is ``-E alpha dT``, and
  in the mixed formulation the pressure is ``E alpha dT / 3``.
- The same bar heated into yield with Johnson-Cook, compared with the
  material point under the same history.
- A be_bar cube and a plane stress square expanding freely.
- The builder: the temperature field starts at the reference temperature,
  and a material without a thermal subtree is refused.
"""
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.cli.common import build_fe_problem_from_deck
from cmad.cli.main import main as cmad_main
from cmad.fem.dof import dof_physical_coords
from cmad.fem.mesh import StructuredHexMesh, StructuredQuadMesh
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.params_builder import build_parameters
from cmad.io.results import FieldSpec
from cmad.models.deformation_types import DefType
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.var_types import VarType
from tests.models.test_rate_dependent_uniaxial import JOHNSON_COOK
from tests.models.test_thermal_expansion import _drive

_E, _NU, _ALPHA = 200e3, 0.3, 1.2e-5
_T_REF, _T_HOT = 300.0, 500.0
_DT = _T_HOT - _T_REF
_THERMAL = {"conductivity": 16.0}
_J2_VOCE = {
    "effective stress": {"J2": {}},
    "flow stress": {"initial yield": {"Y": 200.0},
                    "hardening": {"voce": {"S": 200.0, "D": 20.0}}},
}


def _material(plastic: dict[str, Any] | None = None) -> dict[str, Any]:
    material: dict[str, Any] = {
        "elastic": {"E": _E, "nu": _NU},
        "thermal expansion": {"alpha": _ALPHA, "reference temperature": _T_REF},
        "thermal": _THERMAL,
    }
    if plastic is not None:
        material["plastic"] = plastic
    return material


def _deck(
        mesh_filename: str, out_dir: str, model_type: str,
        material: dict[str, Any], dirichlet: dict[str, Any],
        num_steps: int = 1, def_type: str = "full_3d", mixed: bool = False,
        outputs: tuple[str, ...] = ("cauchy",),
        reference_temperature: float = _T_REF,
) -> dict[str, Any]:
    gr: dict[str, Any] = {"type": "thermomechanics", "def_type": def_type}
    if mixed:
        gr["mixed"] = True
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": mesh_filename,
            "num steps": num_steps,
            "step size": 1.0 / num_steps,
        },
        "residuals": {
            "global residual": gr,
            "local residual": {
                "type": model_type,
                "reference temperature": reference_temperature,
                "materials": {"all": material},
            },
        },
        "dirichlet bcs": {"expression": dirichlet},
        "output": {
            "path": out_dir,
            "exodus filename": "primal.exo",
            "global residual": ["u", "p", "T"] if mixed else ["u", "T"],
            "local residual": {"all": list(outputs)},
        },
    }


def _symmetry(ndims: int) -> dict[str, Any]:
    """``u`` fixed normal to the three (or two) minimum faces."""
    faces = ("xmin_sides", "ymin_sides", "zmin_sides")[:ndims]
    return {
        f"sym_{d}": ["equilibrium", d, face, "0.0"]
        for d, face in enumerate(faces)
    }


def _hot_x_faces(T_expression: Any) -> dict[str, Any]:
    return {
        "hot_x_min": ["energy balance", 0, "xmin_sides", T_expression],
        "hot_x_max": ["energy balance", 0, "xmax_sides", T_expression],
    }


def _constrained_bar(T_expression: Any) -> dict[str, Any]:
    return {
        **_symmetry(3),
        "fixed_x_max": ["equilibrium", 0, "xmax_sides", "0.0"],
        **_hot_x_faces(T_expression),
    }


def _run(tmp: Path, mesh: Any, deck_fn: Any, nodal: list[FieldSpec],
         element: list[FieldSpec]) -> Any:
    """Write the mesh and the input file, run ``cmad primal``, read the
    results back."""
    with ExodusWriter(str(tmp / "mesh.exo"), mesh):
        pass
    deck_path = tmp / "deck.yaml"
    deck_path.write_text(yaml.safe_dump(
        deck_fn(str(tmp / "mesh.exo"), str(tmp / "out")), sort_keys=False))
    assert cmad_main(["primal", str(deck_path)]) == 0
    return read_results(
        tmp / "out" / "primal.exo", nodal_field_specs=nodal,
        element_field_specs={"all": element})


class TestConstrainedBar(unittest.TestCase):

    def _check(self, mixed: bool) -> None:
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (4, 1, 1))
        nodal = [FieldSpec("u", VarType.VECTOR), FieldSpec("T", VarType.SCALAR)]
        if mixed:
            nodal.append(FieldSpec("p", VarType.SCALAR))
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run(
                Path(tmpdir), mesh,
                lambda m, o: _deck(
                    m, o, "elastic", _material(), _constrained_bar(_T_HOT),
                    mixed=mixed, outputs=("cauchy", "heat flux")),
                nodal,
                [FieldSpec("cauchy", VarType.SYM_TENSOR),
                 FieldSpec("heat flux", VarType.VECTOR)])
        T = results.nodal["T"]
        np.testing.assert_array_equal(T[0].reshape(-1), _T_REF)
        np.testing.assert_allclose(T[-1].reshape(-1), _T_HOT, rtol=1e-12)
        u = results.nodal["u"][-1]
        lateral = (1.0 + _NU) * _ALPHA * _DT
        np.testing.assert_allclose(u[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            u[:, 1], lateral * mesh.nodes[:, 1], rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(
            u[:, 2], lateral * mesh.nodes[:, 2], rtol=1e-9, atol=1e-15)
        cauchy = results.element["all"]["cauchy"][-1]
        np.testing.assert_allclose(cauchy[:, 0], -_E * _ALPHA * _DT, rtol=1e-9)
        np.testing.assert_allclose(cauchy[:, 1:], 0.0, atol=1e-7)
        np.testing.assert_allclose(
            results.element["all"]["heat flux"][-1], 0.0, atol=1e-8)
        if mixed:
            np.testing.assert_allclose(
                results.nodal["p"][-1].reshape(-1), _E * _ALPHA * _DT / 3.0,
                rtol=1e-9)

    def test_elastic(self) -> None:
        self._check(mixed=False)

    def test_elastic_mixed(self) -> None:
        self._check(mixed=True)

    def test_johnson_cook_heated_into_yield_against_the_material_point(
            self) -> None:
        num_steps, T_end = 10, 600.0
        plastic = {"effective stress": {"J2": {}}, "flow stress": JOHNSON_COOK}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run(
                Path(tmpdir), StructuredHexMesh((1.0, 1.0, 1.0), (4, 1, 1)),
                lambda m, o: _deck(
                    m, o, "small_elastic_plastic", _material(plastic),
                    _constrained_bar(f"{_T_REF} + {T_end - _T_REF} * t"),
                    num_steps=num_steps, outputs=("cauchy", "alpha")),
                [FieldSpec("T", VarType.SCALAR)],
                [FieldSpec("cauchy", VarType.SYM_TENSOR),
                 FieldSpec("alpha", VarType.SCALAR)])
        # the same history at a point under uniaxial stress with the axial
        # strain held at zero
        model = SmallElasticPlastic(
            build_parameters(_material(plastic)), DefType.UNIAXIAL_STRESS,
            reference_temperature=_T_REF)
        T = np.linspace(_T_REF, T_end, num_steps + 1)
        cauchy, xi = _drive(
            model, np.ones((1, 1, num_steps + 1)), T,
            np.linspace(0.0, 1.0, num_steps + 1))
        self.assertGreater(float(xi[1][0]), 0.0)
        cauchy_fe = results.element["all"]["cauchy"][-1]
        np.testing.assert_allclose(cauchy_fe[:, 0], cauchy[0, 0, -1], rtol=1e-8)
        np.testing.assert_allclose(cauchy_fe[:, 1:], 0.0, atol=1e-6)
        np.testing.assert_allclose(
            results.element["all"]["alpha"][-1].reshape(-1), float(xi[1][0]),
            rtol=1e-8)


class TestFreeExpansion(unittest.TestCase):

    def test_be_bar_cube(self) -> None:
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run(
                Path(tmpdir), mesh,
                lambda m, o: _deck(
                    m, o, "be_bar_elastic_plastic", _material(_J2_VOCE),
                    {**_symmetry(3), **_hot_x_faces(_T_HOT)},
                    outputs=("cauchy", "alpha")),
                [FieldSpec("u", VarType.VECTOR), FieldSpec("T", VarType.SCALAR)],
                [FieldSpec("cauchy", VarType.SYM_TENSOR),
                 FieldSpec("alpha", VarType.SCALAR)])
        np.testing.assert_allclose(
            results.nodal["u"][-1], _ALPHA * _DT * mesh.nodes, rtol=1e-9,
            atol=1e-15)
        np.testing.assert_allclose(
            results.element["all"]["cauchy"][-1], 0.0, atol=1e-7)
        np.testing.assert_allclose(
            results.element["all"]["alpha"][-1], 0.0, atol=1e-14)

    def test_plane_stress_square(self) -> None:
        mesh = StructuredQuadMesh((1.0, 1.0), (2, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run(
                Path(tmpdir), mesh,
                lambda m, o: _deck(
                    m, o, "elastic", _material(),
                    {**_symmetry(2), **_hot_x_faces(_T_HOT)},
                    def_type="plane_stress"),
                [FieldSpec("u", VarType.VECTOR), FieldSpec("T", VarType.SCALAR)],
                [FieldSpec("cauchy", VarType.SYM_TENSOR)])
        np.testing.assert_allclose(
            results.nodal["u"][-1], _ALPHA * _DT * mesh.nodes, rtol=1e-9,
            atol=1e-15)
        np.testing.assert_allclose(
            results.element["all"]["cauchy"][-1], 0.0, atol=1e-7)


class TestBuilder(unittest.TestCase):

    def _bundle(self, tmp: Path, deck: dict[str, Any]) -> Any:
        with ExodusWriter(str(tmp / "mesh.exo"), StructuredHexMesh(
                (1.0, 1.0, 1.0), (2, 1, 1))):
            pass
        deck["discretization"]["mesh file"] = str(tmp / "mesh.exo")
        deck_path = tmp / "deck.yaml"
        deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
        return build_fe_problem_from_deck(deck_path, "primal")

    def test_temperature_starts_at_the_reference_temperature(self) -> None:
        reference_temperature = 350.0
        thermomechanics = _deck(
            "", "out", "elastic", _material(), _constrained_bar(_T_HOT),
            reference_temperature=reference_temperature)
        heat = {
            "problem": {"type": "fe"},
            "discretization": {"num steps": 1, "step size": 1.0},
            "residuals": {
                "global residual": {"type": "heat_transfer"},
                "local residual": {
                    "type": "conduction",
                    "reference temperature": reference_temperature,
                    "materials": {"all": {"thermal": _THERMAL}},
                },
            },
            "dirichlet bcs": {"expression": _hot_x_faces(_T_HOT)},
        }
        for deck in (thermomechanics, heat):
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle = self._bundle(Path(tmpdir), deck)
            fe_problem = bundle.fe_problem
            _, eq = dof_physical_coords(fe_problem.mesh, fe_problem.dof_map, "T")
            assert bundle.U_init is not None
            np.testing.assert_array_equal(
                bundle.U_init[eq[:, 0]], reference_temperature)
            others = np.setdiff1d(
                np.arange(bundle.U_init.shape[0]), eq[:, 0])
            np.testing.assert_array_equal(bundle.U_init[others], 0.0)

    def test_material_without_thermal_raises(self) -> None:
        material = _material()
        del material["thermal"]
        with tempfile.TemporaryDirectory() as tmpdir, \
                self.assertRaisesRegex(ValueError, "thermal subtree"):
            self._bundle(Path(tmpdir), _deck(
                "", "out", "elastic", material, _constrained_bar(_T_HOT)))


if __name__ == "__main__":
    unittest.main()
