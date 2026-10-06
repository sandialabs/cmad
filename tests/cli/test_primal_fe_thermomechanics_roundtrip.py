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
- The unit slab of ``test_heat_transient.py`` on the thermomechanics
  residual with a free Elastic body: the temperature is the heat transfer
  run's to round off and matches the Fourier series, and the body moves.
- One insulated hex of the small strain model with Johnson-Cook and
  ``taylor-quinney``, pulled to five percent strain: the temperature
  follows the uniaxial return map with the temperature raised per step by
  the plastic work, and ``rho c dT`` equals ``beta`` times the plastic work
  summed from the written stress and plastic strain.
- Steady conduction through a neohookean body clamped at one end and
  pulled at the other, on tets and on plane strain triangles, against the
  heat transfer residual on the deformed mesh.
- The plastic heating of the finite deformation models on the insulated
  hex against the small strain model at two small strains.
"""
import dataclasses
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import brentq

from cmad.cli.common import build_fe_problem_from_deck
from cmad.cli.main import main as cmad_main
from cmad.fem.dof import dof_physical_coords
from cmad.fem.mesh import (
    StructuredHexMesh,
    StructuredQuadMesh,
    hex_to_tet_split,
    quad_to_tri_split,
)
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.params_builder import build_parameters
from cmad.io.results import FieldSpec
from cmad.models.deformation_types import DefType
from cmad.models.flow_stress import POWER_LAW_OFFSET
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.var_types import VarType
from tests.fem.test_heat_transient import slab_series
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
        local_residual: dict[str, Any] | None = None,
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
                **(local_residual or {}),
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


def _heat_deck(
        mesh_filename: str, out_dir: str, thermal: dict[str, Any],
        dirichlet: dict[str, Any], num_steps: int = 1,
) -> dict[str, Any]:
    """A heat transfer input file with the given material and boundary
    temperatures."""
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": mesh_filename, "num steps": num_steps,
            "step size": 1.0 / num_steps,
        },
        "residuals": {
            "global residual": {"type": "heat_transfer"},
            "local residual": {
                "type": "conduction",
                "materials": {"all": {"thermal": thermal}},
            },
        },
        "dirichlet bcs": {"expression": dirichlet},
        "output": {
            "path": out_dir,
            "exodus filename": "primal.exo",
            "global residual": ["T"],
            "local residual": {"all": ["heat flux"]},
        },
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


_SLAB_THERMAL = {"conductivity": 1.0, "density": 1.0, "specific heat": 1.0}
_SLAB_STEPS, _SLAB_T_FINAL = 100, 0.1


def _slab_deck(mesh_filename: str, out_dir: str,
               thermomechanics: bool) -> dict[str, Any]:
    """The slab of ``test_heat_transient.py`` as an input file, shifted by
    the reference temperature: one degree above it at the start, both ends
    held at it, on the thermomechanics residual with a free Elastic body
    or on the heat transfer residual."""
    if thermomechanics:
        material = {
            "elastic": {"E": _E, "nu": _NU},
            "thermal expansion": {"alpha": _ALPHA, "reference temperature": _T_REF},
            "thermal": _SLAB_THERMAL,
        }
        deck = _deck(
            mesh_filename, out_dir, "elastic", material,
            {**_symmetry(3), **_hot_x_faces(_T_REF)}, num_steps=_SLAB_STEPS)
    else:
        deck = _heat_deck(
            mesh_filename, out_dir, _SLAB_THERMAL, _hot_x_faces(_T_REF),
            num_steps=_SLAB_STEPS)
    deck["discretization"]["step size"] = _SLAB_T_FINAL / _SLAB_STEPS
    deck["initial conditions"] = {"T": _T_REF + 1.0}
    return deck


class TestSlab(unittest.TestCase):

    def test_temperature_is_the_heat_transfer_runs(self) -> None:
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (32, 1, 1))
        T = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run(
                Path(tmpdir), mesh, lambda m, o: _slab_deck(m, o, True),
                [FieldSpec("u", VarType.VECTOR), FieldSpec("T", VarType.SCALAR)],
                [FieldSpec("cauchy", VarType.SYM_TENSOR)])
            T["thermomechanics"] = results.nodal["T"][-1].reshape(-1)
            u = results.nodal["u"][-1]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run(
                Path(tmpdir), mesh, lambda m, o: _slab_deck(m, o, False),
                [FieldSpec("T", VarType.SCALAR)],
                [FieldSpec("heat flux", VarType.VECTOR)])
            T["heat transfer"] = results.nodal["T"][-1].reshape(-1)
        np.testing.assert_allclose(
            T["thermomechanics"], T["heat transfer"], rtol=1e-12)
        # the discretization error of the coarse slab in test_heat_transient
        self.assertLess(
            np.abs(T["heat transfer"] - _T_REF
                   - slab_series(mesh.nodes[:, 0], _SLAB_T_FINAL)).max(), 3e-3)
        # the body is undeformed at the reference temperature and is at
        # most one degree above it, so the far face moves out, and no
        # farther than alpha times the length
        u_end = u[mesh.nodes[:, 0] == 1.0, 0]
        self.assertGreater(u_end.min(), 0.0)
        self.assertLess(u_end.max(), _ALPHA)


_BETA, _RHO_C = 0.9, 3.9
_PULL_STRAIN, _PULL_STEPS = 0.05, 20
_JOHNSON_COOK_PLASTIC = {
    "effective stress": {"J2": {}},
    "flow stress": JOHNSON_COOK, "taylor-quinney": _BETA,
}


def _johnson_cook_flow_stress(alpha: float, rate: float, T: float) -> float:
    """The Johnson-Cook flow stress of ``JOHNSON_COOK`` at a temperature."""
    p = JOHNSON_COOK["johnson_cook"]
    a0 = POWER_LAW_OFFSET
    T_star = max((T - p["reference temperature"])
                 / (p["melt temperature"] - p["reference temperature"]), 0.0)
    return (p["A"] + p["B"] * (alpha + a0) ** p["n"]) \
        * (1.0 + p["C"] * np.log(max(rate / p["reference rate"], 1.0))) \
        * (1.0 - (T_star + a0) ** p["m"])


def _heated_return_map(strains: Any, times: Any, T_0: float) -> Any:
    """The uniaxial return map of ``test_rate_dependent_uniaxial.py`` with
    the thermal strain ``alpha (T - T_0)`` and the temperature raised per
    step by ``rho c (T_n - T_(n-1)) = beta sigma_n delta_gamma_n``, the
    flow stress at the step's temperature.

    Returns ``sigma_11``, ``alpha``, the axial plastic strain, and the
    temperature per step.
    """
    n = len(strains)
    sigma, alpha, eps_p = np.zeros(n), np.zeros(n), np.zeros(n)
    T = np.full(n, T_0)
    for k in range(1, n):
        dt = times[k] - times[k - 1]
        alpha[k], eps_p[k], T[k] = alpha[k - 1], eps_p[k - 1], T[k - 1]
        sigma_tr = _E * (strains[k] - eps_p[k - 1] - _ALPHA * (T[k - 1] - T_0))
        sign = np.sign(sigma_tr)
        if abs(sigma_tr) <= _johnson_cook_flow_stress(
                alpha[k - 1], 0.0, T[k - 1]):
            sigma[k] = sigma_tr
            continue

        def step(dg, k=k, sign=sign):
            """The stress and the temperature at ``dg``, linear in each
            other once ``dg`` is fixed."""
            e = strains[k] - eps_p[k - 1] - dg * sign
            c = _BETA * dg * sign * _E / _RHO_C
            T_n = (T[k - 1] + c * (e + _ALPHA * T_0)) / (1.0 + c * _ALPHA)
            return _E * (e - _ALPHA * (T_n - T_0)), T_n

        def g(dg, k=k, dt=dt):
            sigma_n, T_n = step(dg)
            return abs(sigma_n) - _johnson_cook_flow_stress(
                alpha[k - 1] + dg, dg / dt, T_n)

        dg = brentq(g, 0.0, abs(sigma_tr) / _E, xtol=1e-18, rtol=1e-15)
        sigma[k], T[k] = step(dg)
        alpha[k] = alpha[k - 1] + dg
        eps_p[k] = eps_p[k - 1] + dg * sign
    return sigma, alpha, eps_p, T


def _sym_tensor_contraction(a: Any, b: Any) -> float:
    """``a : b`` summed over steps for symmetric tensors stored as
    ``[xx, xy, xz, yy, yz, zz]``."""
    normal, shear = [0, 3, 5], [1, 2, 4]
    return float(np.sum(a[:, normal] * b[:, normal])
                 + 2.0 * np.sum(a[:, shear] * b[:, shear]))


class TestInsulatedHex(unittest.TestCase):

    def test_plastic_heating_follows_the_return_map(self) -> None:
        material = _material(_JOHNSON_COOK_PLASTIC)
        material["thermal"] = {"conductivity": 16.0, "density": 1.0,
                               "specific heat": _RHO_C}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run(
                Path(tmpdir), StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1)),
                lambda m, o: _deck(
                    m, o, "small_elastic_plastic", material,
                    {**_symmetry(3),
                     "pull_x_max": ["equilibrium", 0, "xmax_sides",
                                    f"{_PULL_STRAIN} * t"]},
                    num_steps=_PULL_STEPS,
                    outputs=("cauchy", "plastic strain", "alpha")),
                [FieldSpec("T", VarType.SCALAR)],
                [FieldSpec("cauchy", VarType.SYM_TENSOR),
                 FieldSpec("plastic strain", VarType.SYM_TENSOR),
                 FieldSpec("alpha", VarType.SCALAR)])
        times = np.linspace(0.0, 1.0, _PULL_STEPS + 1)
        sigma, alpha, _, T = _heated_return_map(
            _PULL_STRAIN * times, times, _T_REF)
        self.assertGreater(T[-1] - _T_REF, 1.0)
        T_fe = np.array([
            results.nodal["T"][k].reshape(-1) for k in range(len(times))])
        np.testing.assert_allclose(
            T_fe - _T_REF, np.broadcast_to((T - _T_REF)[:, None], T_fe.shape),
            rtol=1e-8, atol=1e-10)
        cauchy = np.array([
            results.element["all"]["cauchy"][k][0] for k in range(len(times))])
        np.testing.assert_allclose(cauchy[-1, 0], sigma[-1], rtol=1e-8)
        np.testing.assert_allclose(
            results.element["all"]["alpha"][-1].reshape(-1), alpha[-1], rtol=1e-8)
        # the heat stored equals beta times the plastic work, both from the
        # written fields
        eps_p = np.array([
            results.element["all"]["plastic strain"][k][0]
            for k in range(len(times))])
        work = _sym_tensor_contraction(cauchy[1:], np.diff(eps_p, axis=0))
        np.testing.assert_allclose(
            _RHO_C * (T_fe[-1, 0] - _T_REF), _BETA * work, rtol=1e-8)


def _deformed(mesh: Any, u: Any) -> Any:
    """The mesh with its nodes moved by ``u``."""
    return dataclasses.replace(mesh, nodes=mesh.nodes + u)


class TestConductionOnADeformedBody(unittest.TestCase):
    """Steady conduction between two end temperatures through a neohookean
    body clamped at one end and pulled at the other, against the heat
    transfer residual on the deformed mesh: the same discrete problem when
    ``F`` is constant on each element."""

    def _check(self, mesh: Any, def_type: str, ndims: int) -> None:
        material = {
            "elastic": {"E": _E, "nu": _NU},
            "thermal expansion": {"alpha": _ALPHA, "reference temperature": _T_REF},
            "thermal": _THERMAL,
        }
        ends = {
            "cold_x_min": ["energy balance", 0, "xmin_sides", _T_REF],
            "hot_x_max": ["energy balance", 0, "xmax_sides", _T_HOT],
        }
        clamp = {f"clamp_{d}": ["equilibrium", d, "xmin_sides", 0.0]
                 for d in range(ndims)}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run(
                Path(tmpdir), mesh,
                lambda m, o: _deck(
                    m, o, "elastic", material,
                    {**clamp, "pull_x_max": ["equilibrium", 0, "xmax_sides", 0.3],
                     **ends},
                    def_type=def_type,
                    local_residual={"elastic_stress": "neohookean"}),
                [FieldSpec("u", VarType.VECTOR), FieldSpec("T", VarType.SCALAR)],
                [FieldSpec("cauchy", VarType.SYM_TENSOR)])
        u = results.nodal["u"][-1]
        T = results.nodal["T"][-1].reshape(-1)
        # the deformation is not homogeneous, so the temperature is not
        # linear in the reference coordinate
        self.assertGreater(
            np.abs(T - (_T_REF + _DT * mesh.nodes[:, 0])).max(), 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            results = _run(
                Path(tmpdir), _deformed(mesh, u),
                lambda m, o: _heat_deck(m, o, _THERMAL, ends),
                [FieldSpec("T", VarType.SCALAR)],
                [FieldSpec("heat flux", VarType.VECTOR)])
        np.testing.assert_allclose(
            T, results.nodal["T"][-1].reshape(-1), rtol=1e-9)

    def test_tets(self) -> None:
        self._check(
            hex_to_tet_split(StructuredHexMesh((1.0, 1.0, 1.0), (4, 4, 4))),
            "full_3d", 3)

    def test_plane_strain_triangles(self) -> None:
        self._check(
            quad_to_tri_split(StructuredQuadMesh((1.0, 1.0), (6, 6))),
            "plane_strain", 2)


def _insulated_hex_rise(
        model_type: str, plastic: dict[str, Any], strain: float,
        local_residual: dict[str, Any] | None = None) -> float:
    """The temperature rise of one insulated hex pulled to ``strain`` in
    one second on the thermomechanics residual."""
    material = _material(plastic)
    material["thermal"] = {"conductivity": 16.0, "density": 1.0,
                           "specific heat": _RHO_C}
    with tempfile.TemporaryDirectory() as tmpdir:
        results = _run(
            Path(tmpdir), StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1)),
            lambda m, o: _deck(
                m, o, model_type, material,
                {**_symmetry(3),
                 "pull_x_max": ["equilibrium", 0, "xmax_sides", f"{strain} * t"]},
                num_steps=_PULL_STEPS, local_residual=local_residual),
            [FieldSpec("T", VarType.SCALAR)],
            [FieldSpec("cauchy", VarType.SYM_TENSOR)])
    return float(results.nodal["T"][-1].reshape(-1)[0]) - _T_REF


class TestFiniteModelsOnTheInsulatedHex(unittest.TestCase):
    """The plastic heating of the finite deformation models against the
    small strain model at two small strains: within a few percent, the
    gap shrinking with the strain."""

    def _check(self, model_type: str, plastic: dict[str, Any],
               local_residual: dict[str, Any] | None = None) -> None:
        gaps = []
        for strain in (0.02, 0.01):
            small = _insulated_hex_rise("small_elastic_plastic", plastic, strain)
            finite = _insulated_hex_rise(
                model_type, plastic, strain, local_residual)
            self.assertGreater(small, 0.1)
            gaps.append(abs(finite - small) / small)
        self.assertLess(gaps[0], 0.05)
        self.assertLess(gaps[1], 0.75 * gaps[0])

    def test_rate_model(self) -> None:
        self._check("rate_elastic_plastic", _JOHNSON_COOK_PLASTIC,
                    {"finite deformation": True})

    def test_be_bar(self) -> None:
        self._check("be_bar_elastic_plastic",
                    {**_J2_VOCE, "taylor-quinney": _BETA})


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
