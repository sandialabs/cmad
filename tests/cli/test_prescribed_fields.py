"""A prescribed field: data at every node over time that the models read
beside the unknowns, never solved for.

- The constrained bar heated by a ramp: the mechanics residual with the
  ramp as a prescribed temperature against the thermomechanics residual
  with the ramp as boundary temperatures, the displacement to round-off
  and the same Newton iterations per step.
- The one degree slab of the thermomechanics round trip: the heat transfer
  run's temperature history, written into a calibration data archive on
  every node, drives the free Elastic body on the mechanics residual and
  gives the coupled run's displacement.
- The builder refuses a prescribed field that is a solved field, an
  archive missing a node, and a schedule outside the archive's times.
"""
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.cli.common import build_fe_problem_from_deck
from cmad.fem.driver import fe_quasistatic_drive
from cmad.fem.mesh import StructuredHexMesh
from cmad.io.calibration_data import CalibrationData
from cmad.io.exodus import ExodusWriter
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType
from tests.cli.test_primal_fe_thermomechanics_roundtrip import (
    _ALPHA,
    _E,
    _NU,
    _SLAB_STEPS,
    _SLAB_T_FINAL,
    _T_REF,
    _constrained_bar,
    _deck,
    _material,
    _run,
    _slab_deck,
    _symmetry,
)

_RAMP = f"{_T_REF} + 200.0 * t"
_ELASTIC = {
    "elastic": {"E": _E, "nu": _NU},
    "thermal expansion": {"alpha": _ALPHA, "reference temperature": _T_REF},
}


def _mechanics_deck(
        mesh_filename: str, out_dir: str, dirichlet: dict[str, Any],
        prescribed: dict[str, Any], num_steps: int, step_size: float,
) -> dict[str, Any]:
    """The Elastic body on the mechanics residual with a prescribed
    temperature."""
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": mesh_filename,
            "num steps": num_steps,
            "step size": step_size,
        },
        "residuals": {
            "global residual": {"type": "mechanics", "def_type": "full_3d"},
            "local residual": {
                "type": "elastic",
                "reference temperature": _T_REF,
                "materials": {"all": _ELASTIC},
            },
        },
        "dirichlet bcs": {"expression": dirichlet},
        "prescribed fields": prescribed,
        "output": {
            "path": out_dir,
            "exodus filename": "primal.exo",
            "global residual": ["u"],
            "local residual": {"all": ["cauchy"]},
        },
    }


def _fixed_bar() -> dict[str, Any]:
    """The displacement conditions of the constrained bar."""
    return {
        **_symmetry(3),
        "fixed_x_max": ["equilibrium", 0, "xmax_sides", "0.0"],
    }


def _write_mesh_and_deck(tmp: Path, mesh: Any, deck: dict[str, Any]) -> Path:
    with ExodusWriter(str(tmp / "mesh.exo"), mesh):
        pass
    deck["discretization"]["mesh file"] = str(tmp / "mesh.exo")
    deck_path = tmp / "deck.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
    return deck_path


def _drive_deck(tmp: Path, mesh: Any, deck: dict[str, Any]) -> Any:
    """Build the problem from ``deck`` and run its time loop."""
    bundle = build_fe_problem_from_deck(
        _write_mesh_and_deck(tmp, mesh, deck), "primal")
    return fe_quasistatic_drive(
        bundle.fe_problem, bundle.t_schedule.tolist(), U_init=bundle.U_init,
        linear_solver_settings=bundle.resolved["linear solver"])


def _write_archive(
        path: Path, times: Any, temperature: Any, node_ids: Any,
        mesh_num_nodes: int,
) -> None:
    """A calibration data archive holding a temperature history and a zero
    displacement."""
    num_frames = len(times)
    CalibrationData(
        times=np.asarray(times, dtype=np.float64),
        frame_ids=np.arange(num_frames, dtype=np.intp),
        load=np.zeros(num_frames),
        node_ids=np.asarray(node_ids, dtype=np.intp),
        sidesets={},
        displacement=np.zeros((num_frames, len(node_ids), 3)),
        mesh_file="mesh.exo",
        mesh_num_nodes=mesh_num_nodes,
        temperature=np.asarray(temperature, dtype=np.float64),
    ).write(path)


class TestConstrainedBar(unittest.TestCase):
    """The ramp as a prescribed temperature on the mechanics residual
    against the ramp as boundary temperatures on the thermomechanics
    residual; the temperature is uniform at every step and exact in both."""

    def test_displacement_and_newton_iterations(self) -> None:
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (4, 1, 1))
        num_steps = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_tm, _, status_tm = _drive_deck(tmp, mesh, _deck(
                "", str(tmp / "out"), "elastic", _material(),
                _constrained_bar(_RAMP), num_steps=num_steps))
            state_m, _, status_m = _drive_deck(tmp, mesh, _mechanics_deck(
                "", str(tmp / "out"), _fixed_bar(),
                {"T": {"expression": _RAMP}}, num_steps, 1.0 / num_steps))
        self.assertTrue(status_tm.converged and status_m.converged)
        self.assertEqual(status_m.iters_per_step, status_tm.iters_per_step)
        n_u = state_m.U_at(0).shape[0]
        for k in range(1, num_steps + 1):
            np.testing.assert_allclose(
                state_m.U_at(k), state_tm.U_at(k)[:n_u], rtol=1e-10, atol=1e-16)
        # the bar fixed along x expands laterally
        u = state_m.U_at(num_steps).reshape(-1, 3)
        lateral = (1.0 + _NU) * _ALPHA * 200.0
        np.testing.assert_allclose(
            u[:, 1], lateral * mesh.nodes[:, 1], rtol=1e-9, atol=1e-15)


class TestSlabFromAnArchive(unittest.TestCase):

    def test_displacement_is_the_coupled_runs(self) -> None:
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (32, 1, 1))
        num_nodes = mesh.nodes.shape[0]
        nodal_u = [FieldSpec("u", VarType.VECTOR)]
        cauchy = [FieldSpec("cauchy", VarType.SYM_TENSOR)]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            heat = _run(
                tmp, mesh, lambda m, o: _slab_deck(m, o, False),
                [FieldSpec("T", VarType.SCALAR)],
                [FieldSpec("heat flux", VarType.VECTOR)])
            archive = tmp / "slab_T.npz"
            _write_archive(
                archive, np.linspace(0.0, _SLAB_T_FINAL, _SLAB_STEPS + 1),
                np.asarray(heat.nodal["T"]).reshape(_SLAB_STEPS + 1, -1),
                np.arange(num_nodes), num_nodes)
            coupled = _run(
                tmp, mesh, lambda m, o: _slab_deck(m, o, True), nodal_u, cauchy)
            measured = _run(
                tmp, mesh,
                lambda m, o: _mechanics_deck(
                    m, o, _symmetry(3), {"T": {"data file": str(archive)}},
                    _SLAB_STEPS, _SLAB_T_FINAL / _SLAB_STEPS),
                nodal_u, cauchy)
        u_end = measured.nodal["u"][-1]
        self.assertGreater(u_end[mesh.nodes[:, 0] == 1.0, 0].min(), 0.0)
        for k in range(_SLAB_STEPS + 1):
            np.testing.assert_allclose(
                measured.nodal["u"][k], coupled.nodal["u"][k],
                rtol=1e-9, atol=1e-16)


class TestBuilderRefusals(unittest.TestCase):

    def _build(self, tmp: Path, deck: dict[str, Any]) -> Any:
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 1, 1))
        return build_fe_problem_from_deck(
            _write_mesh_and_deck(tmp, mesh, deck), "primal")

    def test_a_solved_field_is_refused(self) -> None:
        deck = _mechanics_deck(
            "", "out", _fixed_bar(), {"u": {"expression": "0.0"}}, 1, 1.0)
        with tempfile.TemporaryDirectory() as tmpdir, \
                self.assertRaisesRegex(ValueError, "solved field"):
            self._build(Path(tmpdir), deck)

    def test_an_archive_missing_a_node_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
                self.assertRaisesRegex(ValueError, "not in the calibration data"):
            archive = Path(tmpdir) / "T.npz"
            _write_archive(
                archive, [0.0, 1.0], np.full((2, 11), _T_REF),
                np.arange(1, 12), 12)
            self._build(Path(tmpdir), _mechanics_deck(
                "", "out", _fixed_bar(), {"T": {"data file": str(archive)}},
                1, 1.0))

    def test_a_schedule_outside_the_archive_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
                self.assertRaisesRegex(ValueError, "the schedule spans"):
            archive = Path(tmpdir) / "T.npz"
            _write_archive(
                archive, [0.0, 0.5], np.full((2, 12), _T_REF), np.arange(12), 12)
            self._build(Path(tmpdir), _mechanics_deck(
                "", "out", _fixed_bar(), {"T": {"data file": str(archive)}},
                1, 1.0))


if __name__ == "__main__":
    unittest.main()
