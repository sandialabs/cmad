"""The temperature match against hand built fields on a unit square of
quads: its value is the closed form of the squared mismatch over the
squared measured change from the initial frame, a shift of every
temperature by a constant leaves it unchanged, and a term without a
calibration data archive fails the schema."""
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from cmad.cli.common import build_fe_problem_from_deck, load_fe_input
from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.mesh import StructuredQuadMesh
from cmad.io.calibration_data import CalibrationData
from cmad.models.global_fields import StepTime
from cmad.qois.fe_temperature_match import FETemperatureMatch
from tests.cli.test_prescribed_fields import _write_mesh_and_deck
from tests.cli.test_primal_fe_thermomechanics_roundtrip import (
    _THERMAL,
    _heat_deck,
)

_T_SCHEDULE = [0.0, 1.0]
_T_0 = 300.0


def _write_archive(path: Path, mesh: Any, rows: Any) -> None:
    """An archive on every node with one temperature row per schedule
    time and the whole mesh as the region of interest."""
    num_nodes = mesh.nodes.shape[0]
    CalibrationData(
        times=np.asarray(_T_SCHEDULE),
        frame_ids=np.arange(2, dtype=np.intp),
        load=np.zeros(2),
        node_ids=np.arange(num_nodes, dtype=np.intp),
        sidesets={},
        displacement=np.zeros((2, num_nodes, 2)),
        mesh_file="mesh.exo",
        mesh_num_nodes=num_nodes,
        roi={"elements": np.arange(mesh.connectivity.shape[0], dtype=np.intp)},
        temperature=np.asarray(rows, dtype=np.float64),
    ).write(path)


def _deck(out_dir: str) -> dict[str, Any]:
    return _heat_deck(
        "", out_dir, _THERMAL,
        {"cold_x_min": ["energy balance", 0, "xmin_sides", _T_0]})


class TestTemperatureMatch(unittest.TestCase):

    def _value(self, tmp: Path, mesh: Any, a: float, c: float,
               shift: float) -> float:
        """The match of the field ``T_0 + shift + a x`` against the data
        ``T_0 + shift + c`` measured from the initial frame ``T_0 +
        shift``."""
        num_nodes = mesh.nodes.shape[0]
        archive = tmp / "T.npz"
        _write_archive(archive, mesh, np.stack([
            np.full(num_nodes, _T_0 + shift),
            np.full(num_nodes, _T_0 + shift + c)]))
        bundle = build_fe_problem_from_deck(
            _write_mesh_and_deck(tmp, mesh, _deck(str(tmp / "out"))), "primal")
        fe_problem = bundle.fe_problem
        qoi = FETemperatureMatch.from_deck(
            {"name": "fe_temperature_match",
             "calibration_data_file": str(archive)},
            fe_problem, _T_SCHEDULE)
        closure = qoi.step_contribution(
            params_by_block_from_models(fe_problem), fe_problem.kernel_arrays)
        U = _T_0 + shift + a * mesh.nodes[:, 0]
        return float(closure(U, U, {}, {}, StepTime(1.0, 0.0)))

    def test_closed_form_and_shift_invariance(self) -> None:
        mesh = StructuredQuadMesh((1.0, 1.0), (2, 2))
        a, c = 3.0, 2.0
        # the field a x against the uniform change c over the unit square
        expected = (a ** 2 / 3.0 - a * c + c ** 2) / c ** 2
        with tempfile.TemporaryDirectory() as tmpdir:
            value = self._value(Path(tmpdir), mesh, a, c, 0.0)
            shifted = self._value(Path(tmpdir), mesh, a, c, 50.0)
        print(f"temperature match {value:.15g}, expected {expected:.15g}, "
              f"shifted by 50 K {shifted:.15g}")
        self.assertAlmostEqual(value, expected, places=12)
        self.assertAlmostEqual(shifted, value, places=12)

    def test_without_an_archive_fails_the_schema(self) -> None:
        mesh = StructuredQuadMesh((1.0, 1.0), (2, 2))
        deck = _deck("out")
        deck["qoi"] = {"name": "fe_temperature_match", "data_file": "T.npy"}
        with tempfile.TemporaryDirectory() as tmpdir, \
                self.assertRaisesRegex(ValueError, "calibration_data_file"):
            load_fe_input(
                _write_mesh_and_deck(Path(tmpdir), mesh, deck), "objective")


if __name__ == "__main__":
    unittest.main()
