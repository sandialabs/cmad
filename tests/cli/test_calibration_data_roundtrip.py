"""Consumers of a calibration data archive on ``problem.type=fe``.

A truth solve on an elastic cube, ``u_x`` ramped on ``xmax_sides``,
supplies a nodal history and a reaction series; an archive built from
them by hand (frames at the schedule times, the ``zmax_sides`` face as
the region of interest, ``xmax_sides`` as the Dirichlet sideset) then
drives the same problem through the archive paths:

- a field Dirichlet condition read from the archive reproduces the
  expression solution, which checks the node id scatter;
- the displacement and load matches give the same objective from the
  archive as from the plain files;
- with a match subset, solve steps that are not match times score
  nothing: the objective equals the one on the coarse schedule of the
  match times alone (elastic, so the states at those times coincide);
- a match time that is not an archive frame is rejected.
"""
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.cli.main import main as cmad_main
from cmad.fem.mesh import StructuredHexMesh, side_set_nodes
from cmad.io.calibration_data import CalibrationData
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType

KAPPA_TRUTH = 100.0
MU_TRUTH = 50.0
SCHEDULE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COARSE = [0.0, 0.4, 0.8, 1.0]
LOADED = "xmax_sides"
OBSERVED = "zmax_sides"
U_SPEC = [FieldSpec("u", VarType.VECTOR)]


def _deck(
        mesh_path: Path,
        out_dir: Path,
        *,
        times: list[float],
        loaded_face: Path | None,
        kappa: float,
        qoi: dict[str, Any] | None = None,
        exodus_filename: str | None = None,
) -> dict[str, Any]:
    """A cube held on the three min faces and pulled on ``xmax_sides``,
    by the expression ramp or by the field data in ``loaded_face``."""
    dbcs: dict[str, Any] = {
        "expression": {
            "sym_x": ["equilibrium", 0, "xmin_sides", "0.0"],
            "sym_y": ["equilibrium", 1, "ymin_sides", "0.0"],
            "sym_z": ["equilibrium", 2, "zmin_sides", "0.0"],
        },
    }
    if loaded_face is None:
        dbcs["expression"]["load_x"] = ["equilibrium", 0, LOADED, "0.05 * t"]
    else:
        dbcs["field data file"] = str(loaded_face)
        dbcs["field"] = {"load_x": ["equilibrium", 0, LOADED]}
    deck: dict[str, Any] = {
        "problem": {"type": "fe"},
        "discretization": {"mesh file": str(mesh_path), "times": list(times)},
        "residuals": {
            "global residual": {"type": "mechanics", "def_type": "full_3d"},
            "local residual": {
                "type": "elastic",
                "materials": {
                    "all": {"elastic": {"kappa": kappa, "mu": MU_TRUTH}},
                },
            },
        },
        "dirichlet bcs": dbcs,
        "output": {"path": str(out_dir)},
    }
    if exodus_filename is not None:
        deck["output"]["exodus filename"] = exodus_filename
    if qoi is not None:
        deck["qoi"] = qoi
    return deck


def _run(subcommand: str, deck: dict[str, Any], path: Path) -> int:
    path.write_text(yaml.safe_dump(deck, sort_keys=False))
    return cmad_main([subcommand, str(path)])


class TestCalibrationDataRoundTrip(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)
        cls.mesh_path = cls.dir / "cube.exo"
        cls.mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        with ExodusWriter(str(cls.mesh_path), cls.mesh):
            pass

        cls.load_csv = cls.dir / "load.csv"
        code = _run("primal", _deck(
            cls.mesh_path, cls.dir / "truth", times=SCHEDULE,
            loaded_face=None, kappa=KAPPA_TRUTH,
            qoi={
                "name": "fe_load_match", "output_file": str(cls.load_csv),
                "sideset": LOADED, "components": [0],
            },
            exodus_filename="truth.exo",
        ), cls.dir / "truth.yaml")
        assert code == 0
        cls.truth_exo = cls.dir / "truth" / "truth.exo"
        cls.truth_u = np.asarray(
            read_results(cls.truth_exo, nodal_field_specs=U_SPEC).nodal["u"],
        )

        roi_sides = cls.mesh.side_sets[OBSERVED]
        loaded_nodes = side_set_nodes(cls.mesh, LOADED)
        node_ids = np.union1d(
            side_set_nodes(cls.mesh, OBSERVED), loaded_nodes,
        ).astype(np.intp)
        cls.archive = cls.dir / "cube_calibration_data.npz"
        CalibrationData(
            times=np.array(SCHEDULE),
            frame_ids=np.arange(-1, len(SCHEDULE) - 1, dtype=np.intp),
            load=np.loadtxt(cls.load_csv, delimiter=","),
            node_ids=node_ids,
            sidesets={LOADED: loaded_nodes},
            values=cls.truth_u[:, node_ids, :],
            mesh_file=str(cls.mesh_path),
            mesh_num_nodes=int(cls.mesh.nodes.shape[0]),
            roi={
                "sides": roi_sides,
                "band": np.float64(0.0), "max_gap": np.float64(0.0),
            },
        ).write(cls.archive)
        cls.roi_file = cls.dir / "roi.npz"
        np.savez(cls.roi_file, sides=roi_sides)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _terms(
            self, *, archive: bool, match_times_file: Path | None = None,
    ) -> dict[str, Any]:
        if archive:
            disp: dict[str, Any] = {
                "name": "fe_displacement_match",
                "calibration_data_file": str(self.archive),
            }
            load: dict[str, Any] = {
                "name": "fe_load_match",
                "calibration_data_file": str(self.archive),
                "sideset": LOADED, "components": [0],
            }
            if match_times_file is not None:
                disp["match_times_file"] = str(match_times_file)
                load["match_times_file"] = str(match_times_file)
        else:
            disp = {
                "name": "fe_displacement_match",
                "data_file": str(self.truth_exo),
                "roi_file": str(self.roi_file),
            }
            load = {
                "name": "fe_load_match", "data_file": str(self.load_csv),
                "sideset": LOADED, "components": [0],
            }
        return {"name": "fe_weighted_sum", "terms": [disp, load]}

    def _objective(
            self, tag: str, qoi: dict[str, Any],
            times: list[float] = SCHEDULE,
    ) -> float:
        out = self.dir / f"out_{tag}"
        deck = _deck(
            self.mesh_path, out, times=times, loaded_face=None,
            kappa=1.5 * KAPPA_TRUTH, qoi=qoi,
        )
        self.assertEqual(_run("objective", deck, self.dir / f"{tag}.yaml"), 0)
        with (out / "J.json").open() as f:
            return float(json.load(f)["J"])

    def test_field_bc_from_the_archive_reproduces_the_expression(self) -> None:
        out = self.dir / "out_bc"
        deck = _deck(
            self.mesh_path, out, times=SCHEDULE, loaded_face=self.archive,
            kappa=KAPPA_TRUTH, exodus_filename="bc.exo",
        )
        self.assertEqual(_run("primal", deck, self.dir / "bc.yaml"), 0)
        u = read_results(out / "bc.exo", nodal_field_specs=U_SPEC).nodal["u"]
        np.testing.assert_allclose(u, self.truth_u, atol=1e-12)

    def test_objective_from_the_archive_equals_the_plain_files(self) -> None:
        J_plain = self._objective("plain", self._terms(archive=False))
        J_archive = self._objective("archive", self._terms(archive=True))
        self.assertGreater(J_plain, 1e-8)
        np.testing.assert_allclose(J_archive, J_plain, rtol=1e-12)

    def test_match_subset_equals_the_coarse_schedule(self) -> None:
        match_file = self.dir / "match_times.txt"
        np.savetxt(match_file, COARSE)
        J_fine = self._objective(
            "fine", self._terms(archive=True, match_times_file=match_file),
        )
        J_coarse = self._objective(
            "coarse", self._terms(archive=True), times=COARSE,
        )
        self.assertGreater(J_coarse, 1e-8)
        np.testing.assert_allclose(J_fine, J_coarse, rtol=1e-10)

    def test_a_match_time_off_the_frames_raises(self) -> None:
        match_file = self.dir / "bad_match_times.txt"
        np.savetxt(match_file, [0.0, 0.3, 1.0])
        with self.assertRaisesRegex(ValueError, "0.3 is not a frame"):
            self._objective(
                "bad", self._terms(archive=True, match_times_file=match_file),
            )


if __name__ == "__main__":
    unittest.main()
