"""DIC full field calibration on a notched thin plate.

A fine be_bar finite plasticity solve of a notched thin plate under a
displacement ramp supplies the synthetic measurements: a stereo DIC displacement
cloud sampled on the front face (the ``dic cloud`` primal output) and a reaction
series on the loaded face (the ``fe_load_match`` write mode). A coarser mesh then
recovers the hardening parameters from that data through a weighted sum of a DIC
match and a load match.

Run from the repo root:
    python examples/dic_calibration_notch.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from jax import jit

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dic_sampling_viz import build_measurement_grid  # noqa: E402
from notch_mesh import generate_notch_msh  # noqa: E402

from cmad.cli.common import (  # noqa: E402
    build_fe_J_of_params_flat,
    build_fe_problem_from_deck,
)
from cmad.cli.main import main as cmad_main  # noqa: E402
from cmad.io.point_cloud import PointCloud, write_point_cloud  # noqa: E402

# material regime (settled by the flow curve and notch probes)
E, NU = 200e3, 0.3
Y, S, D = 200.0, 200.0, 6.0
# perturbed start for the inversion (fixing E, nu at truth)
Y0, S0, D0 = 260.0, 140.0, 7.2
# geometry and loading
PLATE = (1.0, 1.0, 0.1)          # thin plate; DIC on the zmax face, load on ymax
NOTCH_R = 0.20
GRIP_DISP = 0.15                 # nominal dL/L at the final step (L_y = 1)
NUM_STEPS = 15
# measurement sampling (the dic_sampling_viz defaults)
PITCH, EDGE_MARGIN, FREE_EDGE_BAND = 0.020, 0.02, 0.020
# fine truth vs a modestly coarser inversion mesh, both resolving the notch and
# through thickness so the mesh mismatch stays small
FINE_H, COARSE_H = 0.04, 0.05

WORK_DIR = _HERE.parent / "scratch" / "dic_calibration"


def _discretization(mesh_file: Path) -> dict:
    return {
        "mesh file": str(mesh_file),
        "build coordinate sidesets": True,
        "num steps": NUM_STEPS,
        "step size": 1.0 / NUM_STEPS,
    }


def _dbcs() -> dict:
    return {"expression": {
        "sym_x": ["equilibrium", 0, "xmin_sides", "0.0"],
        "sym_y": ["equilibrium", 1, "ymin_sides", "0.0"],
        "sym_z": ["equilibrium", 2, "zmin_sides", "0.0"],
        "load_y": ["equilibrium", 1, "ymax_sides", f"{GRIP_DISP} * t"],
    }}


def _global_residual() -> dict:
    return {
        "type": "mechanics", "def_type": "full_3d",
        "nonlinear max iters": 100,
        "nonlinear absolute tol": 1.0e-8,
        "nonlinear relative tol": 1.0e-8,
        "line search": {"max evals": 10},
    }


def _local_residual(materials: dict) -> dict:
    return {
        "type": "be_bar_elastic_plastic",
        "nonlinear max iters": 10,
        "nonlinear absolute tol": 1.0e-12,
        "nonlinear relative tol": 1.0e-12,
        "line search": {"max evals": 0},
        "materials": materials,
    }


def _truth_material() -> dict:
    return {"solid": {
        "elastic": {"E": E, "nu": NU},
        "plastic": {
            "effective stress": {"J2": 0.0},
            "flow stress": {
                "initial yield": {"Y": Y},
                "hardening": {"voce": {"S": S, "D": D}},
            },
        },
    }}


def _active_material() -> dict:
    """Truth material with Y, S, D made active and log transformed."""
    def leaf(value: float, ref: float) -> dict:
        return {"value": value, "active": True, "transform": {"log": ref}}
    return {"solid": {
        "elastic": {"E": E, "nu": NU},
        "plastic": {
            "effective stress": {"J2": 0.0},
            "flow stress": {
                "initial yield": {"Y": leaf(Y0, Y)},
                "hardening": {"voce": {"S": leaf(S0, S), "D": leaf(D0, D)}},
            },
        },
    }}


def truth_deck(mesh_file: Path, out_dir: Path, points_file: Path) -> dict:
    """Fine mesh primal that writes the DIC cloud and the reaction series."""
    return {
        "problem": {"type": "fe", "name": "dic_truth"},
        "discretization": _discretization(mesh_file),
        "residuals": {
            "global residual": _global_residual(),
            "local residual": _local_residual(_truth_material()),
        },
        "dirichlet bcs": _dbcs(),
        "qoi": {
            "name": "fe_load_match",
            "output_file": str(out_dir / "load.csv"),
            "sideset": "ymax_sides",
            "components": [1],
        },
        "output": {
            "path": str(out_dir),
            "write exodus": False,
            "dic cloud": {
                "points file": str(points_file),
                "sideset": "zmax_sides",
                "output file": "dic.h5",
            },
        },
    }


def calibrate_deck(
        mesh_file: Path, out_dir: Path, dic_file: Path, load_file: Path,
) -> dict:
    """Coarse mesh calibrate against a DIC cloud and a load series."""
    return {
        "problem": {"type": "fe", "name": "dic_invert"},
        "discretization": _discretization(mesh_file),
        "residuals": {
            "global residual": _global_residual(),
            "local residual": _local_residual(_active_material()),
        },
        "dirichlet bcs": _dbcs(),
        "qoi": {
            "name": "fe_weighted_sum",
            "terms": [
                {"name": "fe_dic_match", "dic_file": str(dic_file),
                 "sideset": "zmax_sides", "weight": 1.0},
                {"name": "fe_load_match", "data_file": str(load_file),
                 "sideset": "ymax_sides", "components": [1], "weight": 1.0},
            ],
        },
        "optimizer": {
            "algorithm": "L-BFGS-B",
            "options": {"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 60},
        },
        "output": {"path": str(out_dir)},
    }


def write_measurement_points(points_file: Path) -> int:
    """Lay the measurement grid on the zmax face (z = lz) and write it."""
    captured, _dropped = build_measurement_grid(
        PITCH, EDGE_MARGIN, NOTCH_R, FREE_EDGE_BAND,
        width=PLATE[0], height=PLATE[1],
    )
    coords = np.column_stack([captured, np.full(len(captured), PLATE[2])])
    write_point_cloud(
        points_file,
        PointCloud(coords=coords, times=np.array([0.0]), fields={}),
        write_xdmf=False,
    )
    return len(coords)


def generate_truth() -> tuple[Path, Path]:
    """Fine solve writing the DIC cloud and the load series (skip if present)."""
    dic_file, load_file = WORK_DIR / "dic.h5", WORK_DIR / "load.csv"
    if dic_file.exists() and load_file.exists():
        print("truth data present, skipping the fine solve")
        return dic_file, load_file

    fine_mesh = WORK_DIR / "notch_fine.msh"
    n_tets = generate_notch_msh(
        fine_mesh, FINE_H, plate=PLATE, notch_radius=NOTCH_R,
    )
    points_file = WORK_DIR / "dic_points.h5"
    n_pts = write_measurement_points(points_file)
    print(f"fine mesh: {n_tets} tets at h={FINE_H}; {n_pts} measurement points")

    deck_path = WORK_DIR / "truth.yaml"
    deck_path.write_text(
        yaml.safe_dump(truth_deck(fine_mesh, WORK_DIR, points_file),
                       sort_keys=False),
    )
    t0 = time.perf_counter()
    if cmad_main(["primal", str(deck_path)]) != 0:
        raise SystemExit("truth primal failed")
    print(f"truth solve: {time.perf_counter() - t0:.1f} s")
    return dic_file, load_file


def _balanced_weights(deck: dict) -> dict[str, float]:
    """Weight each fe_weighted_sum term by 1 / (its value at the start params),
    so the DIC and load terms contribute equally at the initial guess rather than
    the raw magnitudes letting the load term dominate."""
    probe_path = WORK_DIR / "weight_probe.yaml"
    weights = {}
    for term in deck["qoi"]["terms"]:
        probe = {**deck, "qoi": {
            "name": "fe_weighted_sum", "terms": [{**term, "weight": 1.0}]}}
        probe_path.write_text(yaml.safe_dump(probe, sort_keys=False))
        bundle = build_fe_problem_from_deck(probe_path, "calibrate")
        params_flat, state_init, cost = build_fe_J_of_params_flat(bundle)
        raw = float(jit(cost)(
            np.asarray(params_flat, dtype=np.float64), state_init,
            bundle.fe_problem.kernel_arrays))
        weights[term["name"]] = 1.0 / raw
    return weights


def run_inversion(
        coarse_mesh: Path, dic_file: Path, load_file: Path, tag: str,
) -> tuple[dict, dict]:
    cal_out = WORK_DIR / f"cal_{tag}"
    deck = calibrate_deck(coarse_mesh, cal_out, dic_file, load_file)
    weights = _balanced_weights(deck)
    for term in deck["qoi"]["terms"]:
        term["weight"] = weights[term["name"]]
    deck_path = WORK_DIR / f"cal_{tag}.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
    t0 = time.perf_counter()
    if cmad_main(["calibrate", str(deck_path)]) != 0:
        raise SystemExit(f"calibrate ({tag}) failed")
    elapsed = time.perf_counter() - t0
    active = json.loads((cal_out / "active_params.json").read_text())
    status = json.loads((cal_out / "opt_status.json").read_text())
    status["seconds"] = elapsed
    return active, status


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dic_file, load_file = generate_truth()

    coarse_mesh = WORK_DIR / "notch_coarse.msh"
    n_tets = generate_notch_msh(
        coarse_mesh, COARSE_H, plate=PLATE, notch_radius=NOTCH_R,
    )
    print(f"coarse mesh: {n_tets} tets at h={COARSE_H}")

    active, status = run_inversion(coarse_mesh, dic_file, load_file, "clean")
    print(f"\nclean inversion: success={status['success']}, "
          f"fun={status['fun']:.3e}, {status['seconds']:.1f} s, "
          f"{status.get('nit', '?')} iters, {status.get('nfev', '?')} evals")
    truth = {"Y": Y, "S": S, "D": D}
    for key, val in active.items():
        name = key.split(".")[-1]
        ref = truth.get(name)
        rel = f"  ({100 * (val / ref - 1):+.2f}% vs truth {ref})" if ref else ""
        print(f"  {key} = {val:.4f}{rel}")


if __name__ == "__main__":
    main()
