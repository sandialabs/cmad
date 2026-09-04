"""Replay the Jones 304L pipeline from the specimen record.

For each chosen specimen in examples/jones_304l_specimens.yaml:
regenerate its meshes, regenerate its calibration data archives, and
write input files under examples/jones_304l_inputs/<specimen>/, a
primal and a calibrate file per dimension. The primal files write the
solved field to exodus and the reaction series to a CSV, the inputs
of jones_304l_compare.py. Calibrate weights are placeholders.

Usage:
    python examples/jones_304l_setup.py --specimens o5
    python examples/jones_304l_setup.py --specimens o5 --dims 2
    python examples/jones_304l_setup.py --steps inputs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

EXAMPLES = Path(__file__).resolve().parent
RECORD = EXAMPLES / "jones_304l_specimens.yaml"
MESH_DIR = EXAMPLES / "meshes"
INPUT_DIR = EXAMPLES / "jones_304l_inputs"

GLOBAL_NEWTON: dict[str, Any] = {
    "nonlinear max iters": 30,
    "nonlinear absolute tol": 1.0e-12,
    "nonlinear relative tol": 1.0e-10,
    "line search": {"max evals": 5},
    "time refinement": {"max depth": 2},
}
LOCAL_NEWTON: dict[str, Any] = {
    "nonlinear max iters": 20,
    "nonlinear absolute tol": 1.0e-12,
    "nonlinear relative tol": 1.0e-12,
}
ELASTIC = {"E": 192.7e3, "nu": 0.27}
START = {"Y": 330.0, "H": 2500.0, "D": 2.5}


def load_record() -> tuple[float, dict[str, dict[str, Any]]]:
    record = yaml.safe_load(RECORD.read_text())
    thickness = float(record.pop("thickness"))
    return thickness, record


def mesh_name(tag: str, entry: dict[str, Any], thickness: float,
              ndims: int) -> str:
    cuts = f"y{entry['y min']:g}_{entry['y max']:g}_h{entry['h']:g}"
    if ndims == 2:
        return f"jones_304l_{tag}_2d_{cuts}.msh"
    return f"jones_304l_{tag}_3d_t{thickness:g}_{cuts}.msh"


def data_stem(tag: str, entry: dict[str, Any], thickness: float,
              ndims: int) -> str:
    mesh = mesh_name(tag, entry, thickness, ndims)
    return f"data/jones_304l/{Path(mesh).stem}"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def make_meshes(tag: str, entry: dict[str, Any], thickness: float,
                dims: list[int]) -> None:
    base = [
        sys.executable, str(EXAMPLES / "jones_304l_mesh.py"),
        "--geometry", entry["geometry"],
        "--y-min", str(entry["y min"]), "--y-max", str(entry["y max"]),
        "--h", str(entry["h"]),
        "--curvature-elements", str(entry["curvature elements"]),
    ]
    for ndims in dims:
        cmd = list(base)
        if ndims == 3:
            cmd += ["--thickness", str(thickness)]
        cmd += ["--out", str(MESH_DIR / mesh_name(tag, entry, thickness, ndims))]
        run(cmd)


def make_archives(tag: str, entry: dict[str, Any], thickness: float,
                  dims: list[int]) -> None:
    for ndims in dims:
        cmd = [
            sys.executable, str(EXAMPLES / "jones_304l_dic_data.py"),
            "--geometry", entry["geometry"], "--data", entry["data"],
            "--mesh", str(MESH_DIR / mesh_name(tag, entry, thickness, ndims)),
            "--y-min", str(entry["y min"]), "--y-max", str(entry["y max"]),
            "--frame-min", str(entry["frame min"]),
            "--frame-max", str(entry["frame max"]),
            "--drop-threshold", str(entry["drop threshold"]),
            "--num-steps", str(entry["num steps"]),
            "--roi-band", str(entry["roi band"]),
        ]
        if entry["exclude frames"]:
            cmd += ["--exclude-frames"]
            cmd += [str(f) for f in entry["exclude frames"]]
        run(cmd)


def materials_section(active: bool) -> dict[str, Any]:
    def param(value: float) -> Any:
        if not active:
            return value
        return {"value": value, "active": True, "transform": {"log": value}}

    return {
        "solid": {
            "elastic": dict(ELASTIC),
            "plastic": {
                "effective stress": {"J2": {}},
                "flow stress": {
                    "initial yield": {"Y": param(START["Y"])},
                    "hardening": {
                        "voce_modulus": {
                            "H": param(START["H"]), "D": param(START["D"]),
                        },
                    },
                },
            },
        },
    }


def residuals_section(ndims: int, active: bool) -> dict[str, Any]:
    if ndims == 2:
        gr: dict[str, Any] = {"type": "mechanics", "def_type": "plane_stress"}
    else:
        gr = {
            "type": "mechanics", "def_type": "full_3d",
            "mixed": True, "stabilization multiplier": 1.0,
        }
    gr.update(GLOBAL_NEWTON)
    local: dict[str, Any] = {"type": "be_bar_elastic_plastic"}
    local.update(LOCAL_NEWTON)
    local["materials"] = materials_section(active)
    return {"global residual": gr, "local residual": local}


def discretization_section(tag: str, entry: dict[str, Any], thickness: float,
                           ndims: int) -> dict[str, Any]:
    section: dict[str, Any] = {
        "mesh file": f"examples/meshes/{mesh_name(tag, entry, thickness, ndims)}",
        "build coordinate sidesets": True,
        "times file": f"{data_stem(tag, entry, thickness, ndims)}_solve_times.txt",
    }
    if ndims == 2:
        section["thickness"] = thickness
    return section


def dirichlet_section(tag: str, entry: dict[str, Any], thickness: float,
                      ndims: int) -> dict[str, Any]:
    archive = f"{data_stem(tag, entry, thickness, ndims)}_calibration_data.npz"
    field = {
        "bot_x": ["equilibrium", 0, "ymin_sides"],
        "bot_y": ["equilibrium", 1, "ymin_sides"],
        "top_x": ["equilibrium", 0, "ymax_sides"],
        "top_y": ["equilibrium", 1, "ymax_sides"],
    }
    if ndims == 3:
        field["bot_z"] = ["equilibrium", 2, "ymin_sides"]
        field["top_z"] = ["equilibrium", 2, "ymax_sides"]
    return {"field data file": archive, "field": field}


def input_file(tag: str, entry: dict[str, Any], thickness: float,
               ndims: int, kind: str) -> dict[str, Any]:
    archive = f"{data_stem(tag, entry, thickness, ndims)}_calibration_data.npz"
    out_path = f"results/jones_304l_{tag}/{kind}_{ndims}d"
    deck: dict[str, Any] = {
        "problem": {"type": "fe", "name": f"{tag}_{kind}_{ndims}d"},
        "discretization": discretization_section(tag, entry, thickness, ndims),
        "residuals": residuals_section(ndims, active=kind == "calibrate"),
        "dirichlet bcs": dirichlet_section(tag, entry, thickness, ndims),
        "output": {"path": out_path},
    }
    if kind == "primal":
        deck["output"]["global residual"] = ["u"]
        deck["output"]["local residual"] = {"solid": ["cauchy", "alpha"]}
        deck["qoi"] = {
            "name": "fe_load_match",
            "sideset": "ymax_sides",
            "components": [1],
            "output_file": f"{out_path}/reaction.csv",
        }
    elif kind == "calibrate":
        deck["qoi"] = {
            "name": "fe_weighted_sum",
            "terms": [
                {
                    "name": "fe_displacement_match",
                    "calibration_data_file": archive,
                    "weight": 1.0,
                },
                {
                    "name": "fe_load_match",
                    "calibration_data_file": archive,
                    "sideset": "ymax_sides",
                    "components": [1],
                    "weight": 1.0,
                },
            ],
        }
        deck["optimizer"] = {
            "algorithm": "L-BFGS-B",
            "options": {"ftol": 1.0e-10, "gtol": 1.0e-3, "maxiter": 40},
        }
    else:
        raise ValueError(f"unknown input file kind {kind!r}")
    return deck


def make_inputs(tag: str, entry: dict[str, Any], thickness: float,
                dims: list[int]) -> None:
    out_dir = INPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "# generated by jones_304l_setup.py from jones_304l_specimens.yaml\n"
    )
    for ndims in dims:
        for kind in ("primal", "calibrate"):
            deck = input_file(tag, entry, thickness, ndims, kind)
            path = out_dir / f"{kind}_{ndims}d.yaml"
            note = "# the weights are placeholders\n" if kind == "calibrate" else ""
            path.write_text(
                header + note + yaml.safe_dump(deck, sort_keys=False),
            )
            print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specimens", nargs="+", default=None,
        help="specimen tags from the record (default: all)",
    )
    parser.add_argument(
        "--dims", nargs="+", type=int, default=[2, 3], choices=(2, 3),
        help="dimensions to process (default: both)",
    )
    parser.add_argument(
        "--steps", default="meshes,archives,inputs",
        help="comma separated subset of meshes,archives,inputs",
    )
    args = parser.parse_args()

    thickness, record = load_record()
    tags = args.specimens or list(record)
    unknown = set(tags) - set(record)
    if unknown:
        raise SystemExit(
            f"unknown specimen(s) {sorted(unknown)}; "
            f"the record has {sorted(record)}"
        )
    steps = args.steps.split(",")
    unknown_steps = set(steps) - {"meshes", "archives", "inputs"}
    if unknown_steps:
        raise SystemExit(f"unknown step(s) {sorted(unknown_steps)}")

    for tag in tags:
        entry = record[tag]
        if "meshes" in steps:
            make_meshes(tag, entry, thickness, args.dims)
        if "archives" in steps:
            make_archives(tag, entry, thickness, args.dims)
        if "inputs" in steps:
            make_inputs(tag, entry, thickness, args.dims)


if __name__ == "__main__":
    main()
