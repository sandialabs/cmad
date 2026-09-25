"""Replay the Jones 304L pipeline from the specimen record.

For each chosen specimen in examples/jones_304l_specimens.yaml:
regenerate its meshes, regenerate its calibration data archives, and
write its input files under examples/jones_304l_inputs/<specimen>/, a
primal and a calibrate file per dimension. With more than one specimen
chosen, one calibrate input file per dimension over exactly those
specimens is written under examples/jones_304l_inputs/joint/, which
cmad calibrate and cmad cross_validate read. The primal files write the
solved field to exodus and the reaction series to a CSV, which
jones_304l_compare.py reads. --materials takes a calibration's
opt_params.yaml as the start values of the calibrate files and the
values of the primal files.

Usage (--specimens lists the tags; --steps picks which of meshes,
archives, and inputs run, all three by default):
    python examples/jones_304l_setup.py --specimens o5
    python examples/jones_304l_setup.py --specimens o5 --dims 2
    python examples/jones_304l_setup.py --specimens xt10 --h 1.5
    # input files only, for three specimens and the joint file over them
    python examples/jones_304l_setup.py --steps inputs --specimens xt10 o5 o14
    # the same, starting the calibrate files from a finished calibration
    python examples/jones_304l_setup.py --steps inputs --specimens xt10 o5 o14 \
        --materials results/jones_304l_xt10/calibrate_2d/opt_params.yaml
"""
from __future__ import annotations

import argparse
import copy
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
    "nonlinear absolute tol": 1.0e-8,
    "nonlinear relative tol": 1.0e-6,
    "line search": {"max evals": 5},
}
LOCAL_NEWTON: dict[str, Any] = {
    "nonlinear max iters": 20,
    "nonlinear absolute tol": 1.0e-12,
    "nonlinear relative tol": 1.0e-12,
}
OPTIMIZER: dict[str, Any] = {
    "algorithm": "L-BFGS-B",
    "options": {"ftol": 1.0e-10, "gtol": 1.0e-3, "maxiter": 100},
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


def load_materials(path: Path) -> dict[str, Any]:
    """The ``materials`` subtree of a calibration's opt_params.yaml."""
    materials: dict[str, Any] = yaml.safe_load(path.read_text())["materials"]
    return materials


def residuals_section(ndims: int, materials: dict[str, Any]) -> dict[str, Any]:
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
    local["materials"] = materials
    return {"global residual": gr, "local residual": local}


def discretization_section(tag: str, entry: dict[str, Any], thickness: float,
                           ndims: int) -> dict[str, Any]:
    section: dict[str, Any] = {
        "mesh file": f"examples/meshes/{mesh_name(tag, entry, thickness, ndims)}",
        "build coordinate sidesets": True,
        "times file": f"{data_stem(tag, entry, thickness, ndims)}_solve_times.txt",
        "time refinement": {"max depth": 2},
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


def calibrate_qoi(tag: str, entry: dict[str, Any], thickness: float,
                  ndims: int) -> dict[str, Any]:
    """The log sum of the displacement and load matches, every weight 1."""
    archive = f"{data_stem(tag, entry, thickness, ndims)}_calibration_data.npz"
    return {
        "name": "fe_log_sum",
        "terms": [
            {
                "name": "fe_displacement_match",
                "calibration_data_file": archive,
            },
            {
                "name": "fe_load_match",
                "calibration_data_file": archive,
                "sideset": "ymax_sides",
                "components": [1],
            },
        ],
    }


def input_file(tag: str, entry: dict[str, Any], thickness: float,
               ndims: int, kind: str,
               materials: dict[str, Any] | None = None) -> dict[str, Any]:
    out_path = f"results/jones_304l_{tag}/{kind}_{ndims}d"
    # Without --materials the script's own values go in, active in a
    # calibrate file and fixed in a primal file; with it, the given
    # subtree goes into both kinds as it is.
    if materials is None:
        if kind == "calibrate":
            materials = materials_section(active=True)
        else:
            materials = materials_section(active=False)
    deck: dict[str, Any] = {
        "problem": {"type": "fe", "name": f"{tag}_{kind}_{ndims}d"},
        "discretization": discretization_section(tag, entry, thickness, ndims),
        "residuals": residuals_section(ndims, materials),
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
        deck["qoi"] = calibrate_qoi(tag, entry, thickness, ndims)
        deck["optimizer"] = copy.deepcopy(OPTIMIZER)
    else:
        raise ValueError(f"unknown input file kind {kind!r}")
    return deck


def joint_input_file(entries: dict[str, dict[str, Any]], thickness: float,
                     ndims: int,
                     materials: dict[str, Any] | None = None) -> dict[str, Any]:
    """One calibrate file over every specimen in ``entries``: the shared
    sections once, then each specimen's own under ``specimens``."""
    return {
        "problem": {"type": "fe", "name": f"joint_calibrate_{ndims}d"},
        "residuals": residuals_section(
            ndims, materials_section(active=True) if materials is None
            else materials,
        ),
        "optimizer": copy.deepcopy(OPTIMIZER),
        "output": {"path": f"results/jones_304l_joint/calibrate_{ndims}d"},
        "specimens": {
            tag: {
                "discretization": discretization_section(
                    tag, entry, thickness, ndims,
                ),
                "dirichlet bcs": dirichlet_section(tag, entry, thickness, ndims),
                "qoi": calibrate_qoi(tag, entry, thickness, ndims),
            }
            for tag, entry in entries.items()
        },
    }


HEADER = "# generated by jones_304l_setup.py from jones_304l_specimens.yaml\n"


def make_inputs(tag: str, entry: dict[str, Any], thickness: float,
                dims: list[int],
                materials: dict[str, Any] | None = None) -> None:
    out_dir = INPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for ndims in dims:
        for kind in ("primal", "calibrate"):
            deck = input_file(tag, entry, thickness, ndims, kind, materials)
            path = out_dir / f"{kind}_{ndims}d.yaml"
            path.write_text(HEADER + yaml.safe_dump(deck, sort_keys=False))
            print(f"wrote {path}")


def make_joint_input(entries: dict[str, dict[str, Any]], thickness: float,
                     dims: list[int],
                     materials: dict[str, Any] | None = None) -> None:
    out_dir = INPUT_DIR / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ndims in dims:
        deck = joint_input_file(entries, thickness, ndims, materials)
        path = out_dir / f"calibrate_{ndims}d.yaml"
        path.write_text(HEADER + yaml.safe_dump(deck, sort_keys=False))
        print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specimens", nargs="+", required=True,
        help="specimen tags from the record",
    )
    parser.add_argument(
        "--dims", nargs="+", type=int, default=[2, 3], choices=(2, 3),
        help="dimensions to process (default: both)",
    )
    parser.add_argument(
        "--steps", default="meshes,archives,inputs",
        help="comma separated subset of meshes,archives,inputs",
    )
    parser.add_argument(
        "--h", type=float, default=None,
        help="override the record's element size",
    )
    parser.add_argument(
        "--num-steps", type=int, default=None,
        help="override the record's number of schedule steps",
    )
    parser.add_argument(
        "--materials", type=Path, default=None,
        help="a calibration's opt_params.yaml whose materials subtree "
             "replaces the start values in the calibrate files and the "
             "values in the primal files",
    )
    args = parser.parse_args()
    materials = None if args.materials is None else load_materials(args.materials)

    thickness, record = load_record()
    tags = args.specimens
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

    entries: dict[str, dict[str, Any]] = {}
    for tag in tags:
        entry = dict(record[tag])
        if args.h is not None:
            entry["h"] = args.h
        if args.num_steps is not None:
            entry["num steps"] = args.num_steps
        entries[tag] = entry
        if "meshes" in steps:
            make_meshes(tag, entry, thickness, args.dims)
        if "archives" in steps:
            make_archives(tag, entry, thickness, args.dims)
        if "inputs" in steps:
            make_inputs(tag, entry, thickness, args.dims, materials)
    if "inputs" in steps and len(entries) > 1:
        make_joint_input(entries, thickness, args.dims, materials)


if __name__ == "__main__":
    main()
