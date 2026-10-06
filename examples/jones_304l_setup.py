"""Replay the Jones 304L pipeline from the specimen record.

For each chosen specimen in examples/jones_304l_specimens.yaml:
regenerate its meshes, regenerate its calibration data archives, and
write its input files under examples/jones_304l_inputs/<specimen>/, a
primal and a calibrate file per dimension. With more than one specimen
chosen, one calibrate input file per dimension over exactly those
specimens is written under examples/jones_304l_inputs/joint/, which
cmad calibrate and cmad cross_validate read. The primal files write the
solved field to exodus and the reaction series to a CSV, which
jones_304l_compare.py reads. --materials takes the opt_params.yaml of a
calibration as the initial values of the primal and calibrate files.

--model picks the material model, be_bar with Voce hardening or the rate
model with Johnson-Cook, and --temperature how the temperature enters:
none (isothermal), coupled (the thermomechanics residual with the
plastic heating, the measured temperature at the cuts, and a temperature
match term), or measured (the mechanics residual reading the measured
temperature as a prescribed field).

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
    # the rate model with the measured temperature prescribed
    python examples/jones_304l_setup.py --steps inputs --specimens xt6 \
        --model rate_johnson_cook --temperature measured
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cmad.calibration import active_param_paths
from cmad.io.calibration_data import CalibrationData
from cmad.io.params_builder import build_parameters

EXAMPLES = Path(__file__).resolve().parent
RECORD = EXAMPLES / "jones_304l_specimens.yaml"
MESH_DIR = EXAMPLES / "meshes"
INPUT_DIR = EXAMPLES / "jones_304l_inputs"

GLOBAL_NEWTON: dict[str, Any] = {
    "nonlinear max iters": 10,
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
JOHNSON_COOK_START = {"A": 335.0, "B": 1100.0, "n": 0.6, "C": 0.03, "m": 1.0}
# The reference temperature of an isothermal run; with the temperature
# coupled or measured it is read from the archives.
REFERENCE_TEMPERATURE = 296.0
ALPHA = 1.6e-5
# 304L in mm, Mg, s, and K: W/(mm K), Mg/mm^3, mJ/(Mg K), and W/(mm^2 K).
THERMAL = {"conductivity": 16.0, "density": 7.85e-9, "specific heat": 5.0e+8}
CONVECTION_H = 2.5e-2
TAYLOR_QUINNEY = 0.9
MODELS = ("be_bar", "rate_johnson_cook")
TEMPERATURES = ("none", "coupled", "measured")


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
        if "emissivity bound" in entry:
            cmd += ["--emissivity-bound", str(entry["emissivity bound"])]
        run(cmd)


def ambient_temperature(archive: str) -> float:
    """The mean of the archive's first temperature row, the temperature the
    specimen started at."""
    path = Path(archive)
    if not path.exists():
        raise SystemExit(f"{archive} does not exist; run the archives stage")
    temperature = CalibrationData.read(path).temperature
    if temperature is None:
        raise SystemExit(f"{archive} holds no temperature")
    return round(float(np.mean(temperature[0])), 2)


def run_reference_temperature(
        entries: dict[str, dict[str, Any]], thickness: float, temperature: str,
) -> float:
    """The reference temperature of the material every specimen in the run
    shares: the ambient of the archives averaged over the specimens, or
    296 K when the temperature is not used."""
    if temperature == "none":
        return REFERENCE_TEMPERATURE
    return float(np.mean([
        ambient_temperature(archive_name(tag, entry, thickness, 2))
        for tag, entry in entries.items()
    ]))


def materials_section(
        active: bool, model: str, temperature: str, ndims: int,
        reference_temperature: float, initial_values: dict[str, float],
) -> dict[str, Any]:
    """The materials of ``model``, each constant starting at its entry of
    ``initial_values``, as active parameters or as plain values."""
    def param(name: str) -> Any:
        value = initial_values[name]
        if not active:
            return value
        return {"value": value, "active": True, "transform": {"log": value}}

    if model == "be_bar":
        plastic: dict[str, Any] = {
            "effective stress": {"J2": {}},
            "flow stress": {
                "initial yield": {"Y": param("Y")},
                "hardening": {
                    "voce_modulus": {"H": param("H"), "D": param("D")},
                },
            },
        }
        solid: dict[str, Any] = {"elastic": dict(ELASTIC), "plastic": plastic}
        return {"solid": solid}

    jc = {name: param(name) for name in JOHNSON_COOK_START}
    plastic = {
        "effective stress": {"J2": {}},
        "flow stress": {
            "johnson_cook": {
                **jc,
                "reference rate": 1.0e-4,
                "reference temperature": reference_temperature,
                "melt temperature": 1673.0,
            },
        },
    }
    solid = {
        "elastic": dict(ELASTIC),
        "plastic": plastic,
        "thermal expansion": {
            "alpha": ALPHA, "reference temperature": reference_temperature,
        },
    }
    if temperature == "coupled":
        plastic["taylor-quinney"] = initial_values["taylor-quinney"]
        thermal: dict[str, Any] = dict(THERMAL)
        if ndims == 2:
            thermal["face convection"] = {
                "h": CONVECTION_H, "T_inf": reference_temperature,
            }
        solid["thermal"] = thermal
    return {"solid": solid}


def load_materials(path: Path) -> dict[str, Any]:
    """The ``materials`` subtree of an opt_params.yaml."""
    materials: dict[str, Any] = yaml.safe_load(path.read_text())["materials"]
    return materials


def extract_active_values(materials: dict[str, Any]) -> dict[str, float]:
    """The values of the active parameters of a materials subtree, by name."""
    parameters = build_parameters(materials["solid"])
    names = [path.split(".")[-1] for path in active_param_paths(parameters)]
    values = parameters.flat_active_values(return_canonical=False)
    return {
        name: float(value) for name, value in zip(names, values, strict=True)
    }


def residuals_section(ndims: int, materials: dict[str, Any], model: str,
                      temperature: str,
                      reference_temperature: float) -> dict[str, Any]:
    gr_type = "thermomechanics" if temperature == "coupled" else "mechanics"
    if ndims == 2:
        gr: dict[str, Any] = {"type": gr_type, "def_type": "plane_stress"}
    else:
        gr = {
            "type": gr_type, "def_type": "full_3d",
            "mixed": True, "stabilization multiplier": 1.0,
        }
    gr.update(GLOBAL_NEWTON)
    if model == "be_bar":
        local: dict[str, Any] = {"type": "be_bar_elastic_plastic"}
    else:
        local = {
            "type": "rate_elastic_plastic", "finite deformation": True,
            "reference temperature": reference_temperature,
        }
    local.update(LOCAL_NEWTON)
    local["materials"] = materials
    return {"global residual": gr, "local residual": local}


def discretization_section(tag: str, entry: dict[str, Any], thickness: float,
                           ndims: int, temperature: str) -> dict[str, Any]:
    section: dict[str, Any] = {
        "mesh file": f"examples/meshes/{mesh_name(tag, entry, thickness, ndims)}",
        "build coordinate sidesets": True,
        "times file": f"{data_stem(tag, entry, thickness, ndims)}_solve_times.txt",
        "time refinement": {"max depth": 2},
    }
    if ndims == 2:
        section["thickness"] = thickness
    if temperature == "coupled":
        section["time refinement"] = {"max depth": 4}
    return section


def archive_name(tag: str, entry: dict[str, Any], thickness: float,
                 ndims: int) -> str:
    return f"{data_stem(tag, entry, thickness, ndims)}_calibration_data.npz"


def dirichlet_section(tag: str, entry: dict[str, Any], thickness: float,
                      ndims: int, temperature: str) -> dict[str, Any]:
    field = {
        "bot_x": ["equilibrium", 0, "ymin_sides"],
        "bot_y": ["equilibrium", 1, "ymin_sides"],
        "top_x": ["equilibrium", 0, "ymax_sides"],
        "top_y": ["equilibrium", 1, "ymax_sides"],
    }
    if ndims == 3:
        field["bot_z"] = ["equilibrium", 2, "ymin_sides"]
        field["top_z"] = ["equilibrium", 2, "ymax_sides"]
    if temperature == "coupled":
        field["bot_T"] = ["energy balance", 0, "ymin_sides"]
        field["top_T"] = ["energy balance", 0, "ymax_sides"]
    return {
        "field data file": archive_name(tag, entry, thickness, ndims),
        "field": field,
    }


def convection_section(ndims: int, T_inf: float) -> dict[str, Any]:
    """Convection to the air on the free boundary and, in 3D, on the two
    faces; in 2D the faces are the material's face convection."""
    sidesets = ["free_sides"]
    if ndims == 3:
        sidesets += ["zmin_sides", "zmax_sides"]
    return {"expression": {
        sideset: ["energy balance", sideset, CONVECTION_H, T_inf]
        for sideset in sidesets
    }}


def specimen_thermal_sections(tag: str, entry: dict[str, Any],
                              thickness: float, ndims: int,
                              temperature: str,
                              reference_temperature: float) -> dict[str, Any]:
    """The sections the thermal form adds per specimen: the prescribed
    temperature in the measured form; the convection and the initial
    temperature in the coupled form."""
    archive = archive_name(tag, entry, thickness, ndims)
    if temperature == "measured":
        return {"prescribed fields": {"T": {"data file": archive}}}
    if temperature == "coupled":
        return {
            "convection bcs": convection_section(ndims, reference_temperature),
            "initial conditions": {"T": reference_temperature},
        }
    return {}


def calibrate_qoi(tag: str, entry: dict[str, Any], thickness: float,
                  ndims: int, temperature: str) -> dict[str, Any]:
    """The log sum of the displacement and load matches, every weight 1,
    and of the temperature match in the coupled form."""
    archive = archive_name(tag, entry, thickness, ndims)
    terms = [
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
    ]
    if temperature == "coupled":
        terms.append({
            "name": "fe_temperature_match",
            "calibration_data_file": archive,
        })
    return {"name": "fe_log_sum", "terms": terms}


def file_stem(kind: str, ndims: int, temperature: str) -> str:
    suffix = "_T_measured" if temperature == "measured" else ""
    return f"{kind}_{ndims}d{suffix}"


def input_file(tag: str, entry: dict[str, Any], thickness: float,
               ndims: int, kind: str, model: str, temperature: str,
               reference_temperature: float,
               initial_values: dict[str, float]) -> dict[str, Any]:
    stem = file_stem(kind, ndims, temperature)
    out_path = f"results/jones_304l_{tag}/{stem}"
    materials = materials_section(
        kind != "primal", model, temperature, ndims, reference_temperature,
        initial_values,
    )
    deck: dict[str, Any] = {
        "problem": {"type": "fe", "name": f"{tag}_{stem}"},
        "discretization": discretization_section(
            tag, entry, thickness, ndims, temperature,
        ),
        "residuals": residuals_section(
            ndims, materials, model, temperature, reference_temperature,
        ),
        "dirichlet bcs": dirichlet_section(
            tag, entry, thickness, ndims, temperature,
        ),
        **specimen_thermal_sections(
            tag, entry, thickness, ndims, temperature, reference_temperature,
        ),
        "output": {"path": out_path},
    }
    if kind == "primal":
        deck["residuals"]["global residual"]["print convergence"] = True
        nodal = ["u", "T"] if temperature == "coupled" else ["u"]
        element = ["cauchy", "alpha"]
        if temperature == "coupled":
            element.append("heat flux")
        deck["output"]["global residual"] = nodal
        deck["output"]["local residual"] = {"solid": element}
        deck["qoi"] = {
            "name": "fe_load_match",
            "sideset": "ymax_sides",
            "components": [1],
            "output_file": f"{out_path}/reaction.csv",
        }
    elif kind == "calibrate":
        deck["qoi"] = calibrate_qoi(tag, entry, thickness, ndims, temperature)
        deck["optimizer"] = copy.deepcopy(OPTIMIZER)
    else:
        raise ValueError(f"unknown input file kind {kind!r}")
    return deck


def joint_input_file(entries: dict[str, dict[str, Any]], thickness: float,
                     ndims: int, model: str, temperature: str,
                     reference_temperature: float,
                     initial_values: dict[str, float]) -> dict[str, Any]:
    """One calibrate file over every specimen in ``entries``: the shared
    sections once, then each specimen's own under ``specimens``."""
    stem = file_stem("calibrate", ndims, temperature)
    materials = materials_section(
        True, model, temperature, ndims, reference_temperature, initial_values,
    )
    return {
        "problem": {"type": "fe", "name": f"joint_{stem}"},
        "residuals": residuals_section(
            ndims, materials, model, temperature, reference_temperature,
        ),
        "optimizer": copy.deepcopy(OPTIMIZER),
        "output": {"path": f"results/jones_304l_joint/{stem}"},
        "specimens": {
            tag: {
                "discretization": discretization_section(
                    tag, entry, thickness, ndims, temperature,
                ),
                "dirichlet bcs": dirichlet_section(
                    tag, entry, thickness, ndims, temperature,
                ),
                **specimen_thermal_sections(
                    tag, entry, thickness, ndims, temperature,
                    reference_temperature,
                ),
                "qoi": calibrate_qoi(tag, entry, thickness, ndims, temperature),
            }
            for tag, entry in entries.items()
        },
    }


HEADER = "# generated by jones_304l_setup.py from jones_304l_specimens.yaml\n"


def make_inputs(tag: str, entry: dict[str, Any], thickness: float,
                dims: list[int], model: str, temperature: str,
                reference_temperature: float,
                initial_values: dict[str, float]) -> None:
    out_dir = INPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for ndims in dims:
        for kind in ("primal", "calibrate"):
            deck = input_file(
                tag, entry, thickness, ndims, kind, model, temperature,
                reference_temperature, initial_values,
            )
            path = out_dir / f"{file_stem(kind, ndims, temperature)}.yaml"
            path.write_text(HEADER + yaml.safe_dump(deck, sort_keys=False))
            print(f"wrote {path}")


def make_joint_input(entries: dict[str, dict[str, Any]], thickness: float,
                     dims: list[int], model: str, temperature: str,
                     reference_temperature: float,
                     initial_values: dict[str, float]) -> None:
    out_dir = INPUT_DIR / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ndims in dims:
        deck = joint_input_file(
            entries, thickness, ndims, model, temperature,
            reference_temperature, initial_values,
        )
        path = out_dir / f"{file_stem('calibrate', ndims, temperature)}.yaml"
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
        help="an opt_params.yaml whose active parameters give the initial "
             "values of the primal and calibrate files",
    )
    parser.add_argument(
        "--model", default="be_bar", choices=MODELS,
        help="the material model (default be_bar)",
    )
    parser.add_argument(
        "--temperature", default="none", choices=TEMPERATURES,
        help="how the temperature enters: none, coupled, or measured "
             "(default none)",
    )
    args = parser.parse_args()
    if args.model == "be_bar" and args.temperature != "none":
        raise SystemExit("--temperature applies to --model rate_johnson_cook")
    initial_values = {
        **START, **JOHNSON_COOK_START, "taylor-quinney": TAYLOR_QUINNEY,
    }
    if args.materials is not None:
        initial_values.update(extract_active_values(load_materials(args.materials)))

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
    if "inputs" not in steps:
        return
    reference_temperature = run_reference_temperature(
        entries, thickness, args.temperature,
    )
    for tag, entry in entries.items():
        make_inputs(
            tag, entry, thickness, args.dims, args.model, args.temperature,
            reference_temperature, initial_values,
        )
    if len(entries) > 1:
        make_joint_input(
            entries, thickness, args.dims, args.model, args.temperature,
            reference_temperature, initial_values,
        )


if __name__ == "__main__":
    main()
