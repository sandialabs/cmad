"""Compare a Jones 304L primal run against the measured record.

Reads the calibrate input file, the primal's exodus, and the reaction
CSV the primal wrote. Writes an overlay exodus with u_pred, u_meas,
and u_diff, the measured and difference fields on the nodes of the
matched region and NaN elsewhere, a figure of predicted against
measured load, a figure of the match terms per frame with the spatial
rms of each field mismatch, and a CSV of the values behind the figures.
With a temperature match term in the calibrate file, the temperature is
compared the same way, T_pred, T_meas, and T_diff in the overlay.

Usage:
    python examples/jones_304l_compare.py \
        --deck examples/jones_304l_inputs/o5/calibrate_2d.yaml \
        --exodus results/jones_304l_o5/primal_2d/o5_primal_2d.exo \
        --reaction results/jones_304l_o5/primal_2d/reaction.csv \
        --out-dir results/jones_304l_o5/compare_2d
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from cmad.cli.common import build_fe_problem_from_deck
from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.dof import dof_physical_coords
from cmad.fem.mesh import Mesh
from cmad.fem.topology import _LOCAL_SIDES_PER_ELEMENT
from cmad.io.calibration_data import CalibrationData
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.qoi_data import calibration_data_roi
from cmad.io.results import FieldSpec
from cmad.models.global_fields import StepTime
from cmad.models.var_types import VarType
from cmad.qois.fe_displacement_match import FEDisplacementMatch
from cmad.qois.fe_load_match import FELoadMatch
from cmad.qois.fe_match_term import FEMatchTerm
from cmad.qois.fe_temperature_match import FETemperatureMatch


def roi_nodes(mesh: Mesh, roi: np.ndarray) -> np.ndarray:
    """The node ids in a region of interest, given as element ids on a 2D
    mesh or as (element, local side) pairs on a 3D one."""
    if roi.ndim == 1:
        return np.unique(mesh.connectivity[roi])
    sides = _LOCAL_SIDES_PER_ELEMENT[mesh.element_family]
    return np.unique(np.concatenate([
        mesh.connectivity[elem][list(sides[side])] for elem, side in roi]))


def spatial_rms(term: np.ndarray, qoi: FEMatchTerm, dt: np.ndarray,
                span: float) -> np.ndarray:
    """The spatial rms of a field mismatch per frame from its term: the
    term is ``dt integral |diff|^2 dV / (span V D)`` with ``D`` the data
    mean square, so the mean square over the region is ``term span D /
    dt``."""
    assert qoi.data_mean_square is not None
    rms = np.zeros_like(term)
    rms[1:] = np.sqrt(term[1:] * span * qoi.data_mean_square / dt[1:])
    return rms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deck", required=True,
        help="the calibrate input file naming the mesh, the archive, "
             "the QoI, and the solve times",
    )
    parser.add_argument(
        "--exodus", required=True,
        help="the primal's exodus output carrying the field u",
    )
    parser.add_argument(
        "--reaction", required=True,
        help="the reaction series CSV the primal wrote",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_fe_problem_from_deck(Path(args.deck), "calibrate")
    fe_problem = bundle.fe_problem
    problem_name = bundle.resolved["problem"].get("name", Path(args.deck).stem)
    terms = {t["name"]: t for t in bundle.resolved["qoi"]["terms"]}
    disp_section = terms["fe_displacement_match"]
    load_section = terms["fe_load_match"]
    temp_section = terms.get("fe_temperature_match")
    store = CalibrationData.read(disp_section["calibration_data_file"])

    nodal_specs = [FieldSpec("u", VarType.VECTOR)]
    if temp_section is not None:
        nodal_specs.append(FieldSpec("T", VarType.SCALAR))
    results = read_results(args.exodus, nodal_field_specs=nodal_specs)
    times = np.asarray(results.time, dtype=np.float64)
    u_pred = np.asarray(results.nodal["u"], dtype=np.float64)
    n_nodes = fe_problem.mesh.nodes.shape[0]
    ndims = int(fe_problem.mesh.nodes.shape[1])

    # The measured fields on the nodes of the matched region, NaN elsewhere:
    # the archive stores every node once it has a temperature, but the
    # terms match over the region of interest alone.
    rows = store.frames_at(times)
    inside = np.zeros(n_nodes, dtype=bool)
    inside[roi_nodes(fe_problem.mesh, calibration_data_roi(store, ndims))] = True
    stored = store.node_ids[inside[store.node_ids]]
    u_meas = np.full((times.size, n_nodes, ndims), np.nan)
    u_meas[:, stored] = np.asarray(store.rows(rows, stored))
    if temp_section is not None:
        T_pred = np.asarray(results.nodal["T"], dtype=np.float64)
        T_meas = np.full((times.size, n_nodes), np.nan)
        T_meas[:, stored] = np.asarray(store.rows(rows, stored, field="T"))[:, :, 0]

    # The load term per frame as the QoI computes it: the squared difference
    # between the reaction and the measured load, summed over the components,
    # times the frame's time weight, over the span and the data mean square.
    # The weight of a frame is the time interval before it, so the first
    # frame, the reference state, has weight 0 and is not integrated.
    reaction = np.loadtxt(args.reaction, delimiter=",").reshape(times.size, -1)
    load_meas = np.asarray(store.load[rows], dtype=np.float64)
    components = [int(c) for c in load_section["components"]]
    if load_meas.ndim == 1:
        load_meas = load_meas.reshape(-1, 1)
    dt = np.concatenate([[0.0], np.diff(times)])
    span = float(times[-1] - times[0])
    load_qoi = FELoadMatch.from_deck(load_section, fe_problem, times.tolist())
    assert load_qoi.data_mean_square is not None
    load_sq = np.sum((reaction - load_meas) ** 2, axis=1)
    load_term = dt * load_sq / (span * load_qoi.data_mean_square)

    # The field terms per frame, through the closure of each QoI, and the
    # spatial rms of each mismatch from its term.
    params = params_by_block_from_models(fe_problem)
    disp_qoi = FEDisplacementMatch.from_deck(
        disp_section, fe_problem, times.tolist(),
    )
    disp_closure = disp_qoi.step_contribution(params, fe_problem.kernel_arrays)
    _coords, eq = dof_physical_coords(fe_problem.mesh, fe_problem.dof_map, "u")
    num_dofs = fe_problem.dof_map.num_total_dofs
    if temp_section is not None:
        temp_qoi = FETemperatureMatch.from_deck(
            temp_section, fe_problem, times.tolist(),
        )
        temp_closure = temp_qoi.step_contribution(
            params, fe_problem.kernel_arrays)
        _coords, eq_T = dof_physical_coords(
            fe_problem.mesh, fe_problem.dof_map, "T")
    disp_term = np.zeros(times.size)
    temp_term = np.zeros(times.size)
    for k in range(times.size):
        step_time = StepTime(float(times[k]), float(times[max(k - 1, 0)]))
        U_flat = np.zeros(num_dofs)
        U_flat[eq.reshape(-1)] = u_pred[k].reshape(-1)
        if temp_section is not None:
            U_flat[eq_T.reshape(-1)] = T_pred[k]
        U = jnp.asarray(U_flat)
        disp_term[k] = float(disp_closure(U, U, {}, {}, step_time))
        if temp_section is not None:
            temp_term[k] = float(temp_closure(U, U, {}, {}, step_time))
    disp_rms = spatial_rms(disp_term, disp_qoi, dt, span)
    if temp_section is not None:
        temp_rms = spatial_rms(temp_term, temp_qoi, dt, span)

    overlay_specs = [
        FieldSpec("u_pred", VarType.VECTOR),
        FieldSpec("u_meas", VarType.VECTOR),
        FieldSpec("u_diff", VarType.VECTOR),
    ]
    if temp_section is not None:
        overlay_specs += [
            FieldSpec("T_pred", VarType.SCALAR),
            FieldSpec("T_meas", VarType.SCALAR),
            FieldSpec("T_diff", VarType.SCALAR),
        ]
    with ExodusWriter(
        out_dir / "compare.exo", fe_problem.mesh,
        nodal_field_specs=overlay_specs,
    ) as writer:
        for k in range(times.size):
            nodal_data = {
                "u_pred": u_pred[k],
                "u_meas": u_meas[k],
                "u_diff": u_pred[k] - u_meas[k],
            }
            if temp_section is not None:
                nodal_data.update({
                    "T_pred": T_pred[k],
                    "T_meas": T_meas[k],
                    "T_diff": T_pred[k] - T_meas[k],
                })
            writer.write_step(float(times[k]), nodal_data=nodal_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for c, comp in enumerate(components):
        axes[0].plot(times, load_meas[:, c] / 1e3, "o", ms=3,
                     label=f"measured {comp}")
        axes[0].plot(times, reaction[:, c] / 1e3, "-", lw=1.2,
                     label=f"predicted {comp}")
        axes[1].plot(times, (reaction[:, c] - load_meas[:, c]) / 1e3,
                     "o-", ms=3, lw=0.9)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("load (kN)")
    axes[0].legend(fontsize=8)
    axes[0].grid(lw=0.3, alpha=0.4)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("predicted load - measured load (kN)")
    axes[1].grid(lw=0.3, alpha=0.4)
    fig.suptitle(problem_name)
    fig.tight_layout()
    fig.savefig(out_dir / "load.png", dpi=300)
    plt.close(fig)

    num_axes = 2 if temp_section is None else 3
    fig, axes = plt.subplots(1, num_axes, figsize=(6 * num_axes, 4.5))
    axes[0].plot(times, disp_term, "o-", ms=3, lw=0.9, label="displacement term")
    axes[0].plot(times, load_term, "o-", ms=3, lw=0.9, label="load term")
    if temp_section is not None:
        axes[0].plot(times, temp_term, "o-", ms=3, lw=0.9,
                     label="temperature term")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("contribution to the relative error per frame")
    axes[0].legend(fontsize=8)
    axes[0].grid(lw=0.3, alpha=0.4)
    axes[1].plot(times, disp_rms, "o-", ms=3, lw=0.9)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("spatial rms of the u mismatch (mm)")
    axes[1].grid(lw=0.3, alpha=0.4)
    if temp_section is not None:
        axes[2].plot(times, temp_rms, "o-", ms=3, lw=0.9)
        axes[2].set_xlabel("time (s)")
        axes[2].set_ylabel("spatial rms of the T mismatch (K)")
        axes[2].grid(lw=0.3, alpha=0.4)
    fig.suptitle(problem_name)
    fig.tight_layout()
    fig.savefig(out_dir / "terms.png", dpi=300)
    plt.close(fig)

    columns = [times, reaction, load_meas, disp_term, load_term, disp_rms]
    header = ("time (s), reaction (N), measured load (N), "
              "displacement term, load term, spatial rms (mm)")
    if temp_section is not None:
        columns += [temp_term, temp_rms]
        header += ", temperature term, spatial rms (K)"
    np.savetxt(out_dir / "compare.csv", np.column_stack(columns), header=header)

    # Each term's total, its weight, and the scale of its data: the rms of
    # the measured field, load, or temperature rise.
    assert disp_qoi.data_mean_square is not None
    totals = [
        ("displacement", disp_section, float(disp_term.sum()),
         f"{np.sqrt(disp_qoi.data_mean_square):.3f} mm"),
        ("load", load_section, float(load_term.sum()),
         f"{np.sqrt(load_qoi.data_mean_square) / 1e3:.2f} kN"),
    ]
    if temp_section is not None:
        assert temp_qoi.data_mean_square is not None
        totals.append((
            "temperature", temp_section, float(temp_term.sum()),
            f"{np.sqrt(temp_qoi.data_mean_square):.2f} K",
        ))
    qoi_name = bundle.resolved["qoi"]["name"]
    J = 0.0
    weights = []
    for name, section, total, scale in totals:
        weight = float(section.get("weight", 1.0))
        weights.append(f"{weight:g}")
        J += weight * (np.log(total) if qoi_name == "fe_log_sum" else total)
        print(f"{name}: relative mean square {total:.6e}, "
              f"rms {100.0 * np.sqrt(total):.2f} % of {scale}")
    print(f"J ({qoi_name}, weights {', '.join(weights)}) {J:.6e}")
    print(f"wrote {out_dir}/compare.exo, load.png, terms.png, compare.csv")


if __name__ == "__main__":
    main()
