"""Compare a Jones 304L primal run against the measured record.

Reads the calibrate input file, the primal's exodus, and the reaction
CSV the primal wrote. Writes an overlay exodus with u_pred, u_meas,
and u_diff on the stored nodes, a figure of predicted against
measured load, a figure of the match terms per frame, and a CSV of
the values behind the figures.

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
from cmad.io.calibration_data import CalibrationData
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.results import FieldSpec
from cmad.models.global_fields import StepTime
from cmad.models.var_types import VarType
from cmad.qois.fe_displacement_match import FEDisplacementMatch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deck", required=True,
        help="the calibrate input file naming the mesh, the archive, "
             "the weights, and the solve times",
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
    terms = {t["name"]: t for t in bundle.resolved["qoi"]["terms"]}
    disp_section = terms["fe_displacement_match"]
    load_section = terms["fe_load_match"]
    store = CalibrationData.read(disp_section["calibration_data_file"])

    results = read_results(
        args.exodus,
        nodal_field_specs=[FieldSpec("u", VarType.VECTOR)],
    )
    times = np.asarray(results.time, dtype=np.float64)
    u_pred = np.asarray(results.nodal["u"], dtype=np.float64)
    n_nodes = fe_problem.mesh.nodes.shape[0]
    ndims = int(fe_problem.mesh.nodes.shape[1])

    # The measured field on the stored nodes, NaN elsewhere.
    rows = store.frames_at(times)
    measured = np.asarray(store.rows(rows, store.node_ids))
    u_meas = np.full((times.size, n_nodes, ndims), np.nan)
    u_meas[:, store.node_ids] = measured

    # The load term per frame, from the reaction series and the archive:
    # (w / T) dt_n sum_c (R_cn - d_cn)^2 with dt_1 = 0.
    reaction = np.loadtxt(args.reaction, delimiter=",").reshape(times.size, -1)
    load_meas = np.asarray(store.load[rows], dtype=np.float64)
    components = [int(c) for c in load_section["components"]]
    if load_meas.ndim == 1:
        load_meas = load_meas.reshape(-1, 1)
    dt = np.concatenate([[0.0], np.diff(times)])
    span = float(times[-1] - times[0])
    load_sq = np.sum((reaction - load_meas) ** 2, axis=1)
    load_term = float(load_section["weight"]) / span * dt * load_sq

    # The displacement term per frame, through the QoI's own closure.
    disp_qoi = FEDisplacementMatch.from_deck(
        disp_section, fe_problem, times.tolist(),
    )
    closure = disp_qoi.step_contribution(
        params_by_block_from_models(fe_problem), fe_problem.kernel_arrays,
    )
    _coords, eq = dof_physical_coords(fe_problem.mesh, fe_problem.dof_map, "u")
    num_dofs = fe_problem.dof_map.num_total_dofs
    # The spatial rms of the mismatch over the region of interest,
    # rms_k = sqrt(integral |u_pred - u_meas|^2 dV / V), from the same
    # closure: a second evaluation at the data plus 1 mm in x has the
    # integrand 1, so the ratio of the two is the mean square and the
    # normalization and the dt weight cancel.
    unit = np.zeros(num_dofs)
    unit[eq[:, 0]] = 1.0
    node_eq = eq[store.node_ids].reshape(-1)
    disp_term = np.zeros(times.size)
    disp_rms = np.zeros(times.size)
    for k in range(times.size):
        step_time = StepTime(float(times[k]), float(times[max(k - 1, 0)]))
        U_flat = np.zeros(num_dofs)
        U_flat[eq.reshape(-1)] = u_pred[k].reshape(-1)
        disp_term[k] = float(closure(
            jnp.asarray(U_flat), jnp.asarray(U_flat), {}, {}, step_time,
        ))
        if k == 0:
            continue
        data_flat = np.zeros(num_dofs)
        data_flat[node_eq] = measured[k].reshape(-1)
        unit_term = float(closure(
            jnp.asarray(data_flat + unit), jnp.asarray(data_flat + unit),
            {}, {}, step_time,
        ))
        disp_rms[k] = float(np.sqrt(disp_term[k] / unit_term))

    with ExodusWriter(
        out_dir / "compare.exo", fe_problem.mesh,
        nodal_field_specs=[
            FieldSpec("u_pred", VarType.VECTOR),
            FieldSpec("u_meas", VarType.VECTOR),
            FieldSpec("u_diff", VarType.VECTOR),
        ],
    ) as writer:
        for k in range(times.size):
            writer.write_step(
                float(times[k]),
                nodal_data={
                    "u_pred": u_pred[k],
                    "u_meas": u_meas[k],
                    "u_diff": u_pred[k] - u_meas[k],
                },
            )

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
    fig.tight_layout()
    fig.savefig(out_dir / "load.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(times, disp_term, "o-", ms=3, lw=0.9, label="displacement term")
    axes[0].plot(times, load_term, "o-", ms=3, lw=0.9, label="load term")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("term contribution per frame")
    axes[0].legend(fontsize=8)
    axes[0].grid(lw=0.3, alpha=0.4)
    axes[1].plot(times, disp_rms, "o-", ms=3, lw=0.9)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("spatial rms of the u mismatch (mm)")
    axes[1].grid(lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "terms.png", dpi=160)
    plt.close(fig)

    table = np.column_stack(
        [times, reaction, load_meas, disp_term, load_term, disp_rms],
    )
    np.savetxt(
        out_dir / "compare.csv", table,
        header="time (s), reaction (N), measured load (N), "
               "displacement term, load term, spatial rms (mm)",
    )

    print(f"displacement term total {disp_term.sum():.6e} "
          f"(weight {disp_section['weight']:g})")
    print(f"load term total {load_term.sum():.6e} "
          f"(weight {load_section['weight']:g})")
    print(f"J {disp_term.sum() + load_term.sum():.6e}")
    print(f"wrote {out_dir}/compare.exo, load.png, terms.png, compare.csv")


if __name__ == "__main__":
    main()
