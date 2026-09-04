"""Solve times and match times from a calibration data archive.

Writes a solve times file and a match times file beside the archive.

Usage:
    python examples/jones_304l_schedule.py \
        --archive data/jones_304l/jones_304l_o5_2d_y40_132_h3_calibration_data.npz \
        --num-steps 100
    python examples/jones_304l_schedule.py --archive ... --num-steps 100 --select force
    python examples/jones_304l_schedule.py --archive ... --times my_solve_times.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from jones_304l_dic_data import select_frames

from cmad.io.calibration_data import CalibrationData


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", required=True,
        help="a *_calibration_data.npz written by jones_304l_dic_data.py",
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--num-steps", type=int,
        help="number of frames to select from the archive",
    )
    which.add_argument(
        "--times",
        help="custom solve times, one per line; every entry must be an "
             "archive frame time (the t = 0 reference is added when absent)",
    )
    parser.add_argument(
        "--select", default="index", choices=("index", "force"),
        help="spread the selection evenly over the frame order or over "
             "the load (default index); extension is not in the archive",
    )
    parser.add_argument(
        "--out-solve-times", default=None,
        help="output path (default <archive stem>_solve_times_<tag>.txt)",
    )
    parser.add_argument(
        "--out-match-times", default=None,
        help="output path (default <archive stem>_match_times_<tag>.txt)",
    )
    args = parser.parse_args()

    store = CalibrationData.read(args.archive)
    if args.times is not None:
        wanted = np.unique(np.loadtxt(args.times, dtype=np.float64).ravel())
        rows = np.asarray(
            [store.frame_at(t) for t in wanted], dtype=np.intp,
        )
        schedule = store.times[rows]
        if schedule[0] != 0.0:
            schedule = np.concatenate([[0.0], schedule])
        tag = Path(args.times).stem
    else:
        measured = np.flatnonzero(store.frame_ids >= 0).astype(np.intp)
        by = None if args.select == "index" else store.load
        rows = select_frames(measured, args.num_steps, by=by)
        schedule = np.concatenate([[0.0], store.times[rows]])
        tag = f"n{args.num_steps}"

    stem = str(args.archive).removesuffix("_calibration_data.npz")
    solve_path = Path(args.out_solve_times or f"{stem}_solve_times_{tag}.txt")
    match_path = Path(args.out_match_times or f"{stem}_match_times_{tag}.txt")
    np.savetxt(solve_path, schedule)
    np.savetxt(match_path, schedule)

    print(
        f"{schedule.size} solve times (reference included), "
        f"t {schedule[1]:g} to {schedule[-1]:g}, "
        f"load {store.load[rows][0]:g} to {store.load[rows][-1]:g}"
    )
    print(f"wrote {solve_path}")
    print(f"      {match_path}")


if __name__ == "__main__":
    main()
