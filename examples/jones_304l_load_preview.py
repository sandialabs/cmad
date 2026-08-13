"""Preview the Jones et al. 304L load record and choose a frame range.

The measured record runs past both ends of what is worth modeling: it
starts before the test does, and it ends at specimen failure, which no
material model here represents. This plots load against time and against
extension, shades a candidate frame range, and reports what each bound
trims, so the range can be chosen by looking rather than by guessing.

Time is zeroed at the test start, so negative times are the pre-test
window. Load is read from the 6DOF cell when the specimen has one and
from the single uniaxial cell otherwise, and is absent at both ends of
the record.

The chosen bounds carry to jones_304l_dic_data.py. The undeformed
reference state is separate from this choice: it is a constructed entry
at t = 0 with zero load, so trimming early frames never costs it.

Usage:
    python examples/jones_304l_load_preview.py
    python examples/jones_304l_load_preview.py --frame-min 60 --frame-max 560
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

# Row of force6DOF along the loading axis. Rows 0-2 are forces, 3-5 moments.
LOAD_ROW_FY = 1


def read_global_channels(
        path: str | Path, load_row: int = LOAD_ROW_FY,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], str]:
    """Read ``(times, load, extension, channel_name)``.

    Prefers the 6DOF load cell, which ships with only some specimens, and
    falls back to the single uniaxial cell.
    """
    with h5py.File(Path(path), "r") as handle:
        times = np.asarray(handle["time"][0, :], dtype=np.float64)
        extension = np.asarray(handle["extension"][0, :], dtype=np.float64)
        if "force6DOF" in handle:
            load = np.asarray(handle["force6DOF"][load_row, :], np.float64)
            channel = f"force6DOF row {load_row} (Fy)"
        else:
            load = np.asarray(handle["force"][0, :], dtype=np.float64)
            channel = "force"
    return times, load, extension, channel


def usable_frames(
        times: NDArray[np.float64], load: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Frames carrying a load reading at or after the start of the test.

    The lower bound is the pre-test window, the upper is where the record
    ends at failure. Both are read off the data rather than tabulated, so
    they carry to the other specimens.
    """
    return np.flatnonzero(np.isfinite(load) & (times >= 0.0)).astype(np.intp)


def resolve_range(
        candidates: NDArray[np.intp],
        frame_min: int | None,
        frame_max: int | None,
) -> NDArray[np.intp]:
    """Apply the requested bounds to the usable frames."""
    lo = candidates[0] if frame_min is None else int(frame_min)
    hi = candidates[-1] if frame_max is None else int(frame_max)
    if lo > hi:
        raise ValueError(f"frame-min ({lo}) is above frame-max ({hi})")
    kept = candidates[(candidates >= lo) & (candidates <= hi)]
    if kept.size == 0:
        raise ValueError(
            f"frames {lo} to {hi} contain none of the usable range "
            f"{candidates[0]} to {candidates[-1]}"
        )
    return kept


def report(
        times: NDArray[np.float64],
        load: NDArray[np.float64],
        extension: NDArray[np.float64],
        candidates: NDArray[np.intp],
        kept: NDArray[np.intp],
        channel: str,
) -> str:
    """Describe the record and what the chosen bounds trim."""
    dropped_low = candidates[candidates < kept[0]]
    dropped_high = candidates[candidates > kept[-1]]
    lines = [
        f"load channel: {channel}",
        f"record {times.size} frames, "
        f"{int(np.sum(~np.isfinite(load)))} without a load reading",
        f"usable frames {candidates[0]} to {candidates[-1]} "
        f"({candidates.size}), time {times[candidates[0]]:.1f} to "
        f"{times[candidates[-1]]:.1f} s",
        f"keeping frames {kept[0]} to {kept[-1]} ({kept.size}), "
        f"load {load[kept].min():.4f} to {load[kept].max():.4f}, "
        f"extension {extension[kept].min():.4f} to "
        f"{extension[kept].max():.4f}",
    ]
    if dropped_low.size:
        lines.append(
            f"  trimmed {dropped_low.size} at the start, up to load "
            f"{load[dropped_low].max():.4f} and extension "
            f"{extension[dropped_low].max():.4f}"
        )
    if dropped_high.size:
        lines.append(
            f"  trimmed {dropped_high.size} at the end, from load "
            f"{load[dropped_high[0]]:.4f}"
        )
    peak = int(candidates[load[candidates].argmax()])
    lines.append(
        f"largest load sample {load[peak]:.4f} at frame {peak}, "
        f"t {times[peak]:.1f} s, with {int(np.sum(candidates > peak))} usable "
        f"frames past it"
    )
    negative = candidates[load[candidates] < 0.0]
    if negative.size:
        lines.append(
            f"  {negative.size} usable frames carry negative load, up to "
            f"frame {int(negative.max())}"
        )
    return "\n".join(lines)


def _plot_end(
        ax: plt.Axes,
        times: NDArray[np.float64],
        load: NDArray[np.float64],
        candidates: NDArray[np.intp],
        kept: NDArray[np.intp],
        *,
        at_start: bool,
        span: int,
) -> None:
    """Zoom on one end of the kept range, one marker per frame.

    Drawn from the usable frames rather than the kept ones so the frames
    just outside the bound are visible too, and with markers so a single
    stray sample reads as a point rather than disappearing into a line.
    """
    bound = kept[0] if at_start else kept[-1]
    near = candidates[np.abs(candidates - bound) <= span]
    inside = (near >= kept[0]) & (near <= kept[-1])
    # Frame index on the axis, since that is what the bounds are given
    # in; time rides along on top for physical context.
    ax.plot(near, load[near], "-", lw=0.8, color="#bbbbbb")
    ax.plot(near[~inside], load[near[~inside]], "o", ms=3.5,
            color="#bbbbbb", label="trimmed")
    ax.plot(near[inside], load[near[inside]], "o", ms=3.5,
            color="#1f77b4", label="kept")
    ax.axvline(bound, color="#ff7f0e", ls="--", lw=1.2)

    frames = np.arange(times.size, dtype=np.float64)

    def frame_to_time(value: ArrayLike) -> NDArray[np.float64]:
        return np.interp(np.asarray(value, dtype=np.float64), frames, times)

    def time_to_frame(value: ArrayLike) -> NDArray[np.float64]:
        return np.interp(np.asarray(value, dtype=np.float64), times, frames)

    ax.secondary_xaxis(
        "top", functions=(frame_to_time, time_to_frame),
    ).set_xlabel("time (s)")

    ax.set_xlabel("frame index")
    ax.set_ylabel("load (kN)")
    ax.set_title(
        f"{'start' if at_start else 'end'} of range, frame {int(bound)} "
        f"at t {times[bound]:.1f} s"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.4)


def plot_record(
        out_path: Path,
        times: NDArray[np.float64],
        load: NDArray[np.float64],
        extension: NDArray[np.float64],
        candidates: NDArray[np.intp],
        kept: NDArray[np.intp],
        *,
        zoom_span: int = 25,
) -> None:
    """Plot the record, the chosen range, and a zoom on each of its ends."""
    finite = np.isfinite(load)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.plot(times[finite], load[finite], "-", lw=0.9, color="#bbbbbb",
            label="whole record")
    ax.plot(times[candidates], load[candidates], "-", lw=1.1, color="#7fb3d5",
            label=f"usable ({candidates.size})")
    ax.plot(times[kept], load[kept], "-", lw=1.8, color="#1f77b4",
            label=f"kept ({kept.size})")
    ax.axvspan(times[kept[0]], times[kept[-1]], color="#2ca02c", alpha=0.10)
    for frame, name in ((kept[0], "frame-min"), (kept[-1], "frame-max")):
        ax.axvline(times[frame], color="#ff7f0e", ls="--", lw=1.2)
        ax.annotate(
            f"{name} = {int(frame)}", xy=(times[frame], load[candidates].max()),
            xytext=(3, -10), textcoords="offset points", fontsize=8,
            color="#d2691e", rotation=90, va="top",
        )
    ax.axhline(0.0, color="#999999", lw=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("load (kN)")
    ax.set_title("load against time")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.4)

    ax = axes[0, 1]
    ax.plot(extension[finite], load[finite], "-", lw=0.9, color="#bbbbbb")
    ax.plot(extension[candidates], load[candidates], "-", lw=1.1,
            color="#7fb3d5")
    ax.plot(extension[kept], load[kept], "-", lw=1.8, color="#1f77b4")
    ax.plot(extension[[kept[0], kept[-1]]], load[[kept[0], kept[-1]]], "o",
            ms=5, color="#ff7f0e")
    ax.axhline(0.0, color="#999999", lw=0.6)
    ax.set_xlabel("extension (mm)")
    ax.set_ylabel("load (kN)")
    ax.set_title("load against extension")
    ax.grid(True, lw=0.3, alpha=0.4)

    _plot_end(axes[1, 0], times, load, candidates, kept,
              at_start=True, span=zoom_span)
    _plot_end(axes[1, 1], times, load, candidates, kept,
              at_start=False, span=zoom_span)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/jones_304l/XT10-Data.mat", help="DIC data .mat",
    )
    parser.add_argument(
        "--frame-min", type=int, default=None,
        help="first frame to keep (default: the first usable one)",
    )
    parser.add_argument(
        "--frame-max", type=int, default=None,
        help="last frame to keep (default: the last usable one)",
    )
    parser.add_argument(
        "--load-component", type=int, default=LOAD_ROW_FY,
        help=f"force6DOF row to read (default {LOAD_ROW_FY}, Fy)",
    )
    parser.add_argument(
        "--zoom-span", type=int, default=25,
        help="frames either side of each bound in the zoom panels "
             "(default 25)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output image (default scratch/load_preview_f{min}_{max}.png)",
    )
    args = parser.parse_args()

    times, load, extension, channel = read_global_channels(
        args.data, args.load_component,
    )
    candidates = usable_frames(times, load)
    kept = resolve_range(candidates, args.frame_min, args.frame_max)

    out = Path(
        args.out or f"scratch/load_preview_f{kept[0]}_{kept[-1]}.png"
    )
    plot_record(
        out, times, load, extension, candidates, kept,
        zoom_span=args.zoom_span,
    )
    print(report(times, load, extension, candidates, kept, channel))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
