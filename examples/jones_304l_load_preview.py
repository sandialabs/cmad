"""Preview the Jones et al. 304L load record and choose a frame range.

The measured record runs past both ends of what is worth modeling: it
starts before the test does, and it ends at specimen failure, which no
material model here represents. This plots load against time and against
extension, shades a candidate frame range, and reports what each bound
trims, so the range can be chosen by looking rather than by guessing.

Time is zeroed at the test start, so negative times are the pre-test
window. Load is read from the 6DOF cell when the specimen has one and
from the single uniaxial cell otherwise, and is absent at both ends of
the record. Frames sitting in the record's spurious force drops are
excluded by default; see :func:`load_drop_frames`. A second figure of
zoom panels shows the kept range's excluded drop events for review by
eye, the kept neighbor frames visible around each.

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
from matplotlib.ticker import MaxNLocator, MultipleLocator
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import median_filter

# Row of force6DOF along the loading axis. Rows 0-2 are forces, 3-5 moments.
LOAD_ROW_FY = 1

# Rolling median width and default threshold, in kN, for the force drop
# artifacts: the slow tests carry occasional drops that recover with
# continued loading, which the dataset attributes to the anti-rotate bars
# catching on their guides and recommends removing before calibration.
# Measured on XT10 and O5-4, the artifact frames deviate 0.12 to 1.5 kN,
# in both directions, against a residual noise floor of 0.08 kN.
MEDIAN_WIDTH = 5
DROP_THRESHOLD = 0.1


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


def load_drop_frames(
        load: NDArray[np.float64],
        candidates: NDArray[np.intp],
        threshold: float = DROP_THRESHOLD,
) -> NDArray[np.intp]:
    """Frames inside a spurious force artifact, a subset of ``candidates``.

    A frame is excluded when its load deviates from the rolling median
    of ``MEDIAN_WIDTH`` neighbors by more than ``threshold``, in either
    direction. The worst deviation is flagged first, and flagged values
    are replaced by interpolation before the median is recomputed, so an
    artifact does not drag the reference and take healthy neighbors with
    it; flagging repeats until nothing exceeds the threshold. A
    threshold of zero or below excludes nothing.
    """
    if threshold <= 0.0 or candidates.size == 0:
        return np.zeros(0, dtype=np.intp)
    flagged, _deviation = _drop_flags(load[candidates], threshold)
    return candidates[flagged]


def merge_hand_picks(
        dropped: NDArray[np.intp],
        usable: NDArray[np.intp],
        picks: list[int] | None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Union frames excluded by hand into ``dropped``.

    Returns the widened ``dropped`` and the picks as an array. Picks
    that are not usable frames raise, since a typo would otherwise be
    silently ignored.
    """
    if not picks:
        return dropped, np.zeros(0, dtype=np.intp)
    chosen = np.asarray(sorted(set(picks)), dtype=np.intp)
    unknown = np.setdiff1d(chosen, usable)
    if unknown.size:
        raise ValueError(
            f"exclude frames {unknown.tolist()} are not usable frames"
        )
    return np.union1d(dropped, chosen).astype(np.intp), chosen


def _drop_flags(
        series: NDArray[np.float64], threshold: float,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Flags and each frame's deviation from the artifact free median."""
    position = np.arange(series.size, dtype=np.float64)
    flagged = np.zeros(series.size, dtype=bool)
    deviation = np.zeros_like(series)
    while not flagged.all():
        clean = series.copy()
        if flagged.any():
            clean[flagged] = np.interp(
                position[flagged], position[~flagged], series[~flagged],
            )
        deviation = series - median_filter(
            clean, size=MEDIAN_WIDTH, mode="nearest",
        )
        probe = np.abs(deviation)
        probe[flagged] = 0.0
        worst = int(probe.argmax())
        if probe[worst] <= threshold:
            break
        flagged[worst] = True
    return flagged, deviation


def plot_drop_review(
        out_path: Path,
        load: NDArray[np.float64],
        usable: NDArray[np.intp],
        kept: NDArray[np.intp],
        threshold: float,
        *,
        by_hand: NDArray[np.intp] | None = None,
        max_panels: int = 24,
        join_gap: int = 3,
        context: int = 10,
) -> int:
    """Zoom panels of the kept range's excluded drop events.

    One panel per excluded event, circled, with the kept neighbor frames
    visible around it. Returns the number of events drawn.
    """
    if threshold <= 0.0 or usable.size == 0 or kept.size == 0:
        return 0
    series = load[usable]
    flagged, deviation = _drop_flags(series, threshold)
    if by_hand is not None and by_hand.size:
        flagged = flagged | np.isin(usable, by_hand)
    notable = flagged & (usable >= kept[0]) & (usable <= kept[-1])
    idx = np.flatnonzero(notable)
    if idx.size == 0:
        return 0
    splits = np.flatnonzero(np.diff(idx) > join_gap)
    starts = np.concatenate([[0], splits + 1])
    stops = np.concatenate([splits, [idx.size - 1]])
    events = [
        (int(idx[a]), int(idx[b]))
        for a, b in zip(starts, stops, strict=True)
    ]
    if len(events) > max_panels:
        largest = sorted(
            events,
            key=lambda ev: float(np.abs(deviation[ev[0]:ev[1] + 1]).max()),
            reverse=True,
        )[:max_panels]
        print(
            f"drop review: drawing the {max_panels} largest of "
            f"{len(events)} events"
        )
        events = sorted(largest)

    median = series - deviation
    ncols = min(len(events), 3)
    nrows = (len(events) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False,
    )
    for k, (a, b) in enumerate(events):
        ax = axes[k // ncols][k % ncols]
        lo, hi = max(a - context, 0), min(b + context, series.size - 1)
        span = np.arange(lo, hi + 1)
        ax.plot(usable[span], median[span], "--", lw=0.9, color="#999999",
                label="artifact free median")
        ax.plot(usable[span], series[span], "o-", ms=4, lw=0.9,
                color="#1f77b4", label="load")
        bad = span[flagged[span]]
        if bad.size:
            ax.plot(usable[bad], series[bad], "o", ms=7, mfc="none",
                    color="#d62728", label="excluded")
        window = deviation[a:b + 1]
        extreme = window[np.abs(window).argmax()]
        ax.set_title(
            f"{usable[a]}..{usable[b]}  largest deviation {extreme:+.3f} kN",
            fontsize=9,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.grid(True, which="minor", axis="x", lw=0.2, alpha=0.25)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(len(events), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return len(events)


def _frame_spans(frames: NDArray[np.intp]) -> list[tuple[int, int]]:
    """Consecutive frame numbers grouped into ``(first, last)`` spans."""
    if frames.size == 0:
        return []
    splits = np.flatnonzero(np.diff(frames) > 1)
    starts = np.concatenate([[0], splits + 1])
    stops = np.concatenate([splits, [frames.size - 1]])
    return [
        (int(frames[a]), int(frames[b]))
        for a, b in zip(starts, stops, strict=True)
    ]


def drop_summary(dropped: NDArray[np.intp]) -> str:
    """One line naming the excluded force artifact frames."""
    spans = _frame_spans(dropped)
    joined = ", ".join(f"{a}..{b}" if b > a else f"{a}" for a, b in spans)
    return (
        f"excluded {dropped.size} frame(s) in {len(spans)} force artifact "
        f"event(s): {joined}"
    )


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
        dropped: NDArray[np.intp],
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
    if dropped.size:
        lines.insert(2, drop_summary(dropped))
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

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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
        dropped: NDArray[np.intp] | None = None,
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
    if dropped is not None and dropped.size:
        ax.plot(times[dropped], load[dropped], "x", ms=6, mew=1.6,
                color="#d62728", label=f"excluded ({dropped.size})")
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
        "--drop-threshold", type=float, default=DROP_THRESHOLD,
        help=f"exclude frames whose load deviates from a rolling median "
             f"by more than this, in kN; 0 keeps every frame "
             f"(default {DROP_THRESHOLD:g})",
    )
    parser.add_argument(
        "--exclude-frames", type=int, nargs="+", default=None,
        help="frames to exclude by hand, in addition to the detected drops",
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
    usable = usable_frames(times, load)
    dropped = load_drop_frames(load, usable, args.drop_threshold)
    dropped, by_hand = merge_hand_picks(dropped, usable, args.exclude_frames)
    candidates = np.setdiff1d(usable, dropped).astype(np.intp)
    kept = resolve_range(candidates, args.frame_min, args.frame_max)

    out = Path(
        args.out or f"scratch/load_preview_f{kept[0]}_{kept[-1]}.png"
    )
    plot_record(
        out, times, load, extension, candidates, kept,
        zoom_span=args.zoom_span, dropped=dropped,
    )
    print(report(times, load, extension, candidates, kept, dropped, channel))
    if by_hand.size:
        print(f"excluded by hand: {', '.join(str(f) for f in by_hand)}")
    print(f"wrote {out}")
    review = out.with_name(out.stem + "_drops" + out.suffix)
    n_events = plot_drop_review(
        review, load, usable, kept, args.drop_threshold, by_hand=by_hand,
    )
    if n_events:
        print(f"wrote {review} ({n_events} drop event panel(s) in range)")


if __name__ == "__main__":
    main()
