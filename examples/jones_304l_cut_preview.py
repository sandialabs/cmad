"""Preview candidate model cuts against the Jones et al. 304L DIC data.

The coupon outline covers the whole specimen, ends included, while the
model covers a window of it bounded by two cuts in y. The cuts belong
inside the region where DIC data exists, so that the Dirichlet edges have
data to inform them. This plots the outline, the measured point cloud,
and a candidate pair of cuts together, and reports how much data sits
outside each cut and how far the data reaches across each cut edge.

Choose the frame range with jones_304l_load_preview.py first: the
validity shown here is the points valid in every frame of that range,
after the same load processing, since the Dirichlet edges need data at
every frame a run uses. ``--frame`` overrides with a single frame's
validity instead.

Either cut may be given; whichever is omitted defaults to the midpoint of
the outermost region where the section width is constant and the data
reaches, rounded to a whole unit. Cut values are in the coordinate system
of the geometry file, which is also the system the DIC coordinates are
in.

Usage:
    python examples/jones_304l_cut_preview.py
    python examples/jones_304l_cut_preview.py --y-min 62 --y-max 182
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from jones_304l_geometry import (
    Loop,
    clean_loop,
    count_crossings,
    default_cuts,
    loop_area,
    prismatic_windows,
    read_nominal_geometry,
    trim_outline,
)
from jones_304l_load_preview import (
    DROP_THRESHOLD,
    load_drop_frames,
    merge_hand_picks,
    read_global_channels,
    resolve_range,
    usable_frames,
)
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from numpy.typing import NDArray

# Uncorrelated points carry a negative confidence, stored as -1, with NaN
# displacements at the same indices. Any value below this threshold is
# treated as invalid, which also rejects NaN.
MIN_VALID_SIGMA = 0.0


def read_dic_reference(
        path: str | Path, frame: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Read reference coordinates and one frame's validity mask.

    Returns ``(x, y, valid)`` for the ``frame``-th frame. The data file
    stores arrays transposed relative to MATLAB, so the frame is the
    leading axis here. Only three slices are read, which keeps this cheap
    on a multi-gigabyte file.
    """
    with h5py.File(Path(path), "r") as handle:
        x = np.asarray(handle["X0"][0, :], dtype=np.float64)
        y = np.asarray(handle["Y0"][0, :], dtype=np.float64)
        n_frames = handle["sigma"].shape[0]
        if not -n_frames <= frame < n_frames:
            raise IndexError(
                f"frame {frame} is outside the {n_frames} frames in {path}"
            )
        sigma = np.asarray(handle["sigma"][frame, :], dtype=np.float64)
    return x, y, sigma >= MIN_VALID_SIGMA


def range_validity(
        path: str | Path, frames: NDArray[np.intp],
) -> NDArray[np.bool_]:
    """Points valid in every one of ``frames``.

    Reads the dataset in blocks along the point axis: the file's chunks
    span every frame of a few points, so a frame by frame read would
    decompress the whole dataset once per frame.
    """
    # Bounds memory only: any block of a few thousand points reads the
    # chunks exactly once.
    point_block = 8192
    with h5py.File(Path(path), "r") as handle:
        sigma = handle["sigma"]
        n_frames, n_points = sigma.shape
        rows = np.zeros(n_frames, dtype=bool)
        rows[frames] = True
        valid = np.empty(n_points, dtype=bool)
        for start in range(0, n_points, point_block):
            stop = min(start + point_block, n_points)
            piece = np.asarray(sigma[:, start:stop], dtype=np.float64)
            valid[start:stop] = (piece[rows] >= MIN_VALID_SIGMA).all(axis=0)
    return valid


def resolve_cuts(
        outer: Loop, holes: list[Loop], y: NDArray[np.float64],
        valid: NDArray[np.bool_], y_min: float | None, y_max: float | None,
) -> tuple[float, float, list[tuple[float, float]], tuple[float, float]]:
    """Fill in whichever cuts were not given.

    Returns the pair to use, the usable windows, and the suggestion, so a
    caller can report the suggestion even when both cuts were supplied.
    """
    limits = (float(y[valid].min()), float(y[valid].max()))
    windows = prismatic_windows(outer, holes)
    suggested = default_cuts(outer, holes, y_limits=limits)
    return (
        suggested[0] if y_min is None else y_min,
        suggested[1] if y_max is None else y_max,
        windows,
        suggested,
    )


def cut_report(
        x: NDArray[np.float64], y: NDArray[np.float64],
        valid: NDArray[np.bool_], y_min: float, y_max: float,
        *, band: float = 1.0,
) -> str:
    """Describe how the valid data sits relative to a candidate cut pair."""
    xv, yv = x[valid], y[valid]
    inside = (yv >= y_min) & (yv <= y_max)
    lines = [
        f"valid points {valid.sum()} of {valid.size}"
        f"   spanning x [{xv.min():.4g}, {xv.max():.4g}]"
        f"  y [{yv.min():.4g}, {yv.max():.4g}]",
        f"inside the cuts {inside.sum()}"
        f"   below y_min {int((yv < y_min).sum())}"
        f"   above y_max {int((yv > y_max).sum())}",
        f"margin below y_min {y_min - yv.min():.4g} mm"
        f"   above y_max {yv.max() - y_max:.4g} mm",
    ]
    # Coverage across each cut edge: a Dirichlet edge with data missing at
    # its ends is informed only over part of its length.
    for name, cut in (("y_min", y_min), ("y_max", y_max)):
        near = np.abs(yv - cut) <= band
        if near.any():
            lines.append(
                f"at {name}={cut:g}, valid data within {band:g} mm spans "
                f"x [{xv[near].min():.4g}, {xv[near].max():.4g}] "
                f"({int(near.sum())} points)"
            )
        else:
            lines.append(f"at {name}={cut:g}, NO valid data within {band:g} mm")
    return "\n".join(lines)


def _region_patch(outer: Loop, holes: list[Loop], **kwargs: object) -> PathPatch:
    """A filled patch of ``outer`` with each hole removed.

    Hole loops are reversed to wind opposite the outer loop, so the fill
    excludes them under either winding rule.
    """
    outer_sign = np.sign(loop_area(outer))
    vertices, codes = [], []
    for i, loop in enumerate([outer, *holes]):
        if i > 0 and np.sign(loop_area(loop)) == outer_sign:
            loop = loop[::-1]
        vertices.append(np.vstack([loop, loop[:1]]))
        codes.append(
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(loop) - 1)
            + [MplPath.CLOSEPOLY]
        )
    path = MplPath(np.vstack(vertices), np.concatenate(codes))
    return PathPatch(path, **kwargs)


def plot_cut_preview(
        out_path: Path, outer: Loop, holes: list[Loop],
        x: NDArray[np.float64], y: NDArray[np.float64],
        valid: NDArray[np.bool_], y_min: float, y_max: float,
        *, mask_note: str,
) -> None:
    """Write the preview plot to ``out_path``."""
    fig, ax = plt.subplots(figsize=(7, 10))

    ax.scatter(x[valid], y[valid], s=0.4, c="#7fb3d5", linewidths=0,
               rasterized=True, label=f"DIC valid ({int(valid.sum())})")

    # The modeled region, with holes removed so it shows only material.
    if count_crossings(outer, y_min) == 2 and count_crossings(outer, y_max) == 2:
        trimmed = trim_outline(outer, y_min, y_max)
        kept = [
            hole for hole in holes
            if y_min <= hole[:, 1].min() and hole[:, 1].max() <= y_max
        ]
        ax.add_patch(_region_patch(
            trimmed, kept, facecolor="#2ca02c", alpha=0.15, edgecolor="none",
            zorder=0, label="modeled region",
        ))

    closed = np.vstack([outer, outer[:1]])
    ax.plot(closed[:, 0], closed[:, 1], "-", color="#222222", lw=1.2,
            label="nominal outline")
    for hole in holes:
        loop = np.vstack([hole, hole[:1]])
        ax.plot(loop[:, 0], loop[:, 1], "-", color="#222222", lw=1.2)

    x_hi = float(outer[:, 0].max())
    inset = 0.02 * (x_hi - float(outer[:, 0].min()))
    for name, cut in (("y_min", y_min), ("y_max", y_max)):
        ax.axhline(cut, color="#ff7f0e", ls="--", lw=1.4)
        ax.annotate(
            f"{name} = {cut:g}", xy=(x_hi - inset, cut), xytext=(0, 3),
            textcoords="offset points", va="bottom", ha="right",
            fontsize=9, color="#d2691e",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7,
                  "pad": 1.0},
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Candidate cuts; {mask_note}")
    ax.legend(loc="upper left", fontsize=8, markerscale=4, framealpha=0.9)
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry", default="data/jones_304l/X-Specimen-NominalGeometry.mat",
        help="nominal geometry .mat",
    )
    parser.add_argument(
        "--data", default="data/jones_304l/XT10-Data.mat", help="DIC data .mat",
    )
    parser.add_argument(
        "--y-min", type=float, default=None,
        help="lower cut (default: midpoint of the lowest usable window)",
    )
    parser.add_argument(
        "--y-max", type=float, default=None,
        help="upper cut (default: midpoint of the highest usable window)",
    )
    parser.add_argument(
        "--frame-min", type=int, default=None,
        help="first frame of the range (default: the first usable one)",
    )
    parser.add_argument(
        "--frame-max", type=int, default=None,
        help="last frame of the range (default: the last usable one)",
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
        "--frame", type=int, default=None,
        help="plot a single frame's validity instead of the range's",
    )
    parser.add_argument(
        "--band", type=float, default=1.0,
        help="half width in mm for the coverage check at each cut (default 1)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output image (default scratch/cut_preview_y{min}_{max}.png)",
    )
    args = parser.parse_args()

    outer, holes = read_nominal_geometry(args.geometry)
    outer = clean_loop(outer)
    holes = [clean_loop(hole) for hole in holes]
    x, y, valid = read_dic_reference(
        args.data, 0 if args.frame is None else args.frame,
    )
    if args.frame is None:
        times, load, _extension, _channel = read_global_channels(args.data)
        candidates = usable_frames(times, load)
        dropped = load_drop_frames(load, candidates, args.drop_threshold)
        dropped, _by_hand = merge_hand_picks(
            dropped, candidates, args.exclude_frames,
        )
        kept = resolve_range(
            np.setdiff1d(candidates, dropped).astype(np.intp),
            args.frame_min, args.frame_max,
        )
        valid = range_validity(args.data, kept)
        mask_note = f"valid in all {kept.size} frames of {kept[0]}..{kept[-1]}"
    else:
        mask_note = f"valid at frame {args.frame}"

    y_min, y_max, windows, suggested = resolve_cuts(
        outer, holes, y, valid, args.y_min, args.y_max,
    )
    out = Path(
        args.out or f"scratch/cut_preview_y{y_min:g}_{y_max:g}.png"
    )
    plot_cut_preview(
        out, outer, holes, x, y, valid, y_min, y_max, mask_note=mask_note,
    )
    print(f"validity: {mask_note}")

    print(
        "usable cut windows (constant width, clear of holes): "
        + ", ".join(f"[{a:.4g}, {b:.4g}]" for a, b in windows)
    )
    print(
        f"suggested cuts {suggested[0]:g} and {suggested[1]:g}"
        f"   using y_min {y_min:g} and y_max {y_max:g}"
    )
    print(cut_report(x, y, valid, y_min, y_max, band=args.band))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
