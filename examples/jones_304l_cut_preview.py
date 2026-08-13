"""Preview candidate model cuts against the Jones et al. 304L DIC data.

The coupon outline covers the whole specimen, ends included, while the
model covers a window of it bounded by two cuts in y. The cuts belong
inside the region where DIC data exists, so that the Dirichlet edges have
data to inform them. This plots the outline, the measured point cloud,
and a candidate pair of cuts together, and reports how much data sits
outside each cut and how far the data reaches across each cut edge.

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
        *, frame: int = 0,
) -> None:
    """Write the preview plot to ``out_path``."""
    fig, ax = plt.subplots(figsize=(7, 10))

    ax.scatter(x[valid], y[valid], s=0.4, c="#7fb3d5", linewidths=0,
               rasterized=True, label=f"DIC valid ({int(valid.sum())})")
    if (~valid).any():
        ax.scatter(x[~valid], y[~valid], s=6.0, c="#d62728", linewidths=0,
                   label=f"DIC invalid ({int((~valid).sum())})")

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
    ax.set_title(f"Candidate cuts, frame {frame}")
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
        "--frame", type=int, default=0,
        help="frame supplying the validity mask (default 0)",
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
    x, y, valid = read_dic_reference(args.data, args.frame)

    y_min, y_max, windows, suggested = resolve_cuts(
        outer, holes, y, valid, args.y_min, args.y_max,
    )
    out = Path(
        args.out or f"scratch/cut_preview_y{y_min:g}_{y_max:g}.png"
    )
    plot_cut_preview(
        out, outer, holes, x, y, valid, y_min, y_max, frame=args.frame,
    )

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
