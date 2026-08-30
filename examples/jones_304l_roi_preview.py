"""Preview the region of interest and choose its free edge band.

The displacement match integrates over the mesh entities the measurement
covers in every frame of the range. DIC drops out near a free edge,
where the correlation window runs off the specimen, while the cuts slice
through a continuous measured field and carry data right up to them; so
the region is everything farther than a band from the free boundary.
This draws the kept elements (2D) or imaged face sides (3D) for one or
more candidate bands over the measurement cloud, so the band can be
chosen by eye and passed to jones_304l_dic_data.py as ``--roi-band``.

Without ``--roi-band`` the panel shows the suggested band: the largest
distance from the free boundary to a point valid in every frame, plus
``--roi-margin``. The frame range and the force artifact exclusion are
the load preview's.

Usage:
    python examples/jones_304l_roi_preview.py \\
        --mesh examples/meshes/jones_304l_2d_y65_179_h3.msh \\
        --y-min 65 --y-max 179 --frame-min 11 --frame-max 510
    ... --roi-band 1 1.5 2 --zoom -14 -20 14 20
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jones_304l_dic_data import (
    cloud_offset_for_mesh,
    frame_validity,
    free_boundary_segments,
    measured_facets,
    read_reference_coords,
    region_of_interest,
)
from jones_304l_geometry import CENTER_MODES
from jones_304l_load_preview import (
    DROP_THRESHOLD,
    load_drop_frames,
    merge_hand_picks,
    read_global_channels,
    resolve_range,
    usable_frames,
)
from matplotlib.tri import Triangulation
from numpy.typing import NDArray

from cmad.fem.mesh import coordinate_side_sets
from cmad.io.mesh_io import read_mesh_file


def plot_regions(
        out_path: Path,
        nodes: NDArray[np.float64],
        facets: NDArray[np.intp],
        free: NDArray[np.float64],
        cuts: tuple[float, float],
        source: NDArray[np.float64],
        always: NDArray[np.bool_],
        panels: list[tuple[float, NDArray[np.bool_], float, float]],
        kind: str,
        zoom: list[float] | None,
) -> None:
    """One panel per ``(band, kept, max_gap, worst_covered)`` entry."""
    tri = Triangulation(nodes[:, 0], nodes[:, 1], facets)
    zoomed = zoom is not None
    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(5.0 * n, 6.0 if zoomed else 8.0), squeeze=False,
    )
    for ax, (band, kept, max_gap, worst) in zip(axes[0], panels, strict=True):
        ax.tripcolor(tri, facecolors=kept.astype(float), cmap="Blues",
                     vmin=0.0, vmax=1.7, edgecolors="none")
        ax.triplot(tri, color="0.7", lw=0.2)
        ax.plot(source[always, 0], source[always, 1], ".", color="0.35",
                ms=2.4 if zoomed else 0.35, alpha=0.75 if zoomed else 0.35)
        for seg in free:
            ax.plot(seg[:, 0], seg[:, 1], color="tab:red", lw=1.6)
        for cut in cuts:
            ax.axhline(cut, color="tab:green", lw=1.6)
        ax.set_title(
            f"band {band:.3f} mm: {int(kept.sum())}/{kept.size} {kind} kept\n"
            f"largest gap {max_gap:.3f}, worst kept node {worst:.3f} mm",
            fontsize=9,
        )
        ax.set_aspect("equal")
        if zoom is not None:
            ax.set_xlim(zoom[0], zoom[2])
            ax.set_ylim(zoom[1], zoom[3])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "blue kept, white excluded, red free boundary, green cuts,\n"
        "grey points valid in every frame",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
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
    parser.add_argument("--mesh", required=True, help="mesh to remap onto")
    parser.add_argument(
        "--y-min", type=float, required=True,
        help="lower cut the mesh was built with",
    )
    parser.add_argument(
        "--y-max", type=float, required=True,
        help="upper cut the mesh was built with",
    )
    parser.add_argument(
        "--center", default="xy", choices=CENTER_MODES,
        help="centering the mesh was built with (default xy)",
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
        "--roi-band", type=float, nargs="+", default=None,
        help="candidate free edge bands in mm, one panel each (default: "
             "the suggested band)",
    )
    parser.add_argument(
        "--roi-margin", type=float, default=0.15,
        help="added to the largest measured gap for the suggested band "
             "(default 0.15 mm)",
    )
    parser.add_argument(
        "--zoom", type=float, nargs=4, default=None,
        metavar=("X0", "Y0", "X1", "Y1"), help="view window in mesh coords",
    )
    parser.add_argument(
        "--out", default=None,
        help="output image (default scratch/roi_preview_<mesh stem>.png)",
    )
    args = parser.parse_args()

    mesh = read_mesh_file(args.mesh)
    mesh = replace(
        mesh, side_sets={**mesh.side_sets, **coordinate_side_sets(mesh)},
    )
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    offset = cloud_offset_for_mesh(
        args.geometry, nodes, args.y_min, args.y_max, args.center,
    )
    cuts = (args.y_min - offset[1], args.y_max - offset[1])

    times, load, _extension, _channel = read_global_channels(args.data)
    candidates = usable_frames(times, load)
    dropped = load_drop_frames(load, candidates, args.drop_threshold)
    dropped, _by_hand = merge_hand_picks(
        dropped, candidates, args.exclude_frames,
    )
    in_range = resolve_range(
        np.setdiff1d(candidates, dropped).astype(np.intp),
        args.frame_min, args.frame_max,
    )

    x, y = read_reference_coords(args.data)
    source = np.column_stack([x - offset[0], y - offset[1]])
    valid = frame_validity(args.data, in_range)
    always = valid.all(axis=0)
    sometimes = valid.any(axis=0) & ~always
    print(
        f"{int(always.sum())} points valid in all {in_range.size} frames of "
        f"{in_range[0]}..{in_range[-1]}, {int(sometimes.sum())} in some"
    )

    _entities, facets = measured_facets(mesh)
    if facets.shape[1] != 3:
        raise SystemExit("triangular elements or faces expected")
    kind = "elements" if nodes.shape[1] == 2 else "sides"
    free = free_boundary_segments(mesh, facets, *cuts)
    panels = []
    for requested in [None] if args.roi_band is None else args.roi_band:
        kept, band, max_gap, worst = region_of_interest(
            mesh, facets, source, always, *cuts,
            band=requested, margin=args.roi_margin,
        )
        panels.append((band, kept, max_gap, worst))
        print(
            f"band {band:.3f} mm: kept {int(kept.sum())} of {kept.size} "
            f"{kind}; largest gap {max_gap:.3f}; worst kept node "
            f"{worst:.3f} mm from a measurement"
        )

    out = Path(args.out or f"scratch/roi_preview_{Path(args.mesh).stem}.png")
    plot_regions(
        out, nodes, facets, free, cuts, source, always, panels, kind,
        args.zoom,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
