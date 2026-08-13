"""Convert Jones et al. 304L DIC measurements into CMAD input files.

Reads the dataset's per-frame surface displacements, reconstructs them
onto the nodes of an FE mesh with GMLS, and writes the three files a
calibration run consumes: the nodal displacement history, the time
schedule it is sampled on, and the measured load.

The frame range comes from jones_304l_load_preview.py, which plots the
load record so the ends can be trimmed by eye. Within that range the
frames are selected here.

An undeformed reference is prepended as a constructed entry at t = 0
with zero displacement and zero load. The dataset zeroes time at the
extrapolated start of the test, where the specimen is unloaded by
construction, and its own frame 0 carries exactly zero displacement, so
that state is measured rather than assumed. Keeping it separate from the
frame range means trimming early frames never costs the reference.

The reconstruction is a function of the in-plane position, so on a 3D
mesh every node at a given ``(x, y)`` receives the same value whatever
its ``z``: the imaged surface, the back surface, and everything between.
Prescribing ``u_z`` from it on the cut faces therefore misses how
out-of-plane displacement varies through the thickness, while capturing
the specimen's rigid out-of-plane motion, which is measured and is
large. Holding ``u_z`` at zero there, or leaving those faces traction
free in ``z``, discards that motion, so all three components are the
intended boundary values.

Usage:
    python examples/jones_304l_dic_data.py \\
        --mesh examples/meshes/jones_304l_2d_y65_179_h2.msh \\
        --y-min 65 --y-max 179 \\
        --frame-min 11 --frame-max 510 --num-steps 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from jones_304l_geometry import (
    CENTER_MODES,
    centering_offset,
    clean_loop,
    read_nominal_geometry,
    trim_outline,
)
from jones_304l_load_preview import (
    LOAD_ROW_FY,
    read_global_channels,
    resolve_range,
    usable_frames,
)
from numpy.typing import NDArray

from cmad.io.mesh_io import read_mesh_file
from cmad.remap.gmls import build_gmls_operators

# Uncorrelated points carry a negative confidence, stored as -1, with NaN
# displacements at the same indices.
MIN_VALID_SIGMA = 0.0

# The dataset records force in kN; a model in mm and MPa wants newtons.
KN_TO_N = 1000.0

DISPLACEMENT_KEYS = ("U", "V", "W")


def select_frames(
        candidates: NDArray[np.intp],
        num_steps: int,
        *,
        by: NDArray[np.float64] | None = None,
) -> NDArray[np.intp]:
    """Pick ``num_steps`` frames from ``candidates``.

    With ``by`` the picks are spread evenly over that quantity, which
    matters because the record is far from uniform in index: the elastic
    ramp occupies its first tenth. Without it they are spread evenly over
    the candidate order. Both ends of the range are always included.
    """
    if num_steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}")
    if candidates.size < num_steps:
        raise ValueError(
            f"asked for {num_steps} frames but the range holds only "
            f"{candidates.size}"
        )
    if by is None:
        picks = np.round(np.linspace(0, candidates.size - 1, num_steps))
        return candidates[np.unique(picks.astype(np.intp))]
    values = by[candidates]
    targets = np.linspace(values[0], values[-1], num_steps)
    nearest = np.abs(values[:, None] - targets[None, :]).argmin(axis=0)
    return candidates[np.unique(nearest)]


def read_reference_coords(
        path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """In-plane reference coordinates of the measurement points."""
    with h5py.File(Path(path), "r") as handle:
        x = np.asarray(handle["X0"][0, :], dtype=np.float64)
        y = np.asarray(handle["Y0"][0, :], dtype=np.float64)
    return x, y


def cloud_offset_for_mesh(
        geometry_file: str | Path,
        mesh_nodes: NDArray[np.float64],
        y_min: float,
        y_max: float,
        center: str,
) -> NDArray[np.float64]:
    """Return the offset bringing the cloud onto the mesh, and check it.

    The offset follows from the cuts, so cuts other than the ones the
    mesh was built with would displace the cloud by an amount too small
    to notice and too large to tolerate. Comparing the centered outline's
    bounding box against the mesh's catches that at the outset.
    """
    outer, holes = read_nominal_geometry(geometry_file)
    outer = clean_loop(outer)
    holes = [clean_loop(hole) for hole in holes]
    trimmed = trim_outline(outer, y_min, y_max)
    kept = [
        hole for hole in holes
        if y_min <= hole[:, 1].min() and hole[:, 1].max() <= y_max
    ]
    offset = centering_offset(trimmed, kept, center)

    centered = trimmed - offset
    want_lo, want_hi = centered.min(axis=0), centered.max(axis=0)
    got_lo = mesh_nodes[:, :2].min(axis=0)
    got_hi = mesh_nodes[:, :2].max(axis=0)
    tol = 1.0e-3 * float(np.max(want_hi - want_lo))
    if not (
        np.allclose(want_lo, got_lo, atol=tol)
        and np.allclose(want_hi, got_hi, atol=tol)
    ):
        raise ValueError(
            f"cuts y_min={y_min:g}, y_max={y_max:g} give an outline spanning "
            f"x [{want_lo[0]:.4g}, {want_hi[0]:.4g}], "
            f"y [{want_lo[1]:.4g}, {want_hi[1]:.4g}], but the mesh spans "
            f"x [{got_lo[0]:.4g}, {got_hi[0]:.4g}], "
            f"y [{got_lo[1]:.4g}, {got_hi[1]:.4g}]; the cuts must be the ones "
            f"the mesh was built with"
        )
    return offset


def remap_history(
        data_file: str | Path,
        frames: NDArray[np.intp],
        source: NDArray[np.float64],
        targets: NDArray[np.float64],
        num_components: int,
        *,
        per_frame: bool = True,
        poly_order: int = 2,
        support_multiplier: float = 1.6,
) -> tuple[NDArray[np.float64], int]:
    """Reconstruct the measured displacements onto ``targets``.

    With ``per_frame`` each frame is fitted from exactly the points valid
    in that frame; otherwise one operator serves every frame, built from
    the points valid throughout. Returns the ``(num_frames, num_targets,
    num_components)`` history and the fewest source points any fit used.

    The source is every valid measurement point, untrimmed. GMLS support
    is local, so points beyond the modeled region never enter a target's
    neighborhood, while trimming to it would leave targets on the cut
    edges supported from one side only.
    """
    keys = DISPLACEMENT_KEYS[:num_components]
    history = np.zeros(
        (frames.size, targets.shape[0], num_components), dtype=np.float64,
    )
    fewest = source.shape[0]
    shared = None

    with h5py.File(Path(data_file), "r") as handle:
        if not per_frame:
            valid = np.ones(source.shape[0], dtype=bool)
            for frame in frames:
                valid &= handle["sigma"][int(frame), :] >= MIN_VALID_SIGMA
            fewest = int(valid.sum())
            shared = build_gmls_operators(
                source[valid], targets, poly_order=poly_order,
                support_multiplier=support_multiplier,
            )
        for i, frame in enumerate(frames):
            if per_frame:
                valid = handle["sigma"][int(frame), :] >= MIN_VALID_SIGMA
                fewest = min(fewest, int(valid.sum()))
                ops = build_gmls_operators(
                    source[valid], targets, poly_order=poly_order,
                    support_multiplier=support_multiplier,
                )
            else:
                assert shared is not None
                ops = shared
            for c, key in enumerate(keys):
                measured = np.asarray(
                    handle[key][int(frame), :], dtype=np.float64,
                )
                history[i, :, c] = ops.value @ measured[valid]
    return history, fewest


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
        help="first frame to draw from (default: the first usable one)",
    )
    parser.add_argument(
        "--frame-max", type=int, default=None,
        help="last frame to draw from (default: the last usable one)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=20,
        help="frames to select; the reference adds one (default 20)",
    )
    parser.add_argument(
        "--select", default="index", choices=("index", "force", "extension"),
        help="spread the selected frames evenly over this (default index)",
    )
    parser.add_argument(
        "--filter", default="per-frame", choices=("per-frame", "union"),
        help="fit each frame from its own valid points, or from the points "
             "valid in all of them (default per-frame)",
    )
    parser.add_argument("--poly-order", type=int, default=2)
    parser.add_argument("--support-multiplier", type=float, default=1.6)
    parser.add_argument(
        "--load-component", type=int, default=LOAD_ROW_FY,
        help=f"force6DOF row to use as the load (default {LOAD_ROW_FY}, Fy)",
    )
    parser.add_argument(
        "--force-scale", type=float, default=KN_TO_N,
        help="multiplies the load; the data is kN (default 1000, to newtons)",
    )
    parser.add_argument(
        "--out-prefix", default=None,
        help="output stem (default data/jones_304l/<mesh stem>)",
    )
    args = parser.parse_args()

    mesh = read_mesh_file(args.mesh)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    num_components = nodes.shape[1]
    offset = cloud_offset_for_mesh(
        args.geometry, nodes, args.y_min, args.y_max, args.center,
    )

    times, load, extension, channel = read_global_channels(
        args.data, args.load_component,
    )
    in_range = resolve_range(
        usable_frames(times, load), args.frame_min, args.frame_max,
    )
    spread = {"index": None, "force": load, "extension": extension}
    frames = select_frames(in_range, args.num_steps, by=spread[args.select])

    x, y = read_reference_coords(args.data)
    source = np.column_stack([x - offset[0], y - offset[1]])
    deformed, fewest = remap_history(
        args.data, frames, source, nodes[:, :2], num_components,
        per_frame=args.filter == "per-frame",
        poly_order=args.poly_order,
        support_multiplier=args.support_multiplier,
    )

    # The reference is constructed, not measured: t = 0 is the test start
    # the dataset zeroes to, where the specimen is unloaded and undeformed.
    reference = np.zeros_like(deformed[:1])
    history = np.concatenate([reference, deformed], axis=0)
    schedule = np.concatenate([[0.0], times[frames]])
    series = np.concatenate([[0.0], load[frames]]) * args.force_scale

    stem = Path(args.out_prefix or f"data/jones_304l/{Path(args.mesh).stem}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{stem}_u.npz", u=history)
    np.savetxt(f"{stem}_times.txt", schedule)
    np.save(f"{stem}_load.npy", series)

    raw_bytes = history.nbytes
    stored = Path(f"{stem}_u.npz").stat().st_size
    print(f"load channel: {channel}, scaled by {args.force_scale:g}")
    print(
        f"drawing from frames {in_range[0]} to {in_range[-1]} "
        f"({in_range.size}); selected {frames.size} plus the reference"
    )
    print(
        f"source {source.shape[0]} points, fewest valid in a fit {fewest}; "
        f"targets {nodes.shape[0]} nodes, {num_components} components"
    )
    print("\n  frame      time      load    extension     |u|max")
    for i, frame in enumerate(frames):
        print(
            f"  {frame:5d} {times[frame]:9.1f} {load[frame]:9.4f} "
            f"{extension[frame]:11.4f} {np.abs(history[i + 1]).max():10.4f}"
        )
    print(
        f"\nwrote {stem}_u.npz {history.shape}, "
        f"{stored / 1e6:.2f} MB stored against {raw_bytes / 1e6:.2f} MB raw"
    )
    print(f"      {stem}_times.txt, {stem}_load.npy")


if __name__ == "__main__":
    main()
