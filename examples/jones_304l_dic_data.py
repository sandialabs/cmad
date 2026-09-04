"""Convert Jones et al. 304L DIC measurements into CMAD calibration data.

Reconstructs the measured surface displacements onto an FE mesh with
GMLS at every usable frame of the chosen range and writes them, with the
measured load, as a calibration data archive
(:mod:`cmad.io.calibration_data`), plus the time schedule of
``--num-steps`` selected frames and the match times.

The archive also carries the region of interest the displacement match
integrates over: the elements (2D) or imaged face sides (3D) whose nodes
all sit farther from the free boundary than ``--roi-band``, chosen with
jones_304l_roi_preview.py. Only the nodes of that region and of the
Dirichlet sidesets (``--bc-sidesets``) are remapped.

The frame range comes from jones_304l_load_preview.py, which plots the
load record so the ends can be trimmed by eye. Within that range the
frames are selected here, after excluding the record's spurious force
artifacts the same way the preview does; see ``load_drop_frames`` there.

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
        --geometry data/jones_304l/O5-NominalGeometry.mat \\
        --data data/jones_304l/O5-4-Data.mat \\
        --mesh examples/meshes/jones_304l_o5_2d_y40_132_h3.msh \\
        --y-min 40 --y-max 132 \\
        --frame-min 3 --frame-max 700 --num-steps 50 --roi-band 1.7
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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
    DROP_THRESHOLD,
    LOAD_ROW_FY,
    drop_summary,
    load_drop_frames,
    merge_hand_picks,
    read_global_channels,
    resolve_range,
    usable_frames,
)
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from cmad.fem.mesh import (
    _LOCAL_SIDES_PER_ELEMENT,
    Mesh,
    coordinate_side_sets,
    side_set_nodes,
)
from cmad.io.calibration_data import CalibrationData
from cmad.io.mesh_io import read_mesh_file
from cmad.remap.gmls import build_gmls_operators

# Uncorrelated points carry a negative confidence, stored as -1, with NaN
# displacements at the same indices.
MIN_VALID_SIGMA = 0.0

# The cameras face the front of the specimen, which the extrusion puts at
# the top of the mesh. The reconstruction is a function of the in-plane
# position alone, so the two faces differ only in how the model itself
# varies through the thickness.
OBSERVED_SIDESET = "zmax_sides"

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


def measured_facets(mesh: Mesh) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """The mesh entities the measurement covers, and their node ids.

    A 2D mesh is the measured surface itself, so the entities are its
    elements. A 3D mesh is imaged on one face, so they are that face's
    ``(elem_id, local_side_id)`` pairs from ``mesh.side_sets``.
    """
    if mesh.nodes.shape[1] == 2:
        elements = np.arange(mesh.connectivity.shape[0], dtype=np.intp)
        return elements, mesh.connectivity
    if OBSERVED_SIDESET not in mesh.side_sets:
        raise KeyError(
            f"observed sideset {OBSERVED_SIDESET!r} not in the mesh (known: "
            f"{sorted(mesh.side_sets)})"
        )
    sides = mesh.side_sets[OBSERVED_SIDESET]
    local_sides = _LOCAL_SIDES_PER_ELEMENT[mesh.element_family]
    facets = mesh.connectivity[sides[:, 0][:, None], local_sides[sides[:, 1]]]
    return sides, facets


def free_boundary_segments(
        mesh: Mesh, facets: NDArray[np.intp], cut_lo: float, cut_hi: float,
        rel_tol: float = 1.0e-6,
) -> NDArray[np.float64]:
    """In-plane segments bounding ``facets`` that are not on a cut.

    An edge is on the boundary when exactly one facet owns it. The cuts
    slice a subdomain out of a continuous measured field, so the data
    runs right up to them; the rest is free surface, where the
    correlation window runs off the specimen and the measurement stops
    short. An edge counts as on a cut when both of its vertices are within
    ``rel_tol`` of the cut, scaled by the mesh extent.
    """
    # A facet's vertices run around it, so consecutive pairs are its edges.
    edges = np.stack([facets, np.roll(facets, -1, axis=1)], axis=-1)
    keys = np.sort(edges.reshape(-1, 2), axis=1)
    _uniq, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)
    coords = mesh.nodes[keys[counts[inverse] == 1]][..., :2]
    y = coords[..., 1]
    tol = rel_tol * float(np.max(mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0)))
    on_cut = np.all(
        (np.abs(y - cut_lo) < tol) | (np.abs(y - cut_hi) < tol), axis=1)
    return coords[~on_cut]


def distance_to_segments(
        points: NDArray[np.float64], segments: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Distance from each point to the nearest of ``(n, 2, 2)`` segments."""
    start, end = segments[:, 0, :], segments[:, 1, :]
    span = end - start
    length_sq = (span ** 2).sum(-1)
    length_sq = np.where(length_sq > 0.0, length_sq, 1.0)
    t = np.clip(
        ((points[:, None, :] - start) * span).sum(-1) / length_sq, 0.0, 1.0)
    closest = start + t[..., None] * span
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def region_of_interest(
        mesh: Mesh, facets: NDArray[np.intp], source: NDArray[np.float64],
        valid: NDArray[np.bool_], cut_lo: float, cut_hi: float, *,
        band: float | None = None, margin: float = 0.15,
        samples_per_side: int = 20,
) -> tuple[NDArray[np.bool_], float, float, float]:
    """Which facets are far enough from the free boundary to carry measurements.

    A facet is kept when every one of its nodes is farther than ``band``
    from the free boundary. The band defaults to the largest distance
    from the free boundary to the nearest valid measurement, plus
    ``margin``. The gap is measured at ``samples_per_side`` points per
    free boundary segment, so the margin pads for the true maximum falling
    between samples, and keeps the nodes a little inside the cloud edge,
    where a fit has measurements on one side only.

    Returns the keep mask, the band, the largest gap, and the worst
    distance from a kept node to a measurement. A value there beyond the
    point spacing means measurements are missing somewhere the free
    boundary does not predict, which no band will find.
    """
    free = free_boundary_segments(mesh, facets, cut_lo, cut_hi)
    t = np.linspace(0.0, 1.0, samples_per_side + 1)[:-1, None]
    samples = np.vstack([a + t * (b - a) for a, b in free])
    tree = cKDTree(source[valid])
    max_gap = float(tree.query(samples, k=1)[0].max())
    if band is None:
        band = max_gap + margin
    node_distance = distance_to_segments(mesh.nodes[:, :2], free)
    kept = node_distance[facets].min(axis=1) > band
    kept_nodes = np.unique(facets[kept])
    worst_covered = float(tree.query(mesh.nodes[kept_nodes, :2], k=1)[0].max())
    return kept, float(band), max_gap, worst_covered


def read_reference_coords(
        path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """In-plane reference coordinates of the measurement points."""
    with h5py.File(Path(path), "r") as handle:
        x = np.asarray(handle["X0"][0, :], dtype=np.float64)
        y = np.asarray(handle["Y0"][0, :], dtype=np.float64)
    return x, y


def read_frame_rows(
        handle: h5py.File, key: str, frames: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Rows ``frames`` of the dataset ``key``, read in point axis blocks.

    The files chunk every frame of a few points together, so a frame by
    frame read decompresses the whole dataset once per frame; one pass
    along the point axis reads each chunk once.
    """
    # Bounds memory only: any block of a few thousand points reads the
    # chunks exactly once.
    point_block = 8192
    dataset = handle[key]
    _n_frames, n_points = dataset.shape
    rows = np.empty((frames.size, n_points), dtype=np.float64)
    for start in range(0, n_points, point_block):
        stop = min(start + point_block, n_points)
        piece = np.asarray(dataset[:, start:stop], dtype=np.float64)
        rows[:, start:stop] = piece[frames]
    return rows


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


# Frames read per pass. The chunks span every frame, so a block of rows
# costs the same read as one.
FRAME_BLOCK = 128


def frame_validity(
        data_file: str | Path, frames: NDArray[np.intp],
) -> NDArray[np.bool_]:
    """Which measurements are valid at ``frames``.

    Shaped ``(num_frames, num_points)``, read in one pass along the point
    axis as :func:`read_frame_rows` does, and kept as booleans so every
    frame of the record fits in memory.
    """
    point_block = 8192
    with h5py.File(Path(data_file), "r") as handle:
        dataset = handle["sigma"]
        _n_frames, n_points = dataset.shape
        valid = np.empty((frames.size, n_points), dtype=bool)
        for start in range(0, n_points, point_block):
            stop = min(start + point_block, n_points)
            piece = np.asarray(dataset[:, start:stop], dtype=np.float64)
            valid[:, start:stop] = piece[frames] >= MIN_VALID_SIGMA
    return valid


def remap_history(
        data_file: str | Path,
        frames: NDArray[np.intp],
        valid: NDArray[np.bool_],
        source: NDArray[np.float64],
        targets: NDArray[np.float64],
        num_components: int,
        *,
        per_frame: bool = True,
        poly_order: int = 2,
        support_multiplier: float = 1.6,
        frame_block: int = FRAME_BLOCK,
) -> tuple[NDArray[np.float64], int]:
    """Reconstruct the measured displacements at ``frames`` onto ``targets``.

    ``valid`` is the validity from :func:`frame_validity`. With
    ``per_frame`` each frame is fitted from exactly the points valid in
    that frame; otherwise one operator serves every frame, built from the
    points valid throughout. The displacements are read ``frame_block``
    frames at a time. Returns the ``(num_frames, num_targets,
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
    throughout = None if per_frame else valid.all(axis=0)
    shared = None
    if throughout is None:
        fewest = int(valid.sum(axis=1).min())
    else:
        fewest = int(throughout.sum())
        shared = build_gmls_operators(
            source[throughout], targets, poly_order=poly_order,
            support_multiplier=support_multiplier,
        )
    with h5py.File(Path(data_file), "r") as handle:
        for start in range(0, frames.size, frame_block):
            block = frames[start:start + frame_block]
            measured = [read_frame_rows(handle, key, block) for key in keys]
            for j in range(block.size):
                i = start + j
                keep = valid[i] if throughout is None else throughout
                ops = shared if shared is not None else build_gmls_operators(
                    source[keep], targets, poly_order=poly_order,
                    support_multiplier=support_multiplier,
                )
                for c in range(num_components):
                    history[i, :, c] = ops.value @ measured[c][j][keep]
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
    parser.add_argument(
        "--roi-band", type=float, required=True,
        help="free edge band in mm, chosen with jones_304l_roi_preview.py",
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
        "--drop-threshold", type=float, default=DROP_THRESHOLD,
        help=f"exclude frames whose load deviates from a rolling median by "
             f"more than this, in kN before --force-scale; 0 keeps every "
             f"frame (default {DROP_THRESHOLD:g})",
    )
    parser.add_argument(
        "--exclude-frames", type=int, nargs="+", default=None,
        help="frames to exclude by hand, in addition to the detected drops",
    )
    parser.add_argument(
        "--out-prefix", default=None,
        help="output stem (default data/jones_304l/<mesh stem>)",
    )
    parser.add_argument(
        "--bc-sidesets", nargs="+", default=["ymin_sides", "ymax_sides"],
        help="sidesets the Dirichlet conditions prescribe "
             "(default ymin_sides ymax_sides)",
    )
    parser.add_argument(
        "--frame-block", type=int, default=FRAME_BLOCK,
        help=f"frames read per pass (default {FRAME_BLOCK})",
    )
    args = parser.parse_args()

    mesh = read_mesh_file(args.mesh)
    mesh = replace(
        mesh, side_sets={**mesh.side_sets, **coordinate_side_sets(mesh)},
    )
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    num_components = nodes.shape[1]
    offset = cloud_offset_for_mesh(
        args.geometry, nodes, args.y_min, args.y_max, args.center,
    )

    times, load, extension, channel = read_global_channels(
        args.data, args.load_component,
    )
    candidates = usable_frames(times, load)
    dropped = load_drop_frames(load, candidates, args.drop_threshold)
    dropped, by_hand = merge_hand_picks(
        dropped, candidates, args.exclude_frames,
    )
    in_range = resolve_range(
        np.setdiff1d(candidates, dropped).astype(np.intp),
        args.frame_min, args.frame_max,
    )
    spread = {"index": None, "force": load, "extension": extension}
    frames = select_frames(in_range, args.num_steps, by=spread[args.select])

    x, y = read_reference_coords(args.data)
    source = np.column_stack([x - offset[0], y - offset[1]])
    valid = frame_validity(args.data, in_range)

    entities, facets = measured_facets(mesh)
    kept, band, max_gap, worst_covered = region_of_interest(
        mesh, facets, source, valid.all(axis=0),
        args.y_min - offset[1], args.y_max - offset[1], band=args.roi_band,
    )
    sidesets = {name: side_set_nodes(mesh, name) for name in args.bc_sidesets}
    node_ids = np.union1d(
        np.unique(facets[kept]), np.concatenate(list(sidesets.values())),
    ).astype(np.intp)

    deformed, fewest = remap_history(
        args.data, in_range, valid, source, nodes[node_ids, :2],
        num_components,
        per_frame=args.filter == "per-frame",
        poly_order=args.poly_order,
        support_multiplier=args.support_multiplier,
        frame_block=args.frame_block,
    )

    # The reference is constructed, not measured: t = 0 is the test start
    # the dataset zeroes to, where the specimen is unloaded and undeformed.
    entity_kind = "elements" if num_components == 2 else "sides"
    data = CalibrationData(
        times=np.concatenate([[0.0], times[in_range]]),
        frame_ids=np.concatenate([[-1], in_range]).astype(np.intp),
        load=np.concatenate([[0.0], load[in_range]]) * args.force_scale,
        node_ids=node_ids,
        sidesets=sidesets,
        values=np.concatenate([np.zeros_like(deformed[:1]), deformed]),
        mesh_file=str(args.mesh),
        mesh_num_nodes=int(nodes.shape[0]),
        roi={
            entity_kind: entities[kept],
            "band": np.float64(band), "max_gap": np.float64(max_gap),
        },
    )
    schedule = np.concatenate([[0.0], times[frames]])

    stem = Path(args.out_prefix or f"data/jones_304l/{Path(args.mesh).stem}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    archive = Path(f"{stem}_calibration_data.npz")
    data.write(archive)
    np.savetxt(f"{stem}_solve_times.txt", schedule)
    np.savetxt(f"{stem}_match_times.txt", schedule)

    print(f"load channel: {channel}, scaled by {args.force_scale:g}")
    if dropped.size:
        print(drop_summary(dropped))
    if by_hand.size:
        print(f"excluded by hand: {', '.join(str(f) for f in by_hand)}")
    print(
        f"frames {in_range[0]} to {in_range[-1]} ({in_range.size}) plus the "
        f"reference; schedule of {frames.size} selected frames"
    )
    print(
        f"source {source.shape[0]} points, fewest valid in a fit {fewest}; "
        f"targets {node_ids.size} of {nodes.shape[0]} nodes, "
        f"{num_components} components"
    )
    print("\n  frame      time      load    extension     |u|max")
    for row, frame in zip(np.searchsorted(in_range, frames), frames,
                          strict=True):
        print(
            f"  {frame:5d} {times[frame]:9.1f} {load[frame]:9.4f} "
            f"{extension[frame]:11.4f} {np.abs(deformed[row]).max():10.4f}"
        )
    print(
        f"\nwrote {archive} {data.values.shape}, "
        f"{archive.stat().st_size / 1e6:.2f} MB stored against "
        f"{data.values.nbytes / 1e6:.2f} MB raw"
    )
    print(f"      {stem}_solve_times.txt, {stem}_match_times.txt")
    print(
        f"      region of interest {int(kept.sum())} of {kept.size} "
        f"{entity_kind}; free edge band {band:.3f} mm from a largest gap of "
        f"{max_gap:.3f}; worst kept node is {worst_covered:.3f} mm from a "
        f"measurement"
    )


if __name__ == "__main__":
    main()
