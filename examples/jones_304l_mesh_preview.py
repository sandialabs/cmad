"""Preview how far a mesh strays from the hole outlines as sizing changes.

Where a hole turns more tightly than the element size, a mesh sized
uniformly at ``h`` cuts across the turn rather than following it.
Refining from curvature fixes that locally, and this reports what each
setting buys and costs so a value can be chosen before the mesh is
generated. The tightest turn is measured from the geometry file, so the
report describes whichever specimen is being meshed rather than assuming
the holes of any one of them.

For every setting it prints the element count and the boundary error,
which is the largest distance from the middle of a boundary element to
the outline it is meant to follow. That distance is measured against the
curve gmsh meshed rather than against the sampled outline, whose own
1.27 mm chords would otherwise swamp a well resolved corner.

The face is previewed in two dimensions, where the faceting lives; an
extruded mesh inherits the same in-plane boundary.

Usage:
    python examples/jones_304l_mesh_preview.py --y-min 65 --y-max 179
    python examples/jones_304l_mesh_preview.py --y-min 65 --y-max 179 \\
        --h 3 --curvature-elements 0 12 20 28
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from jones_304l_geometry import (
    Loop,
    centering_offset,
    clean_loop,
    read_nominal_geometry,
    trim_outline,
)
from jones_304l_mesh import CURVATURE_ELEMENTS, build_specimen_mesh
from numpy.typing import NDArray

Triangulation = tuple[NDArray[np.float64], NDArray[np.intp]]


def polyline_distance(
        points: NDArray[np.float64], loop: Loop,
) -> NDArray[np.float64]:
    """Distance from each of ``points`` to the closed polyline ``loop``."""
    start, end = loop, np.roll(loop, -1, axis=0)
    span = end - start
    t = np.clip(
        ((points[:, None, :] - start) * span).sum(-1) / (span ** 2).sum(-1),
        0.0, 1.0,
    )
    closest = start + t[..., None] * span
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def turn_radius(loop: Loop) -> NDArray[np.float64]:
    """Radius of the circle through each loop point and its two neighbors.

    Infinite where three consecutive points are collinear, which is what
    the straight runs of an outline give.
    """
    before, here, after = np.roll(loop, 1, axis=0), loop, np.roll(loop, -1, axis=0)
    sides = [
        np.linalg.norm(here - before, axis=1),
        np.linalg.norm(after - here, axis=1),
        np.linalg.norm(before - after, axis=1),
    ]
    twice_area = np.abs(
        (here[:, 0] - before[:, 0]) * (after[:, 1] - before[:, 1])
        - (here[:, 1] - before[:, 1]) * (after[:, 0] - before[:, 0])
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return sides[0] * sides[1] * sides[2] / (2.0 * twice_area)


def curves_by_loop(loops: list[Loop], samples: int = 24) -> list[list[int]]:
    """Group the model's curves by which outline each one lies on.

    A curve is assigned to the loop it sits closest to, taking the worst
    of a few points along it. The loops are tens of mm apart, so no
    tolerance enters and the grouping does not depend on how gmsh
    happened to tag or order the curves.
    """
    grouped: list[list[int]] = [[] for _ in loops]
    for _dim, tag in gmsh.model.getEntities(1):
        lo, hi = gmsh.model.getParametrizationBounds(1, tag)
        t = np.linspace(lo[0], hi[0], samples)
        xy = np.asarray(gmsh.model.getValue(1, tag, t)).reshape(-1, 3)[:, :2]
        gaps = [float(polyline_distance(xy, loop).max()) for loop in loops]
        grouped[int(np.argmin(gaps))].append(tag)
    return grouped


def boundary_error(tags: list[int], samples: int = 2000) -> NDArray[np.float64]:
    """Distance from each boundary element's midpoint to its own curve.

    The midpoint of a straight element spanning a turn falls inside the
    curve by roughly ``length ** 2 / (8 * radius)``, so this is the
    faceting, and it goes to zero as the curve is resolved.
    """
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    index = {int(tag): i for i, tag in enumerate(node_tags)}
    xyz = np.asarray(coords).reshape(-1, 3)

    errors: list[NDArray[np.float64]] = []
    for tag in tags:
        types, _elements, connectivity = gmsh.model.mesh.getElements(1, tag)
        if not types:
            continue
        ends = np.asarray(connectivity[0], dtype=np.int64).reshape(-1, 2)
        first = xyz[[index[int(t)] for t in ends[:, 0]]]
        second = xyz[[index[int(t)] for t in ends[:, 1]]]

        lo, hi = gmsh.model.getParametrizationBounds(1, tag)
        t = np.linspace(lo[0], hi[0], samples)
        dense = np.asarray(gmsh.model.getValue(1, tag, t)).reshape(-1, 3)
        middle = 0.5 * (first + second)
        errors.append(
            np.linalg.norm(middle[:, None, :] - dense[None, :, :], axis=2).min(axis=1)
        )
    return np.concatenate(errors) if errors else np.zeros(0)


def mesh_once(
        path: Path, h: float, curvature_elements: float, loops: list[Loop],
        *, geometry_file: str | Path, y_min: float, y_max: float,
) -> tuple[Triangulation, int, list[NDArray[np.float64]]]:
    """Build one mesh and measure it, in a gmsh session this owns.

    Owning the session is what keeps the model alive after the build, so
    the curves can be sampled: ``build_specimen_mesh`` only finalizes the
    session it started itself.

    Returns the triangulation for plotting, the element count, and the
    boundary error on each loop.
    """
    gmsh.initialize()
    try:
        n_elements, _offset, _area = build_specimen_mesh(
            path, h, geometry_file=geometry_file, y_min=y_min, y_max=y_max,
            curvature_elements=curvature_elements,
        )
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {int(tag): i for i, tag in enumerate(node_tags)}
        xy = np.asarray(coords).reshape(-1, 3)[:, :2]
        _types, _elements, connectivity = gmsh.model.mesh.getElements(2)
        triangles = np.array(
            [index[int(tag)] for tag in np.asarray(connectivity[0])]
        ).reshape(-1, 3)
        errors = [
            boundary_error(tags) for tags in curves_by_loop(loops)
        ]
        return (xy, triangles), n_elements, errors
    finally:
        gmsh.finalize()


def plot_mesh_preview(
        out_path: Path, settings: list[float],
        meshes: list[Triangulation], holes: list[Loop], h: float,
) -> None:
    """Draw each mesh whole and zoomed on every hole, over the true outline."""
    n_columns = 1 + len(holes)
    fig, axes = plt.subplots(
        len(settings), n_columns, squeeze=False,
        figsize=(4.6 * n_columns, 5.6 * len(settings)),
        gridspec_kw={"width_ratios": [0.62] + [1.0] * len(holes)},
    )
    for row, (setting, (xy, triangles)) in enumerate(zip(settings, meshes, strict=True)):
        label = f"h = {h:g}" + (
            "" if setting == 0.0 else f", {setting:g} per turn"
        )
        for column in range(n_columns):
            ax = axes[row][column]
            ax.triplot(xy[:, 0], xy[:, 1], triangles,
                       lw=0.3 if column == 0 else 0.8, color="#333333")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_title(f"{label}\n{len(triangles)} elements", fontsize=10)
                continue

            hole = holes[column - 1]
            closed = np.vstack([hole, hole[:1]])
            ax.plot(closed[:, 0], closed[:, 1], "-", color="#d62728", lw=1.6,
                    label="nominal outline")
            center = 0.5 * (hole.min(axis=0) + hole.max(axis=0))
            reach = 0.62 * float(np.max(np.ptp(hole, axis=0)))
            ax.set_xlim(center[0] - reach, center[0] + reach)
            ax.set_ylim(center[1] - reach, center[1] + reach)
            ax.set_title(f"{label} - hole {column - 1}", fontsize=10)
            if row == 0 and column == n_columns - 1:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry", default="data/jones_304l/X-Specimen-NominalGeometry.mat",
        help="nominal geometry .mat",
    )
    parser.add_argument("--y-min", type=float, required=True, help="lower cut")
    parser.add_argument("--y-max", type=float, required=True, help="upper cut")
    parser.add_argument(
        "--h", type=float, default=3.0, help="element size away from curvature",
    )
    parser.add_argument(
        "--curvature-elements", type=float, nargs="+",
        default=[0.0, CURVATURE_ELEMENTS],
        help=f"settings to compare, 0 meaning uniform "
             f"(default 0 and {CURVATURE_ELEMENTS:g})",
    )
    parser.add_argument(
        "--out", default=None,
        help="output image (default scratch/mesh_preview_h{h}.png)",
    )
    args = parser.parse_args()

    outer, holes = read_nominal_geometry(args.geometry)
    outer = clean_loop(outer)
    holes = [clean_loop(hole) for hole in holes]
    trimmed = trim_outline(outer, args.y_min, args.y_max)
    kept = [
        hole for hole in holes
        if args.y_min <= hole[:, 1].min() and hole[:, 1].max() <= args.y_max
    ]

    # The generator centers the mesh, so compare against loops shifted the
    # same way. The outer loop is carried along because every curve is
    # assigned to its nearest loop, which needs somewhere other than a
    # hole for the outer curves to go.
    offset = centering_offset(trimmed, kept, "xy")
    loops = [trimmed - offset] + [hole - offset for hole in kept]

    settings = list(args.curvature_elements)
    meshes: list[Triangulation] = []
    counts: list[int] = []
    errors: list[list[NDArray[np.float64]]] = []
    for setting in settings:
        triangulation, n_elements, error = mesh_once(
            Path("scratch/mesh_preview.msh"), args.h, setting, loops,
            geometry_file=args.geometry, y_min=args.y_min, y_max=args.y_max,
        )
        meshes.append(triangulation)
        counts.append(n_elements)
        errors.append(error)

    for i, hole in enumerate(loops[1:]):
        radius = turn_radius(hole)
        tightest = float(radius[np.isfinite(radius)].min())
        extent = np.ptp(hole, axis=0)
        print(
            f"hole {i}: {extent[0]:.3g} by {extent[1]:.3g} mm, "
            f"tightest turn radius {tightest:.4g} mm"
        )

    tightest = min(
        float(turn_radius(hole)[np.isfinite(turn_radius(hole))].min())
        for hole in loops[1:]
    )
    print(
        f"\n{'per turn':>9}  {'elements':>9}  {'size at the tightest turn':>25}"
        f"  {'boundary error':>15}"
    )
    for setting, count, error in zip(settings, counts, errors, strict=True):
        hole_error = np.concatenate(error[1:]) if len(error) > 1 else np.zeros(1)
        size = "-" if setting == 0.0 else f"{2.0 * np.pi * tightest / setting:.3f} mm"
        print(
            f"{('uniform' if setting == 0.0 else f'{setting:g}'):>9}"
            f"  {count:>9}  {size:>25}"
            f"  {hole_error.max():>9.4f} mm  "
            f"({100.0 * hole_error.max() / tightest:.1f}% of the radius)"
        )

    out = Path(args.out or f"scratch/mesh_preview_h{args.h:g}.png")
    plot_mesh_preview(out, settings, meshes, loops[1:], args.h)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
