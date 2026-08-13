"""Generate an FE mesh of a Jones et al. 304L specimen as a gmsh ``.msh``.

The nominal outline covers the whole coupon, ends included. The ends are
gripped during the test, so the mesh covers the band between two cuts in
y, with the cut edges carrying the Dirichlet conditions. Choose the cuts
with ``jones_304l_cut_preview.py``, which reports where the data reaches.

Omitting ``--thickness`` meshes the trimmed face in linear triangles.
Supplying it extrudes that face and meshes the solid in linear tets. The
mesh carries one physical group, ``solid``; boundary sets come from
``build coordinate sidesets`` at load time, which gives the cut edges as
``ymin_sides`` and ``ymax_sides``, and in 3D the front and back faces as
``zmin_sides`` and ``zmax_sides``.

``--h`` sets the element size away from curvature. Where the boundary
turns, ``--curvature-elements`` refines it toward the size that puts that
many elements around a full turn, so a tight corner follows the outline
instead of cutting the arc off, and the rest of the face stays at
``--h``. Choose the value with ``jones_304l_mesh_preview.py``, which
reports the error each setting leaves on the specimen being meshed.

Coordinates are translated after trimming so the bounding box is centered
on the origin. A point cloud in the file's own system is brought onto the
mesh by subtracting the same offset, which the run prints and
``jones_304l_geometry.centering_offset`` recomputes.

Usage:
    python examples/jones_304l_mesh.py --y-min 65 --y-max 179 --h 2.0
    python examples/jones_304l_mesh.py --y-min 65 --y-max 179 --h 2.0 \\
        --thickness 3.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gmsh
import numpy as np
from jones_304l_geometry import (
    CENTER_MODES,
    Loop,
    centering_offset,
    clean_loop,
    read_nominal_geometry,
    trim_outline,
    trimmed_area,
    validate_cuts,
)
from numpy.typing import NDArray

# Elements per full turn of the boundary. A mesh coarse enough to be
# affordable elsewhere cuts across a tight turn instead of following it.
# The specimens differ in how tightly their holes turn, so
# jones_304l_mesh_preview.py reports what a setting buys on the one at
# hand.
CURVATURE_ELEMENTS = 20.0


def _corner_indices(loop: Loop, y_min: float, y_max: float) -> NDArray[np.intp]:
    """Indices of the loop points that lie on either cut line."""
    on_cut = np.isclose(loop[:, 1], y_min) | np.isclose(loop[:, 1], y_max)
    return np.flatnonzero(on_cut)


def _locally_collinear(loop: Loop, tol: float) -> NDArray[np.bool_]:
    """True at each point lying within ``tol`` of the chord through its neighbors."""
    previous, following = np.roll(loop, 1, axis=0), np.roll(loop, -1, axis=0)
    chord = following - previous
    offset = loop - previous
    twice_area = np.abs(chord[:, 0] * offset[:, 1] - chord[:, 1] * offset[:, 0])
    return twice_area / np.maximum(np.linalg.norm(chord, axis=1), 1e-300) <= tol


def _loop_curves(
        loop: Loop, z: float, h: float, *, tol: float,
        forced_breaks: NDArray[np.intp] | tuple[()] = (),
) -> list[int]:
    """Add one closed outline as gmsh curves and return their tags.

    The outline is split into runs that are straight and runs that are
    not, so a straight portion becomes an exact line instead of a spline
    through collinear points, which would bow outward where it meets a
    curve. Curved runs become splines, leaving the mesh size free rather
    than pinning a node at every sampled point.

    ``forced_breaks`` names points that must end a run whatever the local
    shape, used for the cut corners so each cut edge is its own curve.
    """
    n_points = len(loop)
    collinear = _locally_collinear(loop, tol)
    tags = [gmsh.model.occ.addPoint(point[0], point[1], z, h) for point in loop]

    # A maximal run of collinear points spans a straight segment whose ends
    # are the points just outside it, so those bound the runs.
    breaks = {int(i) for i in forced_breaks}
    for i in range(n_points):
        if collinear[i] and not collinear[i - 1]:
            breaks.add((i - 1) % n_points)
        if collinear[i] and not collinear[(i + 1) % n_points]:
            breaks.add((i + 1) % n_points)
    if not breaks:
        return [gmsh.model.occ.addSpline([*tags, tags[0]])]

    ordered = sorted(breaks)
    curves: list[int] = []
    for k, start in enumerate(ordered):
        stop = ordered[(k + 1) % len(ordered)]
        step = (stop - start) % n_points or n_points
        run = [(start + offset) % n_points for offset in range(step + 1)]
        if all(collinear[i] for i in run[1:-1]):
            curves.append(gmsh.model.occ.addLine(tags[start], tags[stop]))
        else:
            curves.append(gmsh.model.occ.addSpline([tags[i] for i in run]))
    return curves


def build_specimen_mesh(
        path: Path, h: float, *, geometry_file: str | Path,
        y_min: float, y_max: float, thickness: float | None = None,
        center: str = "xy", straight_tol: float = 1.0e-6,
        curvature_elements: float = CURVATURE_ELEMENTS,
) -> tuple[int, NDArray[np.float64], float]:
    """Write the mesh to ``path``.

    ``straight_tol`` is the distance below which three consecutive
    outline points count as collinear, in the units of the geometry file.

    ``curvature_elements`` is the number of elements gmsh aims to place
    per full turn of the boundary, so a run of radius ``r`` is meshed at
    ``2 * pi * r / curvature_elements`` where that falls below ``h``.
    Zero meshes at ``h`` everywhere.

    Returns the element count, the translation applied to the
    coordinates, and the area of the trimmed face, the last as an
    independent check on the geometry that reached gmsh.
    """
    if center not in CENTER_MODES:
        raise ValueError(f"center mode {center!r} is not one of {CENTER_MODES}")
    if thickness is not None and thickness <= 0.0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    if curvature_elements < 0.0:
        raise ValueError(
            f"curvature_elements must not be negative, got {curvature_elements}"
        )

    outer, holes = read_nominal_geometry(geometry_file)
    outer = clean_loop(outer)
    holes = [clean_loop(hole) for hole in holes]
    validate_cuts(outer, holes, y_min, y_max)

    trimmed = trim_outline(outer, y_min, y_max)
    kept = [
        hole for hole in holes
        if y_min <= hole[:, 1].min() and hole[:, 1].max() <= y_max
    ]
    offset = centering_offset(trimmed, kept, center)
    trimmed = trimmed - offset
    kept = [hole - offset for hole in kept]
    area = trimmed_area(trimmed, kept)

    base_z = -0.5 * thickness if (thickness and center == "xyz") else 0.0

    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("specimen")

        corners = _corner_indices(
            trimmed, y_min - offset[1], y_max - offset[1],
        )
        if corners.size < 4:
            raise ValueError(
                f"found {corners.size} points on the cut lines, expected at "
                f"least 4; the trim did not insert the crossings as expected"
            )
        loops = [gmsh.model.occ.addCurveLoop(
            _loop_curves(trimmed, base_z, h, tol=straight_tol,
                         forced_breaks=corners)
        )]
        loops += [
            gmsh.model.occ.addCurveLoop(
                _loop_curves(hole, base_z, h, tol=straight_tol)
            )
            for hole in kept
        ]
        surface = gmsh.model.occ.addPlaneSurface(loops)

        if thickness is None:
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(2, [surface], name="solid")
            dim = 2
        else:
            extruded = gmsh.model.occ.extrude([(2, surface)], 0.0, 0.0, thickness)
            volumes = [tag for entity_dim, tag in extruded if entity_dim == 3]
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(3, volumes, name="solid")
            dim = 3

        # A size floor of h would cancel the curvature refinement, so the
        # curvature rule alone sets how small an element gets.
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0 if curvature_elements else h)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_elements)
        gmsh.model.mesh.generate(dim)
        _types, tags, _nodes = gmsh.model.mesh.getElements(dim)
        path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(path))
        return int(sum(len(group) for group in tags)), offset, area
    finally:
        if started:
            gmsh.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry", default="data/jones_304l/X-Specimen-NominalGeometry.mat",
        help="nominal geometry .mat",
    )
    parser.add_argument("--y-min", type=float, required=True, help="lower cut")
    parser.add_argument("--y-max", type=float, required=True, help="upper cut")
    parser.add_argument(
        "--h", type=float, default=2.0,
        help="element size away from curvature (default 2)",
    )
    parser.add_argument(
        "--curvature-elements", type=float, default=CURVATURE_ELEMENTS,
        help=f"elements per full turn of the boundary, 0 to mesh at --h "
             f"everywhere (default {CURVATURE_ELEMENTS:g})",
    )
    parser.add_argument(
        "--thickness", type=float, default=None,
        help="extrude to a solid of this thickness; omit for a 2D mesh",
    )
    parser.add_argument(
        "--center", default="xy", choices=CENTER_MODES,
        help="center the trimmed geometry on the origin (default xy)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output path (default examples/meshes/jones_304l_<...>.msh)",
    )
    args = parser.parse_args()

    kind = "2d" if args.thickness is None else f"3d_t{args.thickness:g}"
    out = Path(
        args.out
        or f"examples/meshes/jones_304l_{kind}"
        f"_y{args.y_min:g}_{args.y_max:g}_h{args.h:g}.msh"
    )
    n_elements, offset, area = build_specimen_mesh(
        out, args.h, geometry_file=args.geometry, y_min=args.y_min,
        y_max=args.y_max, thickness=args.thickness, center=args.center,
        curvature_elements=args.curvature_elements,
    )
    print(f"trimmed face area {area:.6f}")
    print(
        f"centering offset ({offset[0]:.6g}, {offset[1]:.6g}); subtract it "
        f"from data coordinates to overlay the mesh"
    )
    print(f"wrote {out} ({n_elements} elements)")


if __name__ == "__main__":
    main()
