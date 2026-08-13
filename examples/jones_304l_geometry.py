"""Nominal specimen geometry for the Jones et al. 304L DIC dataset.

The dataset's ``NominalGeometry`` MATLAB file holds the outline of the
entire coupon: an outer loop and zero or more hole loops, sampled at
uniform arc length in mm. The ends of the coupon sit inside the grips
during the test, so a model covers a window of it rather than the whole
outline, and this module trims to that window.

Cut values are in the coordinate system of the geometry file. Centering
is applied after trimming, so a point cloud in that same system is
overlaid on the trimmed geometry by subtracting
:func:`centering_offset`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

Loop = NDArray[np.float64]

CENTER_MODES = ("none", "xy", "xyz")


class GeometryFormatError(ValueError):
    """The geometry file does not hold the expected outline variables."""


def read_nominal_geometry(path: str | Path) -> tuple[Loop, list[Loop]]:
    """Read the outer outline and the hole outlines from ``path``.

    Returns ``(outer, holes)`` with each loop shaped ``(n_points, 2)`` in
    the file's own units. Loops are open: the last point is not a repeat
    of the first. ``holes`` is empty when the file carries no ``bHole``.
    """
    path = Path(path)
    try:
        contents = loadmat(str(path))
    except FileNotFoundError as err:
        raise GeometryFormatError(f"geometry file not found: {path}") from err

    if "bOut" not in contents:
        raise GeometryFormatError(
            f"{path}: no 'bOut' variable; found "
            f"{sorted(k for k in contents if not k.startswith('__'))}"
        )
    outer = _as_loop(contents["bOut"], "bOut", path)

    holes: list[Loop] = []
    if "bHole" in contents:
        entries = np.asarray(contents["bHole"], dtype=object).ravel()
        holes = [
            _as_loop(entry, f"bHole[{i}]", path) for i, entry in enumerate(entries)
        ]
    return outer, holes


def _as_loop(array: object, name: str, path: Path) -> Loop:
    """Validate one outline variable and return it as ``(n_points, 2)``."""
    loop = np.asarray(array, dtype=np.float64)
    if loop.ndim != 2 or loop.shape[1] != 2 or loop.shape[0] < 3:
        raise GeometryFormatError(
            f"{path}: {name} has shape {loop.shape}, expected (n_points, 2) "
            f"with at least 3 points"
        )
    if not np.all(np.isfinite(loop)):
        raise GeometryFormatError(f"{path}: {name} holds non-finite coordinates")
    return loop


def loop_area(loop: Loop) -> float:
    """Signed area of a closed loop by the shoelace formula.

    Positive for counterclockwise ordering, negative for clockwise. The
    loop is treated as closed, so the final point connects to the first.
    """
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def segment_lengths(loop: Loop) -> NDArray[np.float64]:
    """Length of each segment of a closed loop, including the closing one."""
    return np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1)


def clean_loop(loop: Loop, *, min_segment: float | None = None) -> Loop:
    """Drop points that sit closer than ``min_segment`` to the previous one.

    Near duplicate points produce sliver elements or outright meshing
    failures. ``min_segment`` defaults to a quarter of the loop's own
    median segment length, so a loop sampled at a different spacing is
    cleaned on its own scale. The closing segment is checked too, so the
    last point is dropped when it collapses onto the first.
    """
    if min_segment is None:
        min_segment = 0.25 * float(np.median(segment_lengths(loop)))

    kept = [loop[0]]
    for point in loop[1:]:
        if np.linalg.norm(point - kept[-1]) >= min_segment:
            kept.append(point)
    if len(kept) > 3 and np.linalg.norm(kept[-1] - kept[0]) < min_segment:
        kept.pop()
    return np.asarray(kept, dtype=np.float64)


def count_crossings(loop: Loop, y_cut: float) -> int:
    """Number of times the closed ``loop`` crosses the line ``y = y_cut``."""
    y = loop[:, 1]
    y_next = np.roll(y, -1)
    return int(np.sum((y - y_cut) * (y_next - y_cut) < 0.0))


def validate_cuts(
        outer: Loop, holes: list[Loop], y_min: float, y_max: float,
) -> None:
    """Raise when the cuts do not produce a single trimmed region.

    A cut is rejected when it falls outside the coupon, when it crosses
    the outer outline anywhere other than exactly twice, or when it
    passes through a hole. Crossing exactly twice is what keeps each cut
    edge a single straight segment, and it is the condition
    :func:`trim_outline` relies on.
    """
    if not y_min < y_max:
        raise ValueError(f"y_min ({y_min}) must be below y_max ({y_max})")

    lo, hi = float(outer[:, 1].min()), float(outer[:, 1].max())
    for name, cut in (("y_min", y_min), ("y_max", y_max)):
        if not lo < cut < hi:
            raise ValueError(
                f"{name} ({cut}) is outside the coupon, which spans "
                f"y in [{lo:.6g}, {hi:.6g}]"
            )
        n_cross = count_crossings(outer, cut)
        if n_cross != 2:
            raise ValueError(
                f"{name} ({cut}) crosses the outer outline {n_cross} times, "
                f"expected exactly 2"
            )
        for i, hole in enumerate(holes):
            h_lo, h_hi = float(hole[:, 1].min()), float(hole[:, 1].max())
            if h_lo < cut < h_hi:
                raise ValueError(
                    f"{name} ({cut}) passes through hole {i}, which spans "
                    f"y in [{h_lo:.6g}, {h_hi:.6g}]"
                )


def crossings(loop: Loop, y_cut: float) -> NDArray[np.float64]:
    """Sorted x of every point where the closed ``loop`` meets ``y = y_cut``."""
    y = loop[:, 1]
    x = loop[:, 0]
    y_next, x_next = np.roll(y, -1), np.roll(x, -1)
    hit = (y - y_cut) * (y_next - y_cut) < 0.0
    t = (y_cut - y[hit]) / (y_next[hit] - y[hit])
    return np.sort(x[hit] + t * (x_next[hit] - x[hit]))


def section_width(outer: Loop, y_cut: float) -> float:
    """Width of the section at ``y_cut``, or NaN where it is not a single span."""
    xs = crossings(outer, y_cut)
    return float(xs[-1] - xs[0]) if xs.size == 2 else float("nan")


def prismatic_windows(
        outer: Loop, holes: list[Loop], *,
        n_samples: int = 2000, slope_tol: float = 1.0e-3,
) -> list[tuple[float, float]]:
    """Ranges of y over which a cut gives a clean, constant width section.

    A sample qualifies when the cut crosses the outer outline exactly
    twice, misses every hole, and the section width is locally constant,
    meaning ``|d(width)/dy|`` is at or below ``slope_tol``. The width
    condition is what distinguishes the straight grip section from the
    fillet, without assuming the sides are vertical or axis aligned.

    Returns the qualifying ranges as ``(low, high)`` pairs, ordered by
    ``low``, in the units of the outline.
    """
    lo, hi = float(outer[:, 1].min()), float(outer[:, 1].max())
    span = hi - lo
    y = np.linspace(lo + 1.0e-6 * span, hi - 1.0e-6 * span, n_samples)
    width = np.array([section_width(outer, value) for value in y])

    ok = np.isfinite(width)
    for hole in holes:
        h_lo, h_hi = float(hole[:, 1].min()), float(hole[:, 1].max())
        ok &= ~((y > h_lo) & (y < h_hi))

    slope = np.full_like(width, np.inf)
    interior = slice(1, -1)
    with np.errstate(invalid="ignore"):
        slope[interior] = np.abs(
            (width[2:] - width[:-2]) / (y[2:] - y[:-2])
        )
    ok &= np.isfinite(slope) & (slope <= slope_tol)

    return _runs_to_ranges(y, ok)


def _runs_to_ranges(
        y: NDArray[np.float64], ok: NDArray[np.bool_],
) -> list[tuple[float, float]]:
    """Group consecutive qualifying samples into ``(low, high)`` ranges."""
    ranges: list[tuple[float, float]] = []
    start: int | None = None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            ranges.append((float(y[start]), float(y[i - 1])))
            start = None
    if start is not None:
        ranges.append((float(y[start]), float(y[-1])))
    return ranges


def default_cuts(
        outer: Loop, holes: list[Loop], *,
        y_limits: tuple[float, float] | None = None,
        round_to: float | None = 1.0,
        **window_kwargs: float,
) -> tuple[float, float]:
    """Suggest a cut pair: the midpoints of the outermost usable windows.

    ``y_limits`` restricts the windows to a range, typically the extent
    of the measured data, so the cuts keep data on both sides of each
    Dirichlet edge. The lowest surviving window supplies ``y_min`` and
    the highest supplies ``y_max``, each taken at its midpoint, which
    balances the margin outside the cut against the distance from the
    change in section.

    ``round_to`` snaps each midpoint to a multiple of that value, 1 mm by
    default, since the midpoint's fractional part reflects the sampling
    of the window search rather than anything about the specimen. A
    snapped value that would leave its window is not used; pass ``None``
    to keep the midpoints exactly.
    """
    windows = prismatic_windows(outer, holes, **window_kwargs)  # type: ignore[arg-type]
    if y_limits is not None:
        low, high = y_limits
        windows = [
            (max(a, low), min(b, high)) for a, b in windows
            if min(b, high) > max(a, low)
        ]
    if len(windows) < 2:
        raise ValueError(
            f"found {len(windows)} usable cut window(s), need 2; "
            f"supply the cuts explicitly"
        )
    return (
        _snapped_midpoint(*windows[0], round_to),
        _snapped_midpoint(*windows[-1], round_to),
    )


def _snapped_midpoint(low: float, high: float, round_to: float | None) -> float:
    """Midpoint of ``[low, high]``, snapped to a multiple of ``round_to``.

    Falls back to the exact midpoint when snapping would land on or
    outside an end, which happens when the window is narrower than the
    rounding step.
    """
    middle = 0.5 * (low + high)
    if round_to is None or round_to <= 0.0:
        return middle
    snapped = round(middle / round_to) * round_to
    return snapped if low < snapped < high else middle


def trim_outline(outer: Loop, y_min: float, y_max: float) -> Loop:
    """Clip the outer outline to the band ``y_min <= y <= y_max``.

    Returns the trimmed loop with the crossing points inserted, so the
    two cut edges are straight segments between them. Call
    :func:`validate_cuts` first: the clip is exact for a convex region
    and a connected result, and requiring each cut to cross exactly
    twice is what makes the result a single loop.
    """
    clipped = _clip_half_plane(outer, y_min, keep_above=True)
    return _clip_half_plane(clipped, y_max, keep_above=False)


def _clip_half_plane(loop: Loop, bound: float, *, keep_above: bool) -> Loop:
    """Clip a closed loop against a horizontal line, keeping one side.

    One pass of Sutherland-Hodgman polygon clipping (Sutherland and
    Hodgman, "Reentrant polygon clipping", CACM 17(1), 1974): walk the
    edges, emitting an endpoint when it is inside and a crossing point
    wherever an edge changes side. Clipping to a band is two passes, one
    per bounding line, which is what :func:`trim_outline` does.
    """
    inside = (loop[:, 1] >= bound) if keep_above else (loop[:, 1] <= bound)
    kept: list[NDArray[np.float64]] = []
    n = len(loop)
    for i in range(n):
        j = (i + 1) % n
        if inside[i]:
            kept.append(loop[i])
        if inside[i] != inside[j]:
            span = loop[j, 1] - loop[i, 1]
            t = (bound - loop[i, 1]) / span
            kept.append(loop[i] + t * (loop[j] - loop[i]))
    return np.asarray(kept, dtype=np.float64)


def centering_offset(
        outer: Loop, holes: list[Loop], mode: str = "xy",
) -> NDArray[np.float64]:
    """In plane translation that moves the bounding box center to the origin.

    Returns the ``(dx, dy)`` to SUBTRACT from a coordinate in the file's
    system. ``mode`` of ``"none"`` gives zeros; ``"xy"`` and ``"xyz"``
    give the same in plane shift, and ``"xyz"`` additionally centers an
    extrusion about ``z = 0``, which the mesh generator applies since it
    depends on the thickness rather than on the outline.

    Pass the trimmed outer loop: centering follows the trim, so the
    untrimmed coupon gives a different and wrong shift.
    """
    if mode not in CENTER_MODES:
        raise ValueError(f"center mode {mode!r} is not one of {CENTER_MODES}")
    if mode == "none":
        return np.zeros(2, dtype=np.float64)

    points = np.vstack([outer, *holes]) if holes else outer
    return 0.5 * (points.min(axis=0) + points.max(axis=0))


def trimmed_area(outer: Loop, holes: list[Loop]) -> float:
    """Area enclosed by the outer loop with every hole removed."""
    return abs(loop_area(outer)) - sum(abs(loop_area(hole)) for hole in holes)
