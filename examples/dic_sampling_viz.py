"""Preview the DIC measurement cloud over the fine and coarse notch meshes.

The static twin of ``examples/dic_sampling_viz.html``. It meshes the real notch
geometry with gmsh at the fine (truth) and coarse (inversion) element sizes,
builds the measurement grid the calibration example samples at, and draws them
side by side so the sampling density, edge margins, and notch coverage can be
judged against the actual meshes rather than the page's structured approximation.

``build_measurement_grid`` is shared with the calibration example so the picture
and the sampled data always agree. ``notch_mesh_2d`` builds the same outline used
for a plane stress mesh.

Usage:
    python examples/dic_sampling_viz.py [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

# --- geometry, sampling, and mesh sizes ---------------------------------------
WIDTH = 1.0
HEIGHT = 1.0
NOTCH_RADIUS = 0.20
PITCH = 0.020
EDGE_MARGIN = 0.02
FREE_EDGE_BAND = 0.020
FINE_H = 0.04
COARSE_H = 0.05
# ------------------------------------------------------------------------------

# palette echoing the web helper
TEAL = "#0e8f9c"
FAINT = "#8a97a4"
CLAY = "#c07f5a"
INDIGO = "#5a63a6"


def build_measurement_grid(
        pitch: float,
        margin: float,
        notch_radius: float,
        free_edge_band: float,
        *,
        width: float = 1.0,
        height: float = 1.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Captured and dropped measurement coordinates on the front face.

    A regular grid at ``pitch`` over ``[0, width] x [0, height]`` with the outer
    ``margin`` ring dropped (edge decorrelation) and points within
    ``notch_radius + free_edge_band`` of the origin corner removed (the notch and
    its free edge band). Returns ``(captured, dropped)``, each ``(n, 2)``.
    """
    xs = np.arange(0.0, width + 1e-9, pitch)
    ys = np.arange(0.0, height + 1e-9, pitch)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    x, y = pts[:, 0], pts[:, 1]
    in_margin = (
        (x < margin) | (x > width - margin)
        | (y < margin) | (y > height - margin)
    )
    in_notch = np.hypot(x, y) < notch_radius + free_edge_band
    keep = ~(in_margin | in_notch)
    return pts[keep], pts[~keep]


def notch_mesh_2d(
        h: float,
        *,
        width: float = 1.0,
        height: float = 1.0,
        notch_radius: float = 0.2,
) -> tuple[NDArray[np.floating], NDArray[np.intp]]:
    """A 2D triangulation of the notched plate outline at element size ``h``.

    A ``width`` by ``height`` rectangle minus a quarter disk of ``notch_radius``
    at the origin corner, meshed in tris at uniform ``h``. Returns node coords
    ``(n, 2)`` and triangle connectivity ``(m, 3)``. Same in plane outline as the
    3D notch front face, so it also serves as the plane stress mesh.
    """
    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("notch2d")
        rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, width, height)
        disk = gmsh.model.occ.addDisk(
            0.0, 0.0, 0.0, notch_radius, notch_radius,
        )
        out, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk)])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(
            2, [tag for _dim, tag in out], name="solid",
        )
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.model.mesh.generate(2)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        xy = coords.reshape(-1, 3)[:, :2]
        idx = {int(t): i for i, t in enumerate(node_tags)}
        conn: NDArray[np.intp] = np.empty((0, 3), dtype=np.intp)
        etypes, _etags, enodes = gmsh.model.mesh.getElements(2)
        for etype, enode in zip(etypes, enodes, strict=True):
            if etype == 2:  # linear triangle (gmsh element type 2)
                tri_tags = enode.reshape(-1, 3)
                conn = np.array(
                    [[idx[int(n)] for n in row] for row in tri_tags],
                    dtype=np.intp,
                )
        return xy, conn
    finally:
        if started:
            gmsh.finalize()


def _draw(
        ax: Axes,
        coords: NDArray[np.floating],
        tris: NDArray[np.intp],
        captured: NDArray[np.floating],
        dropped: NDArray[np.floating],
        title: str,
        mesh_color: str,
) -> None:
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.triplot(
        coords[:, 0], coords[:, 1], tris, color=mesh_color, lw=0.5, alpha=0.8,
    )
    if len(dropped):
        ax.scatter(dropped[:, 0], dropped[:, 1], s=4, color=FAINT, alpha=0.5)
    ax.scatter(captured[:, 0], captured[:, 1], s=9, color=TEAL)
    m = EDGE_MARGIN
    ax.add_patch(Rectangle(
        (m, m), WIDTH - 2 * m, HEIGHT - 2 * m,
        fill=False, ec=CLAY, ls="--", lw=1.0,
    ))
    th = np.linspace(0.0, np.pi / 2, 40)
    ax.plot(
        NOTCH_RADIUS * np.cos(th), NOTCH_RADIUS * np.sin(th),
        color=CLAY, lw=1.5,
    )
    ax.annotate(
        "", xy=(0.5, HEIGHT + 0.06), xytext=(0.5, HEIGHT),
        arrowprops={"arrowstyle": "->", "color": TEAL},
    )
    ax.set_xlim(-0.05, WIDTH + 0.05)
    ax.set_ylim(-0.05, HEIGHT + 0.14)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="dic_sampling_viz.png",
        help="output figure path (default dic_sampling_viz.png)",
    )
    args = parser.parse_args()

    captured, dropped = build_measurement_grid(
        PITCH, EDGE_MARGIN, NOTCH_RADIUS, FREE_EDGE_BAND,
        width=WIDTH, height=HEIGHT,
    )
    fine_xy, fine_tris = notch_mesh_2d(
        FINE_H, width=WIDTH, height=HEIGHT, notch_radius=NOTCH_RADIUS,
    )
    coarse_xy, coarse_tris = notch_mesh_2d(
        COARSE_H, width=WIDTH, height=HEIGHT, notch_radius=NOTCH_RADIUS,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    _draw(
        axes[0], fine_xy, fine_tris, captured, dropped,
        f"Fine mesh (truth), h={FINE_H}  |  {len(fine_tris)} tris", FAINT,
    )
    _draw(
        axes[1], coarse_xy, coarse_tris, captured, dropped,
        f"Coarse mesh (inversion), h={COARSE_H}  |  {len(coarse_tris)} tris",
        INDIGO,
    )
    ratio = COARSE_H / PITCH
    fig.suptitle(
        f"DIC sampling: {len(captured)} points, pitch {PITCH} "
        f"({ratio:.1f}x coarse element)  -  margin {EDGE_MARGIN}, "
        f"notch r {NOTCH_RADIUS}",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(
        f"wrote {out}  ({len(captured)} captured points, "
        f"fine {len(fine_tris)} tris, coarse {len(coarse_tris)} tris)"
    )


if __name__ == "__main__":
    main()
