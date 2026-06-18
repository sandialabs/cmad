"""Generate a notched plate mesh as a gmsh ``.msh``.

The geometry is a rectangular plate with a quarter cylinder notch at the origin
corner, meshed in tets at uniform size ``h``, carrying one physical volume
``"solid"``. The plate extent and notch radius are inputs.

Usage:
    python examples/notch_mesh.py [--h H] [--plate LX LY LZ] [--radius R]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gmsh


def generate_notch_msh(
        path: Path, h: float, *,
        plate: tuple[float, float, float] = (1.0, 1.0, 0.1),
        notch_radius: float = 0.2,
) -> int:
    """Write a notch tet mesh at uniform size ``h`` to ``path`` and return the
    element count.

    ``plate`` is the ``(lx, ly, lz)`` extent and ``notch_radius`` the radius of
    the quarter cylinder notch removed at the origin corner.
    """
    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("notch")
        lx, ly, lz = plate
        box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, lx, ly, lz)
        notch = gmsh.model.occ.addCylinder(
            0.0, 0.0, -lz, 0.0, 0.0, 3.0 * lz, notch_radius,
        )
        out, _ = gmsh.model.occ.cut([(3, box)], [(3, notch)])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [tag for _dim, tag in out], name="solid")
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.model.mesh.generate(3)
        _types, tags, _nodes = gmsh.model.mesh.getElements(3)
        gmsh.write(str(path))
        return len(tags[0])
    finally:
        if started:
            gmsh.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h", type=float, default=0.08,
        help="uniform element size (default 0.08)",
    )
    parser.add_argument(
        "--plate", type=float, nargs=3, metavar=("LX", "LY", "LZ"),
        default=(1.0, 1.0, 0.1), help="plate extent (default 1 1 0.1)",
    )
    parser.add_argument(
        "--radius", type=float, default=0.2,
        help="notch radius (default 0.2)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output path (default examples/meshes/notch_h{h}.msh)",
    )
    args = parser.parse_args()
    out = args.out or f"examples/meshes/notch_h{args.h:.3f}.msh"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n_elem = generate_notch_msh(
        Path(out), args.h, plate=tuple(args.plate), notch_radius=args.radius,
    )
    print(f"wrote {out} ({n_elem} tets)")


if __name__ == "__main__":
    main()
