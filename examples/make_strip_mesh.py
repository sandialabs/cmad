"""Generate the strip mesh for the thin_plate_pulled example.

Writes a rectangle of quads in the x-y plane carrying the
``{x,y}{min,max}_sides`` sidesets the example input file's boundary
conditions reference. The lengths and divisions are CLI arguments so the
example can be regenerated at any resolution.

Usage:
    python examples/make_strip_mesh.py [--length L] [--width W] [--nx NX]
        [--ny NY] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cmad.fem.mesh import StructuredQuadMesh
from cmad.io.exodus import ExodusWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--length", type=float, default=10.0,
        help="length in x (default 10)",
    )
    parser.add_argument(
        "--width", type=float, default=2.0,
        help="width in y (default 2)",
    )
    parser.add_argument(
        "--nx", type=int, default=20,
        help="divisions along x (default 20)",
    )
    parser.add_argument(
        "--ny", type=int, default=4,
        help="divisions along y (default 4)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output path (default examples/meshes/strip_quad_{nx}x{ny}.exo)",
    )
    args = parser.parse_args()

    mesh = StructuredQuadMesh((args.length, args.width), (args.nx, args.ny))

    out = args.out or f"examples/meshes/strip_quad_{args.nx}x{args.ny}.exo"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with ExodusWriter(out, mesh):
        pass
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
