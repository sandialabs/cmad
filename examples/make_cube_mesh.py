"""Generate the unit-cube Exodus mesh for the elastic_plastic_uniaxial example.

Writes a structured unit cube (hex, or hex-split-to-tet) carrying the
``{x,y,z}{min,max}_sides`` sidesets the example deck's Dirichlet BCs
reference. Mesh size and element kind are CLI arguments so the example
can be regenerated at any resolution.

Usage:
    python examples/make_cube_mesh.py [--n N] [--kind {hex,tet}] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cmad.fem.mesh import StructuredHexMesh, hex_to_tet_split
from cmad.io.exodus import ExodusWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=8,
        help="per-axis divisions of the unit cube (default 8)",
    )
    parser.add_argument(
        "--kind", choices=("hex", "tet"), default="hex",
    )
    parser.add_argument(
        "--out", default=None,
        help="output path (default examples/cube_{kind}_{n}.exo)",
    )
    args = parser.parse_args()

    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (args.n, args.n, args.n))
    if args.kind == "tet":
        mesh = hex_to_tet_split(mesh)

    out = args.out or f"examples/cube_{args.kind}_{args.n}.exo"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with ExodusWriter(out, mesh):
        pass
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
