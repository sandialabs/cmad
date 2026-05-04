"""Generate ``small_hex.exo`` test fixture using ``meshio``.

Run from a one-shot meshio environment so cmad's env stays clean::

    conda create -n meshio_tmp -c conda-forge meshio python=3.11 -y
    conda activate meshio_tmp
    python tests/io/fixtures/_generate_fixture.py
    conda env remove -n meshio_tmp

The fixture is a 2x2x2 hex mesh on the unit cube, with one element
block ("all") and two nodesets ("xmin_nodes", "xmax_nodes"). meshio
does not have a first-class sideset concept; sideset reading is
covered by round-trip tests via cmad's own writer.

meshio writes ``eb_prop1=[0]`` and ``ns_prop1=[0, 1]``, which violates
Exodus's 1-based-positive ID convention and would be rejected by
:class:`cmad.fem.mesh.Mesh.__post_init__`. The script post-processes
the meshio output via :mod:`netCDF4` to bump those values to 1-based
(meshio depends on netCDF4 internally so it's already in the
generation env).
"""
from __future__ import annotations

from pathlib import Path

import meshio
import netCDF4
import numpy as np


def main() -> None:
    nx, ny, nz = 2, 2, 2
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    zs = np.linspace(0.0, 1.0, nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    vid = np.arange(
        (nx + 1) * (ny + 1) * (nz + 1), dtype=np.int64
    ).reshape(nx + 1, ny + 1, nz + 1)
    EI, EJ, EK = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    connectivity = np.stack(
        [
            vid[EI, EJ, EK],               # 0: (-,-,-)
            vid[EI + 1, EJ, EK],           # 1: (+,-,-)
            vid[EI + 1, EJ + 1, EK],       # 2: (+,+,-)
            vid[EI, EJ + 1, EK],           # 3: (-,+,-)
            vid[EI, EJ, EK + 1],           # 4: (-,-,+)
            vid[EI + 1, EJ, EK + 1],       # 5: (+,-,+)
            vid[EI + 1, EJ + 1, EK + 1],   # 6: (+,+,+)
            vid[EI, EJ + 1, EK + 1],       # 7: (-,+,+)
        ],
        axis=-1,
    ).reshape(-1, 8)

    point_sets = {
        "xmin_nodes": vid[0, :, :].ravel(),
        "xmax_nodes": vid[-1, :, :].ravel(),
    }
    cell_sets = {"all": [np.arange(connectivity.shape[0], dtype=np.int64)]}

    mesh = meshio.Mesh(
        points=points,
        cells=[("hexahedron", connectivity)],
        point_sets=point_sets,
        cell_sets=cell_sets,
    )

    out_path = Path(__file__).parent / "small_hex.exo"
    meshio.write(out_path, mesh, file_format="exodus")

    # meshio writes 0-based prop1 IDs; Exodus convention is 1-based
    # positive integers and cmad's reader enforces this. Bump in place.
    with netCDF4.Dataset(str(out_path), "r+") as ds:
        ds["eb_prop1"][:] = np.array([1], dtype=ds["eb_prop1"].dtype)
        ds["ns_prop1"][:] = np.array([1, 2], dtype=ds["ns_prop1"].dtype)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
