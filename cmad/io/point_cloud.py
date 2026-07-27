"""Scattered point-cloud container with text / HDF5 I/O and XDMF viz.

A :class:`PointCloud` holds point coordinates in the reference
configuration and one or more named fields sampled at those points over a
sequence of steps. The code is total Lagrangian, so the coordinates are the
fixed reference positions for the whole history.

Readers normalize text and our own HDF5 format into the same container;
writers emit HDF5 (compact storage) plus an XDMF file -- a small XML index
ParaView reads as a point cloud animated over time -- and, on request, text.

Field layout follows the FE results convention
(:func:`cmad.io.exodus.read_results`): a field is shaped
``(num_steps, num_points, num_components)`` so a single step is
``field[step]`` of shape ``(num_points, num_components)``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PointCloud:
    """Points plus named fields sampled at them per step.

    ``coords`` has shape ``(num_points, dim)`` with ``dim`` 2 or 3 -- the
    reference positions, fixed for the whole history.

    ``times`` has shape ``(num_steps,)``.

    ``fields`` maps a field name to a ``(num_steps, num_points,
    num_components)`` array (for example ``"displacement"``). The component
    count is whatever the field carries and may differ between fields.

    Frozen dataclass; validation runs in ``__post_init__``.
    """

    coords: NDArray[np.floating]
    times: NDArray[np.floating]
    fields: dict[str, NDArray[np.floating]]

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] not in (2, 3):
            raise ValueError(
                f"coords must have shape (num_points, 2|3); got "
                f"{self.coords.shape}"
            )
        if self.times.ndim != 1:
            raise ValueError(
                f"times must be 1D (num_steps,); got shape {self.times.shape}"
            )
        n_steps = self.times.shape[0]
        n_points = self.coords.shape[0]
        for name, arr in self.fields.items():
            if arr.ndim != 3:
                raise ValueError(
                    f"field '{name}' must be 3D (num_steps, num_points, "
                    f"num_components); got shape {arr.shape}"
                )
            if arr.shape[:2] != (n_steps, n_points):
                raise ValueError(
                    f"field '{name}' must have shape (num_steps={n_steps}, "
                    f"num_points={n_points}, num_components); got {arr.shape}"
                )

    @property
    def num_points(self) -> int:
        return int(self.coords.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.times.shape[0])

    @property
    def dim(self) -> int:
        return int(self.coords.shape[1])


def read_point_cloud(path: str | Path, **text_kwargs: object) -> PointCloud:
    """Read a point cloud from a single file, dispatching on extension.

    ``.h5`` / ``.hdf5`` -> :func:`read_point_cloud_hdf5` (our own format,
    no extra arguments). ``.csv`` / ``.txt`` -> :func:`read_point_cloud_text`,
    which needs the column mapping keyword arguments. A run with one file per
    frame is read by calling :func:`read_point_cloud_text` directly with the
    list of paths.
    """
    suffix = Path(path).suffix.lower()
    if suffix in (".h5", ".hdf5"):
        return read_point_cloud_hdf5(path)
    if suffix in (".csv", ".txt"):
        return read_point_cloud_text(path, **text_kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"point cloud: unsupported extension '{suffix}' (path: {path}); "
        f"supported: .h5/.hdf5, .csv/.txt"
    )


def read_point_cloud_hdf5(path: str | Path) -> PointCloud:
    """Read a point cloud from our HDF5 layout.

    Datasets: ``/coords`` ``(num_points, dim)``, ``/times``
    ``(num_steps,)``, and a ``/fields/<name>`` group per field holding one
    ``(num_points, num_components)`` dataset per step, keyed by a zero-padded
    step index. The per-step datasets are stacked back into the
    ``(num_steps, num_points, num_components)`` array. Written by
    :func:`write_point_cloud`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"point cloud: file not found at {path}")
    with h5py.File(path, "r") as f:
        coords = np.asarray(f["coords"][()], dtype=np.float64)
        times = np.asarray(f["times"][()], dtype=np.float64)
        fields: dict[str, NDArray[np.floating]] = {}
        if "fields" in f:
            group = f["fields"]
            for name in group:
                field_group = group[name]
                steps = sorted(field_group, key=int)
                fields[name] = np.stack(
                    [
                        np.asarray(field_group[s][()], dtype=np.float64)
                        for s in steps
                    ],
                    axis=0,
                )
    return PointCloud(coords=coords, times=times, fields=fields)


def read_point_cloud_text(
        paths: str | Path | Sequence[str | Path],
        *,
        coord_cols: Sequence[int],
        field_cols: Mapping[str, Sequence[int]],
        times: Sequence[float] | None = None,
        delimiter: str | None = None,
        skip_header: int = 0,
) -> PointCloud:
    """Read a point cloud from text, one file per frame.

    Each path is one frame, in order: ``coords`` are read from the first
    frame, and each field in ``field_cols`` is read from every frame and
    stacked over time. A single path is simply one frame.

    ``coord_cols`` picks the reference coordinate columns; ``field_cols``
    maps each field name to its component columns. ``times`` is the time
    stamp of each frame (defaults to ``0, 1, ...``). ``delimiter`` and
    ``skip_header`` pass through to :func:`numpy.loadtxt` (``delimiter=None``
    splits on any whitespace).
    """
    path_list = (
        [Path(paths)]
        if isinstance(paths, (str, Path))
        else [Path(p) for p in paths]
    )
    if not path_list:
        raise ValueError("read_point_cloud_text: no paths given")

    frames: list[NDArray[np.floating]] = []
    for p in path_list:
        if not p.exists():
            raise FileNotFoundError(f"point cloud: file not found at {p}")
        data = np.loadtxt(
            p, delimiter=delimiter, skiprows=skip_header, ndmin=2,
        )
        frames.append(np.asarray(data, dtype=np.float64))

    n_points = frames[0].shape[0]
    for p, frame in zip(path_list, frames, strict=True):
        if frame.shape[0] != n_points:
            raise ValueError(
                f"point cloud: frame '{p}' has {frame.shape[0]} points but "
                f"the first frame has {n_points}; all frames must share the "
                f"same point ordering"
            )

    coords = frames[0][:, list(coord_cols)]
    fields: dict[str, NDArray[np.floating]] = {
        name: np.stack([frame[:, list(cols)] for frame in frames], axis=0)
        for name, cols in field_cols.items()
    }

    n_steps = len(frames)
    if times is None:
        times_arr = np.arange(n_steps, dtype=np.float64)
    else:
        times_arr = np.asarray(times, dtype=np.float64).ravel()
        if times_arr.shape[0] != n_steps:
            raise ValueError(
                f"point cloud: {len(times_arr)} times for {n_steps} frames"
            )
    return PointCloud(coords=coords, times=times_arr, fields=fields)


def write_point_cloud(
        path: str | Path, cloud: PointCloud, *, write_xdmf: bool = True,
) -> None:
    """Write ``cloud`` to HDF5 at ``path`` and (by default) an XDMF file.

    The HDF5 file holds the heavy arrays; the accompanying ``<stem>.xdmf``
    is a small XML index ParaView opens as a point cloud animated over time,
    referencing the HDF5 datasets by relative name. Pass ``write_xdmf=False``
    to skip it.
    """
    path = Path(path)
    write_point_cloud_hdf5(path, cloud)
    if write_xdmf:
        tree = _build_xdmf_tree(path.name, cloud)
        ET.indent(tree, space="  ")
        tree.write(
            path.with_suffix(".xdmf"), encoding="utf-8", xml_declaration=True,
        )


def write_point_cloud_hdf5(path: str | Path, cloud: PointCloud) -> None:
    """Write ``cloud`` to the HDF5 layout read by
    :func:`read_point_cloud_hdf5`.

    Each field is stored as one ``(num_points, num_components)`` dataset per
    step under a ``/fields/<name>`` group, keyed by a zero-padded step index.
    """
    with h5py.File(Path(path), "w") as f:
        f.attrs["dim"] = cloud.dim
        f.create_dataset("coords", data=cloud.coords)
        f.create_dataset("times", data=cloud.times)
        group = f.create_group("fields")
        for name, arr in cloud.fields.items():
            field_group = group.create_group(name)
            for step in range(cloud.num_steps):
                field_group.create_dataset(f"{step:04d}", data=arr[step])


def write_point_cloud_text(
        path: str | Path,
        cloud: PointCloud,
        *,
        field: str = "displacement",
        delimiter: str = " ",
) -> None:
    """Write ``[coords | field components]`` text, one file per frame.

    For a single step the columns go to ``path``; for several steps each
    frame ``k`` is written to ``<stem>_{k:04d}<suffix>``, mirroring the
    layout :func:`read_point_cloud_text` reads.
    """
    path = Path(path)
    arr = cloud.fields[field]
    for step in range(cloud.num_steps):
        block = np.hstack([cloud.coords, arr[step]])
        out = (
            path
            if cloud.num_steps == 1
            else path.with_name(f"{path.stem}_{step:04d}{path.suffix}")
        )
        np.savetxt(out, block, delimiter=delimiter)


def _attr_type(num_components: int) -> str:
    """XDMF ``AttributeType`` for a field's component count."""
    return {1: "Scalar", 6: "Tensor6", 9: "Tensor"}.get(
        num_components, "Vector",
    )


def _build_xdmf_tree(h5_name: str, cloud: PointCloud) -> ET.ElementTree:
    """Build the XDMF tree for ``cloud``, referencing ``h5_name`` datasets.

    A temporal ``Collection`` with one ``Polyvertex`` grid per step; each
    grid shares the ``/coords`` geometry and references that step's
    ``/fields/<name>/<step>`` dataset directly.
    """
    n = cloud.num_points
    dim = cloud.dim
    n_steps = cloud.num_steps
    geom_type = "XYZ" if dim == 3 else "XY"

    xdmf = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(xdmf, "Domain")
    collection = ET.SubElement(
        domain, "Grid", Name="TimeSeries",
        GridType="Collection", CollectionType="Temporal",
    )
    for step in range(n_steps):
        grid = ET.SubElement(
            collection, "Grid", Name=f"step_{step}", GridType="Uniform",
        )
        ET.SubElement(grid, "Time", Value=str(float(cloud.times[step])))
        ET.SubElement(
            grid, "Topology", TopologyType="Polyvertex",
            NumberOfElements=str(n),
        )
        geometry = ET.SubElement(grid, "Geometry", GeometryType=geom_type)
        coords_item = ET.SubElement(
            geometry, "DataItem", Dimensions=f"{n} {dim}",
            Format="HDF", NumberType="Float", Precision="8",
        )
        coords_item.text = f"{h5_name}:/coords"
        for name, arr in cloud.fields.items():
            ncomp = arr.shape[2]
            attribute = ET.SubElement(
                grid, "Attribute", Name=name,
                AttributeType=_attr_type(ncomp), Center="Node",
            )
            item = ET.SubElement(
                attribute, "DataItem", Dimensions=f"{n} {ncomp}",
                Format="HDF", NumberType="Float", Precision="8",
            )
            item.text = f"{h5_name}:/fields/{name}/{step:04d}"
    return ET.ElementTree(xdmf)
