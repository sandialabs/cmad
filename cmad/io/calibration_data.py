"""Measured data remapped onto a mesh, one row per usable frame.

A :class:`CalibrationData` holds every usable frame of a measurement on
the nodes a calibration reads: the nodes of the region of interest the
displacement match integrates over and the nodes of the sidesets the
Dirichlet conditions prescribe, with the measured load per frame. It is
one compressed ``.npz`` per mesh, written by :meth:`CalibrationData.write`
and read back by :meth:`CalibrationData.read`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_STORE_KEYS = frozenset({
    "times", "frame_ids", "load", "node_ids", "sideset_names",
    "sideset_offsets", "sideset_node_ids", "values", "mesh_file",
    "mesh_num_nodes",
})


@dataclass(frozen=True)
class CalibrationData:
    """Measured data at every usable frame on the nodes a calibration
    reads.

    ``times`` ``(F,)`` is strictly increasing and starts with the
    constructed reference; ``frame_ids`` ``(F,)`` is each row's index in
    the measurement record, ``-1`` for the reference; ``load`` ``(F,)``
    is the measured load per frame. ``node_ids`` ``(N,)`` is sorted
    ascending, the union of the region of interest's nodes and the
    Dirichlet sidesets' nodes, and is the node axis of ``values``
    ``(F, N, C)``. ``sidesets`` groups the Dirichlet sidesets' node ids
    by name, each group sorted, the order
    :func:`cmad.fem.dof.sideset_basis_fns` produces. ``mesh_file`` and
    ``mesh_num_nodes`` record the mesh it was built for. ``roi`` is the
    region of interest entries, written into the archive alongside the
    store's own.
    """

    times: NDArray[np.float64]
    frame_ids: NDArray[np.intp]
    load: NDArray[np.float64]
    node_ids: NDArray[np.intp]
    sidesets: dict[str, NDArray[np.intp]]
    values: NDArray[np.float64]
    mesh_file: str
    mesh_num_nodes: int
    roi: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        num_frames = int(self.times.shape[0]) if self.times.ndim == 1 else 0
        if num_frames == 0:
            raise ValueError("times must be a nonempty 1D array")
        if not np.all(np.diff(self.times) > 0.0):
            raise ValueError("times must be strictly increasing")
        for name in ("frame_ids", "load"):
            arr = getattr(self, name)
            if arr.shape[:1] != (num_frames,):
                raise ValueError(
                    f"{name} has shape {tuple(arr.shape)} but there are "
                    f"{num_frames} frames"
                )
        if (
            self.node_ids.ndim != 1 or self.node_ids.size == 0
            or not np.all(np.diff(self.node_ids) > 0)
        ):
            raise ValueError("node_ids must be nonempty, sorted, and unique")
        num_nodes = int(self.node_ids.shape[0])
        if (
            self.values.ndim != 3
            or self.values.shape[:2] != (num_frames, num_nodes)
        ):
            raise ValueError(
                f"values has shape {tuple(self.values.shape)}; expected "
                f"({num_frames}, {num_nodes}, num_components)"
            )
        for name, ids in self.sidesets.items():
            if ids.ndim != 1 or not np.all(np.diff(ids) > 0):
                raise ValueError(
                    f"sideset {name!r} node ids must be sorted and unique"
                )
            if not np.all(np.isin(ids, self.node_ids)):
                raise ValueError(
                    f"sideset {name!r} lists nodes outside node_ids"
                )
        clash = sorted(_STORE_KEYS & set(self.roi))
        if clash:
            raise ValueError(
                f"roi entries {clash} collide with the store's own keys"
            )

    @property
    def num_frames(self) -> int:
        return int(self.times.shape[0])

    @property
    def num_components(self) -> int:
        return int(self.values.shape[2])

    def nearest_frame(self, t: float) -> int:
        """Index of the frame closest to ``t``."""
        return int(np.argmin(np.abs(self.times - t)))

    def frame_at(self, t: float, rtol: float = 1.0e-8) -> int | None:
        """Index of the frame at ``t``, or ``None`` when no frame time is
        within ``rtol`` times the record span of it."""
        i = self.nearest_frame(t)
        if abs(float(self.times[i]) - t) <= self._tolerance(rtol):
            return i
        return None

    def frames_at(
            self,
            times: Sequence[float] | NDArray[np.float64],
            rtol: float = 1.0e-8,
    ) -> NDArray[np.intp]:
        """Index of the frame at each of ``times``; raises when one is
        not a frame."""
        frames = []
        for t in np.asarray(times, dtype=np.float64).ravel():
            i = self.frame_at(float(t), rtol)
            if i is None:
                raise ValueError(
                    f"time {t:g} is not a frame of the calibration data "
                    f"built for {self.mesh_file} (frames span "
                    f"{self.times[0]:g} to {self.times[-1]:g})"
                )
            frames.append(i)
        return np.asarray(frames, dtype=np.intp)

    def frames_within(
            self, t_lo: float, t_hi: float, rtol: float = 1.0e-8,
    ) -> NDArray[np.intp]:
        """Indices of the frames strictly between ``t_lo`` and ``t_hi``."""
        tol = self._tolerance(rtol)
        inside = (self.times > t_lo + tol) & (self.times < t_hi - tol)
        return np.flatnonzero(inside).astype(np.intp)

    def _tolerance(self, rtol: float) -> float:
        return rtol * float(self.times[-1] - self.times[0])

    def positions(
            self, node_ids: Sequence[int] | NDArray[np.intp],
    ) -> NDArray[np.intp]:
        """Indices into the node axis of ``node_ids``.

        Raises when any is absent, which is how a store built for another
        mesh, or covering less than a consumer reads, fails.
        """
        ids = np.asarray(node_ids, dtype=np.intp)
        pos = np.searchsorted(self.node_ids, ids)
        found = self.node_ids[np.minimum(pos, self.node_ids.size - 1)] == ids
        if not np.all(found):
            raise ValueError(
                f"node {int(ids[~found][0])} is not in the calibration data "
                f"built for {self.mesh_file} ({self.mesh_num_nodes} nodes)"
            )
        return pos.astype(np.intp)

    def check_mesh(self, num_nodes: int) -> None:
        """Raise unless the mesh in use has the node count this was built for."""
        if num_nodes != self.mesh_num_nodes:
            raise ValueError(
                f"calibration data built for {self.mesh_file} "
                f"({self.mesh_num_nodes} nodes) but the mesh has {num_nodes}"
            )

    def rows(
            self,
            frames: Sequence[int] | NDArray[np.intp],
            node_ids: Sequence[int] | NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """``values`` at ``frames`` on ``node_ids``, every node when
        ``None``; shaped ``(len(frames), n, num_components)``."""
        idx = np.atleast_1d(np.asarray(frames, dtype=np.intp))
        if node_ids is None:
            return np.asarray(self.values[idx], dtype=np.float64)
        pos = self.positions(node_ids)
        return np.asarray(
            self.values[idx[:, None], pos[None, :]], dtype=np.float64,
        )

    @classmethod
    def read(cls, path: str | Path) -> CalibrationData:
        """Load the archive :meth:`write` produced."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"calibration data: file not found at {path}"
            )
        with np.load(path) as archive:
            missing = sorted(_STORE_KEYS - set(archive.files))
            if missing:
                raise ValueError(
                    f"{path} is not a calibration data archive; missing "
                    f"{missing}"
                )
            names = [str(name) for name in archive["sideset_names"]]
            offsets = np.asarray(archive["sideset_offsets"], dtype=np.intp)
            grouped = np.asarray(archive["sideset_node_ids"], dtype=np.intp)
            sidesets = {
                name: grouped[offsets[i]:offsets[i + 1]]
                for i, name in enumerate(names)
            }
            roi = {
                key: archive[key] for key in archive.files
                if key not in _STORE_KEYS
            }
            return cls(
                times=np.asarray(archive["times"], dtype=np.float64),
                frame_ids=np.asarray(archive["frame_ids"], dtype=np.intp),
                load=np.asarray(archive["load"], dtype=np.float64),
                node_ids=np.asarray(archive["node_ids"], dtype=np.intp),
                sidesets=sidesets,
                values=np.asarray(archive["values"], dtype=np.float64),
                mesh_file=str(archive["mesh_file"]),
                mesh_num_nodes=int(archive["mesh_num_nodes"]),
                roi=roi,
            )

    def write(self, path: str | Path) -> None:
        """Write the data as one compressed ``.npz`` at ``path``."""
        names = list(self.sidesets)
        sizes = [int(self.sidesets[name].size) for name in names]
        grouped = (
            np.concatenate([self.sidesets[name] for name in names])
            if names else np.empty(0, dtype=np.intp)
        )
        entries: dict[str, Any] = {
            "times": self.times,
            "frame_ids": self.frame_ids,
            "load": self.load,
            "node_ids": self.node_ids,
            "sideset_names": np.array(names, dtype=str),
            "sideset_offsets": np.concatenate(
                [[0], np.cumsum(sizes)]).astype(np.intp),
            "sideset_node_ids": grouped.astype(np.intp),
            "values": self.values,
            "mesh_file": np.array(self.mesh_file),
            "mesh_num_nodes": np.array(self.mesh_num_nodes),
            **self.roi,
        }
        np.savez_compressed(Path(path), **entries)


def is_calibration_data(path: str | Path) -> bool:
    """Whether ``path`` is an archive :meth:`CalibrationData.write` wrote."""
    path = Path(path)
    if path.suffix.lower() != ".npz" or not path.exists():
        return False
    with np.load(path) as archive:
        return _STORE_KEYS.issubset(archive.files)
