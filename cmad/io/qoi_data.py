"""QoI data loaders for the CMAD deck driver.

:func:`load_qoi_data` reads the ``data_file`` and the ``weight`` /
``weight_file`` fields of a material-point ``qoi:`` section and returns
the pair of float64 arrays that :meth:`cmad.qois.qoi.QoI.from_deck`
expects (``.npy`` only).

:func:`load_displacement_data` reads the per-step nodal displacement
field an FE displacement-matching QoI compares against (``.npy`` or a
``cmad primal`` Exodus output). :func:`load_reaction_data` reads the
per-step (per-component) load an FE load-matching QoI compares against
(``.npy`` / ``.csv`` / ``.txt``). :func:`load_roi` reads the mesh
entities a full-field measurement covers, which a field-matching QoI
integrates over instead of the whole domain. :func:`load_calibration_data`
reads a :class:`cmad.io.calibration_data.CalibrationData` archive, which
carries all three, and :func:`load_match_times` the times a matching QoI
scores.

No shape checks happen here; each QoI constructor asserts its own shape
contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cmad.io.calibration_data import CalibrationData
from cmad.io.exodus import read_results
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType


def load_roi(qoi_section: dict[str, Any], ndims: int) -> NDArray[np.intp]:
    """Return the mesh entities a full-field measurement covers.

    Reads ``qoi_section["roi_file"]``, an ``.npz`` written by the
    preprocessing that decides where the measurement is trustworthy. A 2D
    mesh is the measured surface itself, so its region of interest is
    ``elements``, global element indices of shape ``(n,)``. A 3D mesh is
    measured on its surface, so its region of interest is ``sides``,
    ``(elem_id, local_side_id)`` pairs of shape ``(n, 2)`` matching
    :attr:`cmad.fem.mesh.Mesh.side_sets`.

    The dimension picks the entry, so a region of interest built for one
    kind of mesh and handed to the other raises here rather than
    integrating over the wrong entities.
    """
    path = Path(qoi_section["roi_file"])
    if not path.exists():
        raise FileNotFoundError(f"qoi.roi_file: file not found at {path}")
    with np.load(path) as archive:
        return _roi_entry(archive, ndims, f"qoi.roi_file: {path}")


def calibration_data_roi(
        data: CalibrationData, ndims: int,
) -> NDArray[np.intp]:
    """Return the region of interest carried by ``data``, as :func:`load_roi`
    reads one from a file."""
    return _roi_entry(
        data.roi, ndims, f"calibration data for {data.mesh_file}",
    )


def _roi_entry(
        entries: Mapping[str, Any], ndims: int, where: str,
) -> NDArray[np.intp]:
    key = "elements" if ndims == 2 else "sides"
    expected_ndim = 1 if ndims == 2 else 2
    if key not in entries:
        raise ValueError(
            f"{where} has no {key!r} entry, which a {ndims}D mesh needs; "
            f"found {sorted(entries)}"
        )
    roi = np.asarray(entries[key], dtype=np.intp)
    if roi.ndim != expected_ndim:
        raise ValueError(
            f"{where} {key!r} has shape {tuple(roi.shape)}; a {ndims}D "
            f"mesh needs a {expected_ndim}D array"
        )
    return roi


def load_calibration_data(qoi_section: dict[str, Any]) -> CalibrationData:
    """Return the archive named by ``qoi_section["calibration_data_file"]``."""
    path = Path(qoi_section["calibration_data_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"qoi.calibration_data_file: file not found at {path}"
        )
    return CalibrationData.read(path)


def load_match_times(
        qoi_section: dict[str, Any], t_schedule: Sequence[float],
) -> NDArray[np.float64]:
    """Return the times a matching QoI scores.

    ``qoi_section["match_times_file"]`` when given (``.txt`` / ``.csv``
    via :func:`numpy.loadtxt`, ``.npy`` via :func:`numpy.load``), else the
    time schedule, which scores every step.
    """
    if "match_times_file" not in qoi_section:
        return np.asarray(t_schedule, dtype=np.float64).ravel()
    path = Path(qoi_section["match_times_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"qoi.match_times_file: file not found at {path}"
        )
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in {".csv", ".txt"}:
        arr = np.loadtxt(path)
    else:
        raise ValueError(
            f"qoi.match_times_file: unsupported extension '{ext}' "
            f"(path: {path}); supported: .npy, .csv, .txt",
        )
    return np.asarray(arr, dtype=np.float64).ravel()


def load_qoi_data(
        qoi_section: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(data, weight)`` arrays for the QoI in ``qoi_section``."""
    data = _load_npy("qoi.data_file", qoi_section["data_file"])
    weight = _load_weight(qoi_section)
    return data, weight


def load_displacement_data(
        qoi_section: dict[str, Any],
) -> NDArray[np.float64]:
    """Return the per-step nodal displacement target for a displacement QoI.

    Reads ``qoi_section["data_file"]`` as a ``(num_steps, num_nodes,
    ndims)`` array of nodal displacements — the layout
    :func:`cmad.io.exodus.read_results` produces for the nodal ``"u"``
    field. The extension dispatches the reader:

    - ``.npy`` -> :func:`numpy.load`.
    - ``.npz`` -> the ``"u"`` entry of the archive, or its sole entry
      when it holds exactly one. Compression matters here because the
      array is dense and grows as steps times nodes.
    - ``.exo`` / ``.ex2`` -> the nodal ``"u"`` field via
      :func:`cmad.io.exodus.read_results`, so a ``cmad primal`` Exodus
      output is itself valid calibration data with no conversion step.
    """
    path = Path(qoi_section["data_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"qoi.data_file: file not found at {path}",
        )
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        arr = _array_from_npz(path)
    elif ext in {".exo", ".ex2"}:
        results = read_results(
            path, nodal_field_specs=[FieldSpec("u", VarType.VECTOR)],
        )
        arr = results.nodal["u"]
    else:
        raise ValueError(
            f"qoi.data_file: unsupported extension '{ext}' "
            f"(path: {path}); supported: .npy, .npz, .exo, .ex2",
        )
    out: NDArray[np.float64] = np.asarray(arr, dtype=np.float64)
    return out


def _array_from_npz(path: Path) -> NDArray[np.float64]:
    """Pull the nodal field out of a ``.npz`` archive.

    Takes the ``"u"`` entry when present, else the sole entry. An archive
    holding several arrays without a ``"u"`` is ambiguous and raises
    naming what it found, rather than picking one.
    """
    with np.load(path) as archive:
        names = list(archive.files)
        if "u" in names:
            return np.asarray(archive["u"], dtype=np.float64)
        if len(names) == 1:
            return np.asarray(archive[names[0]], dtype=np.float64)
    raise ValueError(
        f"qoi.data_file: '{path}' holds {names} but no 'u' entry; name "
        f"the displacement array 'u' or store it alone",
    )


def load_reaction_data(
        qoi_section: dict[str, Any],
) -> NDArray[np.float64]:
    """Return the per-step measured-load target for a load-matching QoI.

    Reads ``qoi_section["data_file"]`` as a per-step load series aligned to
    the FE time schedule: shape ``(num_steps,)`` for a single component or
    ``(num_steps, num_components)``. ``.npy`` -> :func:`numpy.load`; ``.csv``
    / ``.txt`` -> :func:`numpy.loadtxt`.
    """
    path = Path(qoi_section["data_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"qoi.data_file: file not found at {path}",
        )
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in {".csv", ".txt"}:
        arr = np.loadtxt(path)
    else:
        raise ValueError(
            f"qoi.data_file: unsupported extension '{ext}' "
            f"(path: {path}); supported: .npy, .csv, .txt",
        )
    out: NDArray[np.float64] = np.asarray(arr, dtype=np.float64)
    return out


def _load_weight(
        qoi_section: dict[str, Any],
) -> NDArray[np.float64]:
    if "weight" in qoi_section:
        return np.asarray(qoi_section["weight"], dtype=np.float64)
    # Schema enforces oneOf(weight, weight_file); the else branch is
    # safe because the section reached the QoI loader post-validation.
    return _load_npy("qoi.weight_file", qoi_section["weight_file"])


def _load_npy(
        field_name: str, relpath: str,
) -> NDArray[np.float64]:
    path = Path(relpath)
    if not path.exists():
        raise FileNotFoundError(f"{field_name}: file not found at {path}")
    ext = path.suffix.lower()
    if ext != ".npy":
        raise ValueError(
            f"{field_name}: unsupported extension '{ext}' "
            f"(path: {path}); supported: .npy",
        )
    arr: NDArray[np.float64] = np.load(path).astype(np.float64)
    return arr
