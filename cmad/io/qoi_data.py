"""QoI data loaders for the CMAD deck driver.

:func:`load_qoi_data` reads the ``data_file`` and the ``weight`` /
``weight_file`` fields of a material-point ``qoi:`` section and returns
the pair of float64 arrays that :meth:`cmad.qois.qoi.QoI.from_deck`
expects (``.npy`` only).

:func:`load_displacement_data` reads the per-step nodal displacement
field an FE displacement-matching QoI compares against (``.npy`` or a
``cmad primal`` Exodus output). :func:`load_reaction_data` reads the
per-step (per-component) load an FE load-matching QoI compares against
(``.npy`` / ``.csv`` / ``.txt``).

No shape checks happen here; each QoI constructor asserts its own shape
contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cmad.io.exodus import read_results
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType


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
