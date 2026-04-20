"""Deformation-gradient history loader for the CMAD deck driver.

The primary public entry is :func:`load_history`, which accepts the deck's
``deformation:`` section along with the model's expected ``ndims`` and
returns a ``(ndims, ndims, num_steps + 1)`` float64 array whose spatial
dimensions match the model's ``DefType``:

- ``full_3d`` → ``(3, 3, N)``.
- ``plane_stress`` / ``plane_strain`` → ``(2, 2, N)``.
- ``uniaxial_stress`` / ``uniaxial_strain`` → ``(1, 1, N)``.

Two input modes are supported:

- ``history_file: <path>`` — a file on disk. The extension dispatches the
  reader; ``.npy`` is implemented today. ``.csv`` / ``.txt`` are a
  near-future additive branch (see ``_load_from_file``). File arrays are
  canonicalized from either ``(n, n, N)`` (preferred, matches the CMAD
  save convention) or ``(N, n, n)`` to ``(n, n, N)``. When ``N == n``
  the two layouts are indistinguishable; the loader treats the file as
  the preferred ``(n, n, N)``.
- ``inline: [[[...], ...], ...]`` — an inline list of n-by-n matrices for
  small test cases. The natural YAML reading is step-first
  ``(N, n, n)``; the loader always transposes to ``(n, n, N)``. No
  ambiguity at ``N == n``.

``expected_ndims`` is the model's ``ndims`` attribute (populated from
``def_type_ndims`` in every registered model's ``__init__``). Any shape
mismatch raises with the expected ``n`` and the loaded ``n`` both named,
before the array is handed to the primal or sensitivity loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def load_history(
        deformation_section: dict[str, Any],
        deck_dir: Path,
        expected_ndims: int,
) -> NDArray[np.float64]:
    """Load the deformation-gradient history into shape ``(n, n, N)``.

    ``expected_ndims`` comes from the model's ``ndims`` attribute; the
    loaded array's spatial dimensions must match.
    """
    if "history_file" in deformation_section:
        path = Path(deformation_section["history_file"])
        if not path.is_absolute():
            path = deck_dir / path
        arr = _load_from_file(path)
    elif "inline" in deformation_section:
        raw = np.asarray(deformation_section["inline"], dtype=np.float64)
        if raw.ndim != 3 or raw.shape[1] != raw.shape[2]:
            raise ValueError(
                f"deformation.inline: expected a list of n-by-n matrices "
                f"yielding shape (N, n, n); got {raw.shape}",
            )
        arr = np.ascontiguousarray(raw.transpose(1, 2, 0))
    else:
        raise ValueError(
            "deformation: must contain either 'history_file' or 'inline'",
        )
    _check_ndims(arr, expected_ndims)
    return arr


def _load_from_file(path: Path) -> NDArray[np.float64]:
    if not path.exists():
        raise FileNotFoundError(
            f"deformation.history_file: file not found at {path}",
        )
    ext = path.suffix.lower()
    if ext == ".npy":
        arr: NDArray[np.float64] = np.load(path).astype(np.float64)
    # elif ext in {".csv", ".txt"}:
    #     arr = np.loadtxt(path).reshape(-1, n, n).astype(np.float64)
    else:
        raise ValueError(
            f"deformation.history_file: unsupported extension '{ext}' "
            f"(path: {path}); supported: .npy",
        )
    return _canonicalize_file_shape(arr)


def _canonicalize_file_shape(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    # (n, n, N) is preferred and wins at N=n ambiguity; (N, n, n) is
    # accepted and transposed.
    if arr.ndim == 3 and arr.shape[0] == arr.shape[1]:
        return arr
    if arr.ndim == 3 and arr.shape[1] == arr.shape[2]:
        return np.ascontiguousarray(arr.transpose(1, 2, 0))
    raise ValueError(
        f"deformation: expected shape (n, n, N) or (N, n, n); got {arr.shape}",
    )


def _check_ndims(arr: NDArray[np.float64], expected_ndims: int) -> None:
    n = arr.shape[0]
    if n != expected_ndims:
        raise ValueError(
            f"deformation: shape (n, n, N) with n={n} does not match the "
            f"model's expected ndims={expected_ndims} "
            f"(full_3d→3, plane_stress/plane_strain→2, "
            f"uniaxial_stress/uniaxial_strain→1)",
        )
