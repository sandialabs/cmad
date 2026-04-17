"""Deformation-gradient history loader for the CMAD deck driver.

The primary public entry is :func:`load_history`, which accepts the deck's
``deformation:`` section (a dict) and returns a ``(3, 3, num_steps + 1)``
float64 array. Two input modes are supported:

- ``history_file: <path>`` — a file on disk. The extension dispatches the
  reader; ``.npy`` is implemented today. ``.csv`` / ``.txt`` are a
  near-future additive branch (see ``_load_from_file``). File arrays are
  canonicalized from either ``(3, 3, N)`` (preferred, matches the CMAD
  save convention) or ``(N, 3, 3)`` to ``(3, 3, N)``. When ``N == 3``
  the two layouts are indistinguishable; the loader treats the file as
  the preferred ``(3, 3, N)``.
- ``inline: [[[...], ...], ...]`` — an inline list of 3x3 matrices for
  small test cases. The natural YAML reading is step-first
  ``(N, 3, 3)``; the loader always transposes to ``(3, 3, N)``. No
  ambiguity at ``N == 3``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def load_history(
        deformation_section: dict[str, Any],
        deck_dir: Path,
) -> NDArray[np.float64]:
    """Load the deformation-gradient history into shape ``(3, 3, N)``."""
    if "history_file" in deformation_section:
        path = Path(deformation_section["history_file"])
        if not path.is_absolute():
            path = deck_dir / path
        return _load_from_file(path)
    if "inline" in deformation_section:
        arr = np.asarray(deformation_section["inline"], dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1:] != (3, 3):
            raise ValueError(
                f"deformation.inline: expected a list of 3x3 matrices "
                f"yielding shape (N, 3, 3); got {arr.shape}",
            )
        return np.ascontiguousarray(arr.transpose(1, 2, 0))
    raise ValueError(
        "deformation: must contain either 'history_file' or 'inline'",
    )


def _load_from_file(path: Path) -> NDArray[np.float64]:
    if not path.exists():
        raise FileNotFoundError(
            f"deformation.history_file: file not found at {path}",
        )
    ext = path.suffix.lower()
    if ext == ".npy":
        arr: NDArray[np.float64] = np.load(path).astype(np.float64)
    # elif ext in {".csv", ".txt"}:
    #     arr = np.loadtxt(path).reshape(-1, 3, 3).astype(np.float64)
    else:
        raise ValueError(
            f"deformation.history_file: unsupported extension '{ext}' "
            f"(path: {path}); supported: .npy",
        )
    return _canonicalize_file_shape(arr)


def _canonicalize_file_shape(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    # (3, 3, N) is preferred and wins at N=3 ambiguity; (N, 3, 3) is
    # accepted and transposed.
    if arr.ndim == 3 and arr.shape[:2] == (3, 3):
        return arr
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return np.ascontiguousarray(arr.transpose(1, 2, 0))
    raise ValueError(
        f"deformation: expected shape (3, 3, N) or (N, 3, 3); got {arr.shape}",
    )
