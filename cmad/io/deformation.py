"""Deformation-gradient history loader for the CMAD deck driver.

The primary public entry is :func:`load_history`, which accepts the deck's
``deformation:`` section (a dict) and returns a ``(3, 3, num_steps + 1)``
float64 array. Two input modes are supported:

- ``history_file: <path>`` — a file on disk. The extension dispatches the
  reader; ``.npy`` is implemented today. ``.csv`` / ``.txt`` are a
  near-future additive branch (see ``_load_from_file``).
- ``inline: [[[...], ...], ...]`` — an inline list of 3x3 matrices for
  small test cases.

The returned array is always canonicalized to shape ``(3, 3, N)`` (the
convention used by the primal loop); inputs in ``(N, 3, 3)`` are
transposed.
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
        return _canonicalize_shape(arr)
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
    return _canonicalize_shape(arr)


def _canonicalize_shape(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    if arr.ndim == 3 and arr.shape[:2] == (3, 3):
        return arr
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return np.ascontiguousarray(arr.transpose(1, 2, 0))
    raise ValueError(
        f"deformation: expected shape (3, 3, N) or (N, 3, 3); got {arr.shape}",
    )
