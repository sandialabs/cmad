"""Step times from an input file section, ``times`` inline or ``times file``
on disk (``.npy``, ``.txt``, ``.csv``); the FE discretization section and
the material point deformation section read them the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def read_times(
        section: dict[str, Any], where: str,
) -> NDArray[np.float64] | None:
    """The times the section gives, or ``None`` when it gives neither
    entry; ``where`` names the section in errors.
    """
    if "times" in section and "times file" in section:
        raise ValueError(
            f"{where}: give either 'times' or 'times file', not both",
        )
    if "times" in section:
        return np.asarray(section["times"], dtype=np.float64).ravel()
    if "times file" in section:
        path = Path(section["times file"])
        suffix = path.suffix.lower()
        if suffix == ".npy":
            data = np.load(path)
        elif suffix in (".txt", ".csv"):
            data = np.loadtxt(path)
        else:
            raise ValueError(
                f"{where}.times file: unsupported extension '{suffix}' for "
                f"path {path}; expected .npy/.txt/.csv",
            )
        return np.asarray(data, dtype=np.float64).ravel()
    return None
