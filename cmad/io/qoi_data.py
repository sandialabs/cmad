"""QoI data / weight loader for the CMAD deck driver.

:func:`load_qoi_data` reads the ``data_file`` and the ``weight`` /
``weight_file`` fields of a deck's ``qoi:`` section and returns the pair
of float64 arrays that :meth:`cmad.qois.qoi.QoI.from_deck` expects.

Path dispatch (relative-to-``deck_dir``, absolute pass-through) and
``.npy``-only extension handling mirror :mod:`cmad.io.deformation`. No
shape checks happen here; the QoI constructor asserts its own shape
contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def load_qoi_data(
        qoi_section: dict[str, Any], deck_dir: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(data, weight)`` arrays for the QoI in ``qoi_section``."""
    data = _load_npy("qoi.data_file", qoi_section["data_file"], deck_dir)
    weight = _load_weight(qoi_section, deck_dir)
    return data, weight


def _load_weight(
        qoi_section: dict[str, Any], deck_dir: Path,
) -> NDArray[np.float64]:
    if "weight" in qoi_section:
        return np.asarray(qoi_section["weight"], dtype=np.float64)
    # Schema enforces oneOf(weight, weight_file); the else branch is
    # safe because the section reached the QoI loader post-validation.
    return _load_npy(
        "qoi.weight_file", qoi_section["weight_file"], deck_dir,
    )


def _load_npy(
        field_name: str, relpath: str, deck_dir: Path,
) -> NDArray[np.float64]:
    path = Path(relpath)
    if not path.is_absolute():
        path = deck_dir / path
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
