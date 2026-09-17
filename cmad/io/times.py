"""Time-schedule loader shared by the FE and material point deck paths.

Both problem types let a deck spell a time schedule three ways, and both
want the same reader behind them:

- ``times: [t0, t1, ...]`` — an inline list.
- ``times file: <path>`` — a 1D array on disk; ``.npy`` via ``np.load``,
  ``.txt`` / ``.csv`` via ``np.loadtxt``.
- ``num steps`` + ``step size`` — an arithmetic sweep
  ``[0, dt, 2*dt, ...]`` including the initial time.

The FE side reads these out of ``discretization`` (see
:func:`cmad.cli.common._load_t_schedule`); the material point side reads
them out of ``deformation``, beside the deformation gradient history
(see :func:`cmad.io.deformation.load_times`). The section key names are
identical in both, so the branch dispatch lives here once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

TIME_KEYS = ("times", "times file", "num steps", "step size")
"""The deck keys this loader consumes, for presence checks by callers."""


def load_time_schedule(
        section: dict[str, Any],
        context: str = "",
) -> NDArray[np.float64]:
    """Materialize a 1D time schedule from a deck ``section``.

    Three branches mirror the schema ``oneOf`` that guards the section:
    inline ``times``, a ``times file`` on disk, or ``num steps`` +
    ``step size``. The result is always a raveled float64 array whose
    first entry is the initial time.

    ``context`` is the deck section name the keys were read from
    (``"discretization"`` / ``"deformation"``); it prefixes error
    messages so they name a path the user can find in their deck.
    """
    where = f"{context}." if context else ""
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
                f"{where}times file: unsupported extension '{suffix}' "
                f"for path {path}; expected .npy/.txt/.csv",
            )
        return np.asarray(data, dtype=np.float64).ravel()
    n = int(section["num steps"])
    dt = float(section["step size"])
    return np.arange(n + 1, dtype=np.float64) * dt
