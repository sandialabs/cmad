"""Deformation-gradient (and time) history loader for the CMAD deck driver.

The primary public entry is :func:`load_history`, which accepts the deck's
``deformation:`` section along with the model's expected ``ndims`` and
returns a ``(ndims, ndims, num_steps + 1)`` float64 array whose spatial
dimensions match the model's ``DefType``:

- ``full_3d`` → ``(3, 3, N)``.
- ``plane_stress`` / ``plane_strain`` → ``(2, 2, N)``.
- ``uniaxial_stress`` / ``uniaxial_strain`` → ``(1, 1, N)``.

Two input modes are supported:

- ``history_file: <path>`` — a file on disk. The extension dispatches the
  reader; ``.npy``, ``.csv``, and ``.txt`` are supported. ``.npy`` arrays
  are canonicalized from either ``(n, n, N)`` (preferred, matches the
  CMAD save convention) or ``(N, n, n)`` to ``(n, n, N)``; when
  ``N == n`` the two layouts are indistinguishable and the loader treats
  the file as the preferred ``(n, n, N)``. ``.csv`` / ``.txt`` files
  contain one row per step with a flattened row-major n-by-n matrix
  (``n*n`` columns per row); ``.csv`` is comma-delimited, ``.txt`` is
  whitespace-delimited. ``n`` is inferred from the column count and
  validated as a perfect square. Text files are always ``(N, n, n)`` —
  no N==n ambiguity.
- ``inline: [[[...], ...], ...]`` — an inline list of n-by-n matrices for
  small test cases. The natural YAML reading is step-first
  ``(N, n, n)``; the loader always transposes to ``(n, n, N)``. No
  ambiguity at ``N == n``.

``expected_ndims`` is the model's ``ndims`` attribute (populated from
``def_type_ndims`` in every registered model's ``__init__``). Any shape
mismatch raises with the expected ``n`` and the loaded ``n`` both named,
before the array is handed to the primal or sensitivity loop.

The section may also carry a time history, read by :func:`load_times`,
spelled the three ways the FE ``discretization`` section spells its own
(``times``, ``times file``, or ``num steps`` + ``step size``; see
:mod:`cmad.io.times`). It lives here rather than in a section of its own
because its length is tied to the deformation history's: one time per
column of ``F``. Time is optional — a deck that omits it drives every
step at ``dt = 1``, which is all a rate independent model needs. A model
whose flow is rate dependent does need real step sizes, and the deck
builder refuses to run one without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cmad.io.times import TIME_KEYS, load_time_schedule


def load_history(
        deformation_section: dict[str, Any],
        expected_ndims: int,
) -> NDArray[np.float64]:
    """Load the deformation-gradient history into shape ``(n, n, N)``.

    ``expected_ndims`` comes from the model's ``ndims`` attribute; the
    loaded array's spatial dimensions must match.
    """
    if "history_file" in deformation_section:
        path = Path(deformation_section["history_file"])
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


def load_times(
        deformation_section: dict[str, Any],
        num_steps: int,
) -> NDArray[np.float64] | None:
    """Load the step times, or ``None`` when the section carries no time.

    ``num_steps`` is ``F.shape[2] - 1``; the schedule must hold one time
    per column of ``F``, i.e. ``num_steps + 1`` entries. Times must
    increase strictly: a viscoplastic flow rule divides by ``dt``
    (:mod:`cmad.models.rate_dependence`), so a zero or backwards step is
    an error here rather than a divide by zero inside a traced residual.
    """
    if not any(key in deformation_section for key in TIME_KEYS):
        return None

    times = load_time_schedule(deformation_section, context="deformation")

    if times.size != num_steps + 1:
        raise ValueError(
            f"deformation: the time history holds {times.size} times but "
            f"the deformation gradient history holds {num_steps + 1} steps "
            f"(shape (n, n, {num_steps + 1})); one time per column of F "
            f"is required",
        )
    if times.size > 1 and not np.all(np.diff(times) > 0.):
        bad = int(np.argmin(np.diff(times) > 0.))
        raise ValueError(
            f"deformation: the time history must increase strictly; "
            f"times[{bad}]={times[bad]:g} is not less than "
            f"times[{bad + 1}]={times[bad + 1]:g}",
        )
    return times


def _load_from_file(path: Path) -> NDArray[np.float64]:
    if not path.exists():
        raise FileNotFoundError(
            f"deformation.history_file: file not found at {path}",
        )
    ext = path.suffix.lower()
    if ext == ".npy":
        arr: NDArray[np.float64] = np.load(path).astype(np.float64)
    elif ext in {".csv", ".txt"}:
        delimiter = "," if ext == ".csv" else None
        raw = np.loadtxt(path, delimiter=delimiter, ndmin=2).astype(np.float64)
        cols = raw.shape[1]
        n = int(np.sqrt(cols))
        if n * n != cols:
            raise ValueError(
                f"deformation.history_file: expected n*n columns per row "
                f"(flattened n-by-n matrix); got {cols} columns in {path}",
            )
        arr = raw.reshape(raw.shape[0], n, n)
    else:
        raise ValueError(
            f"deformation.history_file: unsupported extension '{ext}' "
            f"(path: {path}); supported: .npy, .csv, .txt",
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
