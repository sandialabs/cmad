"""Output writers for the CMAD deck driver.

The ``cmad primal`` subcommand's four outputs:

- ``cauchy.{npy, csv}`` — the Cauchy stress trajectory, shape
  ``(3, 3, N+1)``.
- ``xi_block_<k>.{npy, csv}`` — per-residual-block state-variable
  trajectories, one file per block, shape ``(N+1, num_eqs_in_block)``.
- ``solver.json`` — per-step Newton diagnostics (always JSON).
- ``deck.resolved.yaml`` — the parsed-and-defaulted deck (always YAML).

Array outputs dispatch on ``output.format`` (``npy`` or ``text``);
structured outputs ignore it.

**F1+ extension point.** When ``GlobalResidual`` lands, it will have its
own block structure (displacement, pressure, possibly more). A sibling
``write_global_fields`` (or similarly named) should mirror :func:`write_xi`'s
per-block file convention so FE-side trajectories follow the same naming
pattern and read-back idiom.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

_CAUCHY_HEADER = "S11 S12 S13 S21 S22 S23 S31 S32 S33"


def write_cauchy(
        out_dir: Path,
        prefix: str,
        cauchy: NDArray[np.floating],
        fmt: str,
) -> None:
    """Write the ``(3, 3, N+1)`` Cauchy trajectory."""
    if fmt == "npy":
        np.save(out_dir / f"{prefix}cauchy.npy", cauchy)
    elif fmt == "text":
        flat = cauchy.transpose(2, 0, 1).reshape(-1, 9)
        np.savetxt(
            out_dir / f"{prefix}cauchy.csv",
            flat,
            header=_CAUCHY_HEADER,
        )
    else:
        raise ValueError(
            f"output.format: expected 'npy' or 'text', got {fmt!r}",
        )


def write_xi(
        out_dir: Path,
        prefix: str,
        xi_trajectory: list[list[NDArray[np.floating]]],
        fmt: str,
) -> None:
    """Write per-residual-block state-variable trajectories.

    ``xi_trajectory[step_idx][block_idx]`` is a 1-D array of length
    ``num_eqs_in_block``. One file per block, shape
    ``(N+1, num_eqs_in_block)``.
    """
    if fmt not in {"npy", "text"}:
        raise ValueError(
            f"output.format: expected 'npy' or 'text', got {fmt!r}",
        )
    if not xi_trajectory:
        return
    num_blocks = len(xi_trajectory[0])
    for k in range(num_blocks):
        per_step = np.stack(
            [xi_trajectory[t][k] for t in range(len(xi_trajectory))],
        )
        if fmt == "npy":
            np.save(out_dir / f"{prefix}xi_block_{k:02d}.npy", per_step)
        else:
            np.savetxt(out_dir / f"{prefix}xi_block_{k:02d}.csv", per_step)


def write_solver_log(
        out_dir: Path,
        prefix: str,
        solver_log: list[dict[str, Any]],
) -> None:
    """Write per-step Newton diagnostics as JSON."""
    with (out_dir / f"{prefix}solver.json").open("w") as f:
        json.dump(solver_log, f, indent=2)


def write_J(out_dir: Path, prefix: str, J: float) -> None:
    """Write the scalar QoI value as JSON."""
    with (out_dir / f"{prefix}J.json").open("w") as f:
        json.dump({"J": J}, f, indent=2)


def write_grad(
        out_dir: Path,
        prefix: str,
        grad: NDArray[np.floating],
        fmt: str,
) -> None:
    """Write the gradient vector of shape ``(num_active_params,)``.

    Canonical-coordinate values; the parameter transforms in the
    resolved deck carry the mapping back to raw coordinates.
    """
    if fmt == "npy":
        np.save(out_dir / f"{prefix}grad.npy", grad)
    elif fmt == "text":
        np.savetxt(out_dir / f"{prefix}grad.csv", grad)
    else:
        raise ValueError(
            f"output.format: expected 'npy' or 'text', got {fmt!r}",
        )


def write_hessian(
        out_dir: Path,
        prefix: str,
        hessian: NDArray[np.floating],
        fmt: str,
) -> None:
    """Write the Hessian matrix of shape ``(num_active_params, num_active_params)``.

    Canonical-coordinate values; the parameter transforms in the
    resolved deck carry the mapping back to raw coordinates.
    """
    if fmt == "npy":
        np.save(out_dir / f"{prefix}hess.npy", hessian)
    elif fmt == "text":
        np.savetxt(out_dir / f"{prefix}hess.csv", hessian)
    else:
        raise ValueError(
            f"output.format: expected 'npy' or 'text', got {fmt!r}",
        )


def write_resolved_deck(
        out_dir: Path,
        prefix: str,
        resolved_deck: dict[str, Any],
) -> None:
    """Write the parsed-and-defaulted deck as YAML.

    Assumes ``resolved_deck`` contains only YAML-native types; numpy
    arrays or jax types will cause :func:`yaml.safe_dump` to raise.
    """
    with (out_dir / f"{prefix}deck.resolved.yaml").open("w") as f:
        yaml.safe_dump(
            resolved_deck, f, default_flow_style=False, sort_keys=False,
        )
