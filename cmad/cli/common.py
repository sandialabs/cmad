"""Shared helpers for CMAD material-point subcommand orchestrators.

The material-point subcommands share a deck-load / defaults / schema /
model / parameters / deformation-history / (optional) QoI construction
prelude and an output-location resolution tail. This module factors
both into reusable helpers so each orchestrator stays focused on the
numerical operation it dispatches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cmad.io.deck import apply_deck_defaults, load_deck
from cmad.io.deformation import load_history
from cmad.io.params_builder import build_parameters
from cmad.io.qoi_data import load_qoi_data
from cmad.io.registry import resolve_model, resolve_qoi
from cmad.io.schema import validate_deck
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.qois.qoi import QoI


@dataclass(frozen=True)
class MPObjectGraph:
    resolved: dict[str, Any]
    parameters: Parameters
    model: Model
    F: NDArray[np.float64]
    qoi: QoI | None


def build_mp_object_graph(
        deck_path: Path, subcommand: str,
) -> MPObjectGraph:
    """Build the material-point object graph shared by all subcommands.

    Runs deck load + defaults + schema validation, resolves the
    registered model and (for all subcommands except ``primal``) QoI,
    builds parameters, and loads the deformation history. The returned
    graph's ``qoi`` is ``None`` iff ``subcommand == "primal"``.
    """
    deck = load_deck(deck_path)
    resolved = apply_deck_defaults(deck)
    validate_deck(resolved, subcommand)

    model_cls = resolve_model(resolved["model"]["name"])
    parameters = build_parameters(resolved["parameters"])
    model = model_cls.from_deck(resolved["model"], parameters)

    F = load_history(
        resolved["deformation"], deck_path.parent,
        expected_ndims=model.ndims,
    )

    qoi: QoI | None = None
    if subcommand != "primal":
        qoi_cls = resolve_qoi(resolved["qoi"]["name"])
        data, weight = load_qoi_data(resolved["qoi"], deck_path.parent)
        qoi = qoi_cls.from_deck(resolved["qoi"], model, data, weight)

    return MPObjectGraph(
        resolved=resolved, parameters=parameters,
        model=model, F=F, qoi=qoi,
    )


def resolve_output(
        resolved: dict[str, Any], deck_path: Path,
) -> tuple[Path, str, str]:
    """Resolve ``out_dir``, ``prefix``, and ``format`` from a validated deck.

    The path is taken relative to ``deck_path.parent`` when not absolute
    and created. ``format`` is always present (schema default ``npy``);
    callers that don't emit array outputs can discard it.
    """
    out_dir = Path(resolved["output"]["path"])
    if not out_dir.is_absolute():
        out_dir = deck_path.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, resolved["output"]["prefix"], resolved["output"]["format"]
