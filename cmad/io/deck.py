"""YAML deck loader and defaults filler for the CMAD driver subcommands.

``load_deck`` reads a YAML file and returns its contents as a plain dict
tree. Schema validation happens separately (``cmad/io/schema.py``); this
module is pure parsing plus a minimal top-level structure check.

``apply_deck_defaults`` runs three normalization passes and fills schema
defaults so a minimal deck (no ``solver:`` / ``residuals.global
residual.nonlinear*``, no ``output.format`` field, etc.) validates
cleanly and so ``deck.resolved.yaml`` reflects the values actually used.
The three passes are:

1. **Top-level wrapper auto-unwrap** via :func:`unwrap_top_level`.
   Calibr8 decks open with a single problem-name key whose value is the
   deck body (``cube_elastic: {problem: ..., ...}``); cmad accepts both
   wrapped and flat forms.
2. **Calibr8-only section strip** via :func:`strip_calibr8_only`.
   ``linear algebra:`` and ``regression:`` come along for the ride on
   Calibr8-portable decks but cmad has no use for them; both are popped
   with a one-line ``UserWarning``.
3. **Default-filling** dispatched on ``problem.type`` (MP fills
   ``solver.newton``; FE fills ``residuals.{global, local}
   residual.nonlinear*`` and ``output.format: exodus``). The optimizer
   defaults are problem-type-agnostic.

Both normalization helpers are also re-imported by ``cmad/io/schema.py``
so that ``validate_deck`` can be called defensively on a not-yet-
normalized deck (idempotent: a deck already unwrapped/stripped passes
through unchanged).
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any, cast

import yaml

_SOLVER_DEFAULTS: dict[str, dict[str, Any]] = {
    "newton": {
        "max_iters": 10,
        "abs_tol": 1e-14,
        "rel_tol": 1e-14,
        "max_ls_evals": 0,
    },
}
_OUTPUT_DEFAULTS: dict[str, Any] = {"prefix": "", "format": "npy"}
_OPTIMIZER_DEFAULTS: dict[str, Any] = {
    "initial_guess": "from_deck",
    "options": {},
    "log_params": True,
}
_FE_RESIDUALS_DEFAULTS: dict[str, dict[str, Any]] = {
    "global residual": {
        "nonlinear max iters": 10,
        "nonlinear absolute tol": 1.0e-12,
        "nonlinear relative tol": 1.0e-12,
        "print convergence": False,
    },
    "local residual": {
        "nonlinear max iters": 30,
        "nonlinear absolute tol": 1.0e-14,
        "nonlinear relative tol": 1.0e-14,
    },
}

_CALIBR8_ONLY_SECTIONS: tuple[str, ...] = ("linear algebra", "regression")


def load_deck(path: Path) -> dict[str, Any]:
    """Load and parse the YAML deck at ``path``.

    Returns the parsed YAML as a dict tree. Does not validate the deck's
    schema; call ``cmad.io.schema.validate_deck`` afterward.
    """
    if not path.exists():
        raise FileNotFoundError(f"deck not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"deck is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(
            f"deck top-level must be a mapping; got {type(data).__name__} "
            f"at {path}",
        )
    return cast(dict[str, Any], data)


def unwrap_top_level(deck: dict[str, Any]) -> dict[str, Any]:
    """Auto-unwrap a single-key top-level wrapper if present.

    A wrapped deck has exactly one top-level key whose value is a
    mapping containing ``problem``; that's the pattern Calibr8 decks
    use (``cube_elastic: {problem: ..., ...}``). Returns the original
    deck unchanged when no wrapper is detected. Idempotent.
    """
    if len(deck) == 1:
        only_key = next(iter(deck))
        only_val = deck[only_key]
        if isinstance(only_val, dict) and "problem" in only_val:
            return cast(dict[str, Any], only_val)
    return deck


def strip_calibr8_only(deck: dict[str, Any]) -> dict[str, Any]:
    """Pop Calibr8-only sections with a one-line warning.

    ``linear algebra:`` and ``regression:`` come along for the ride on
    Calibr8-portable decks; cmad has no use for them. Stripping them
    here lets the deck validate cleanly under cmad's
    ``additionalProperties: false`` schema. Returns the original deck
    unchanged when no Calibr8-only sections are present (no warning
    emitted in that case). Idempotent.
    """
    stripped: dict[str, Any] | None = None
    for section in _CALIBR8_ONLY_SECTIONS:
        if section in deck:
            if stripped is None:
                stripped = dict(deck)
            del stripped[section]
            warnings.warn(
                f"deck section '{section}' is recognized but unused by "
                "cmad (Calibr8-only); ignored",
                UserWarning,
                stacklevel=3,
            )
    return stripped if stripped is not None else deck


def apply_deck_defaults(deck: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copy of ``deck`` normalized and with defaults merged in.

    See module docstring for the three normalization passes. jsonschema's
    ``default`` keyword is advisory; defaults are filled here manually
    so the post-resolution deck written to disk reflects the values
    actually used.
    """
    resolved = copy.deepcopy(deck)
    resolved = unwrap_top_level(resolved)
    resolved = strip_calibr8_only(resolved)

    problem_type = resolved.get("problem", {}).get("type")

    if problem_type == "material_point":
        newton_in = resolved.setdefault("solver", {}).setdefault("newton", {})
        for k, v in _SOLVER_DEFAULTS["newton"].items():
            newton_in.setdefault(k, v)

    if problem_type == "fe":
        residuals = resolved.setdefault("residuals", {})
        for slot, defaults in _FE_RESIDUALS_DEFAULTS.items():
            slot_dict = residuals.setdefault(slot, {})
            for k, v in defaults.items():
                slot_dict.setdefault(k, v)

    output_defaults = dict(_OUTPUT_DEFAULTS)
    if problem_type == "fe":
        output_defaults["format"] = "exodus"
    output_in = resolved.setdefault("output", {})
    for k, v in output_defaults.items():
        output_in.setdefault(k, v)

    if "optimizer" in resolved:
        optimizer_in = resolved["optimizer"]
        for k, v in _OPTIMIZER_DEFAULTS.items():
            optimizer_in.setdefault(k, v)

    return resolved
