"""YAML deck loader and defaults filler for the CMAD driver subcommands.

``load_deck`` reads a YAML file and returns its contents as a plain dict
tree. Schema validation happens separately (``cmad/io/schema.py``); this
module is pure parsing plus a minimal top-level structure check.

``apply_deck_defaults`` fills in the solver-Newton and output defaults
before the schema validator runs, so a minimal deck (no ``solver:``
section, no ``output.format`` field, etc.) validates cleanly, and so
``deck.resolved.yaml`` reflects the values actually used.
"""

from __future__ import annotations

import copy
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


def apply_deck_defaults(deck: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copy of ``deck`` with schema defaults merged in.

    jsonschema's ``default`` keyword is advisory; this helper applies
    the Newton and output defaults manually so required-keys checks
    pass on a minimal deck and so the post-resolution deck written to
    disk reflects the values actually used.
    """
    resolved = copy.deepcopy(deck)
    newton_in = resolved.setdefault("solver", {}).setdefault("newton", {})
    for k, v in _SOLVER_DEFAULTS["newton"].items():
        newton_in.setdefault(k, v)
    output_in = resolved.setdefault("output", {})
    for k, v in _OUTPUT_DEFAULTS.items():
        output_in.setdefault(k, v)
    return resolved
