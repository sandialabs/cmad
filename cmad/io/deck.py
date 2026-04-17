"""YAML deck loader for CMAD driver subcommands.

``load_deck`` reads a YAML file and returns its contents as a plain dict
tree. Schema validation happens separately (``cmad/io/schema.py``); this
module is pure parsing plus a minimal top-level structure check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


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
