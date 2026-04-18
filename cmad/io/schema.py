"""Schema validator for the CMAD deck driver.

Loads YAML-encoded JSON Schema fragments from ``cmad/io/schemas/``,
composes the subcommand-specific schema (stitching in the registered
model's fragment and, for subcommands that have a ``qoi`` section, the
registered QoI's fragment), and validates a parsed deck with aggregated
errors formatted as ``path: reason`` lines per the driver
error-reporting convention.

Callers do not need to eagerly import the concrete model / QoI
modules: lazy resolution happens in :mod:`cmad.io.registry`. The
``registered_*`` helpers used by the pre-flight checks here discover
names from the schema-fragment directories without triggering any
user-code imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from cmad.io.registry import registered_models, registered_qois

_SCHEMAS_DIR = Path(__file__).parent / "schemas"

_SUBCOMMAND_SECTIONS: dict[str, list[str]] = {
    "primal": ["problem", "model", "parameters",
               "deformation", "solver", "output"],
    "objective": ["problem", "model", "parameters",
                  "deformation", "qoi", "solver", "output"],
    "gradient": ["problem", "model", "parameters",
                 "deformation", "qoi", "sensitivity",
                 "solver", "output"],
    "hessian": ["problem", "model", "parameters",
                "deformation", "qoi", "sensitivity",
                "solver", "output"],
    "calibrate": ["problem", "model", "parameters",
                  "deformation", "qoi", "sensitivity",
                  "solver", "optimizer", "output"],
}


def validate_deck(deck: dict[str, Any], subcommand: str) -> None:
    """Validate the deck against the composed schema for ``subcommand``."""
    if subcommand not in _SUBCOMMAND_SECTIONS:
        raise ValueError(
            f"unknown subcommand '{subcommand}'; valid: "
            f"{sorted(_SUBCOMMAND_SECTIONS)}",
        )
    sections = _SUBCOMMAND_SECTIONS[subcommand]
    _check_model_registered(deck)
    qoi_name: str | None = None
    if "qoi" in sections:
        _check_qoi_registered(deck)
        qoi_name = deck["qoi"]["name"]
    composed = _compose_schema(subcommand, deck["model"]["name"], qoi_name)
    errors = list(Draft202012Validator(composed).iter_errors(deck))
    if errors:
        joined = "\n".join(_format_error(e) for e in errors)
        raise ValueError(f"deck validation failed:\n{joined}")


def _check_model_registered(deck: dict[str, Any]) -> None:
    model_section = deck.get("model")
    if not isinstance(model_section, dict) or "name" not in model_section:
        raise ValueError("model: missing 'name' field")
    known = registered_models()
    if model_section["name"] not in known:
        listing = ", ".join(known) if known else "(none)"
        raise ValueError(
            f"model.name: '{model_section['name']}' is not registered. "
            f"Registered model names: {listing}",
        )


def _check_qoi_registered(deck: dict[str, Any]) -> None:
    qoi_section = deck.get("qoi")
    if not isinstance(qoi_section, dict) or "name" not in qoi_section:
        raise ValueError("qoi: missing 'name' field")
    known = registered_qois()
    if qoi_section["name"] not in known:
        listing = ", ".join(known) if known else "(none)"
        raise ValueError(
            f"qoi.name: '{qoi_section['name']}' is not registered. "
            f"Registered qoi names: {listing}",
        )


def _compose_schema(
        subcommand: str, model_name: str, qoi_name: str | None = None,
) -> dict[str, Any]:
    sections = _SUBCOMMAND_SECTIONS[subcommand]
    properties: dict[str, Any] = {}
    merged_defs: dict[str, Any] = {}
    for section in sections:
        if section == "model":
            fragment = _load_fragment(f"models/{model_name}.yaml")
        elif section == "qoi":
            assert qoi_name is not None  # guaranteed by validate_deck
            fragment = _load_fragment(f"qois/{qoi_name}.yaml")
        else:
            fragment = _load_fragment(f"{section}.yaml")
        defs = fragment.pop("$defs", None)
        if defs is not None:
            for name, schema in defs.items():
                if name in merged_defs and merged_defs[name] != schema:
                    raise RuntimeError(
                        f"$defs collision on '{name}' while composing schema",
                    )
                merged_defs[name] = schema
        properties[section] = fragment
    composed: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": sections,
        "additionalProperties": False,
        "properties": properties,
    }
    if merged_defs:
        composed["$defs"] = merged_defs
    return composed


def _load_fragment(relative_path: str) -> dict[str, Any]:
    with (_SCHEMAS_DIR / relative_path).open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(
            f"schema fragment '{relative_path}' is empty or not a mapping",
        )
    return cast(dict[str, Any], data)


def _format_error(error: ValidationError) -> str:
    path = ".".join(str(p) for p in error.absolute_path) or "<root>"
    return f"{path}: {error.message}"
