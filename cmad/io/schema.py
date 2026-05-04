"""Schema validator for the CMAD deck driver.

Loads YAML-encoded JSON Schema fragments from ``cmad/io/schemas/``,
composes the (problem-type, subcommand)-specific schema (stitching in
the registered model's fragment for MP decks and the registered QoI's
fragment when the subcommand uses one), and validates a parsed deck
with aggregated errors formatted as ``path: reason`` lines per the
driver error-reporting convention.

Section names with spaces (FE convention: ``dirichlet bcs``,
``surface flux bcs``, ``body forces``) map to underscored fragment
filenames (``dirichlet_bcs.yaml``, ...) at fragment-load time.

Two normalization passes run before validation, mirroring the same
helpers in :mod:`cmad.io.deck`: optional top-level wrapper auto-unwrap
and Calibr8-only-section strip. Both are idempotent so calling
``validate_deck`` directly on a not-yet-normalized deck produces the
same result as calling ``apply_deck_defaults`` first.

Callers do not need to eagerly import the concrete model / QoI /
global-residual modules: lazy resolution happens in
:mod:`cmad.io.registry`. The ``registered_*`` helpers used by the
pre-flight checks here discover names from the schema-fragment
directories without triggering any user-code imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from cmad.io.deck import unwrap_top_level, strip_calibr8_only
from cmad.io.registry import (
    registered_global_residuals,
    registered_models,
    registered_qois,
)

_SCHEMAS_DIR = Path(__file__).parent / "schemas"

# (problem_type, subcommand) -> (required_sections, optional_sections)
_SECTIONS: dict[tuple[str, str], tuple[list[str], list[str]]] = {
    ("material_point", "primal"): (
        ["problem", "model", "parameters",
         "deformation", "solver", "output"],
        [],
    ),
    ("material_point", "objective"): (
        ["problem", "model", "parameters",
         "deformation", "qoi", "solver", "output"],
        [],
    ),
    ("material_point", "gradient"): (
        ["problem", "model", "parameters",
         "deformation", "qoi", "sensitivity",
         "solver", "output"],
        [],
    ),
    ("material_point", "hessian"): (
        ["problem", "model", "parameters",
         "deformation", "qoi", "sensitivity",
         "solver", "output"],
        [],
    ),
    ("material_point", "calibrate"): (
        ["problem", "model", "parameters",
         "deformation", "qoi", "sensitivity",
         "solver", "optimizer", "output"],
        [],
    ),
    ("fe", "primal"): (
        ["problem", "discretization", "residuals", "output"],
        ["dirichlet bcs", "surface flux bcs", "body forces"],
    ),
}


def validate_deck(deck: dict[str, Any], subcommand: str) -> None:
    """Validate the deck against the composed schema for ``subcommand``."""
    deck = unwrap_top_level(deck)
    deck = strip_calibr8_only(deck)

    problem_section = deck.get("problem")
    if not isinstance(problem_section, dict) or "type" not in problem_section:
        raise ValueError("problem: missing 'type' field")
    problem_type = problem_section["type"]

    if (problem_type, subcommand) not in _SECTIONS:
        valid = sorted(_SECTIONS)
        raise ValueError(
            f"unknown (problem.type, subcommand) pair "
            f"('{problem_type}', '{subcommand}'); valid pairs: {valid}",
        )

    required, optional = _SECTIONS[(problem_type, subcommand)]
    all_sections = required + optional

    _check_model_registered(deck, problem_type)
    if problem_type == "fe":
        _check_global_residual_registered(deck)
    qoi_name: str | None = None
    if "qoi" in all_sections:
        _check_qoi_registered(deck)
        qoi_name = deck["qoi"]["name"]

    model_name: str | None = None
    if problem_type == "material_point":
        model_name = deck["model"]["name"]

    composed = _compose_schema(
        problem_type, subcommand,
        model_name=model_name, qoi_name=qoi_name,
    )
    errors = list(Draft202012Validator(composed).iter_errors(deck))
    if errors:
        joined = "\n".join(_format_error(e) for e in errors)
        raise ValueError(f"deck validation failed:\n{joined}")


def _check_model_registered(
        deck: dict[str, Any], problem_type: str,
) -> None:
    """Verify the deck's model name is in the registry.

    For MP decks the name lives at ``deck.model.name``; for FE decks
    it lives at ``deck.residuals.local residual.type``.
    """
    if problem_type == "material_point":
        section = deck.get("model")
        if not isinstance(section, dict) or "name" not in section:
            raise ValueError("model: missing 'name' field")
        name = section["name"]
        path = "model.name"
    elif problem_type == "fe":
        residuals = deck.get("residuals")
        if not isinstance(residuals, dict):
            raise ValueError("residuals: missing or not a mapping")
        local = residuals.get("local residual")
        if not isinstance(local, dict) or "type" not in local:
            raise ValueError(
                "residuals.local residual: missing 'type' field",
            )
        name = local["type"]
        path = "residuals.local residual.type"
    else:
        return

    known = registered_models()
    if name not in known:
        listing = ", ".join(known) if known else "(none)"
        raise ValueError(
            f"{path}: '{name}' is not registered. "
            f"Registered model names: {listing}",
        )


def _check_global_residual_registered(deck: dict[str, Any]) -> None:
    """Verify the FE deck's GR name is in the registry."""
    residuals = deck.get("residuals")
    if not isinstance(residuals, dict):
        raise ValueError("residuals: missing or not a mapping")
    glob = residuals.get("global residual")
    if not isinstance(glob, dict) or "type" not in glob:
        raise ValueError("residuals.global residual: missing 'type' field")
    name = glob["type"]
    known = registered_global_residuals()
    if name not in known:
        listing = ", ".join(known) if known else "(none)"
        raise ValueError(
            f"residuals.global residual.type: '{name}' is not registered. "
            f"Registered global residual names: {listing}",
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
        problem_type: str,
        subcommand: str,
        model_name: str | None = None,
        qoi_name: str | None = None,
) -> dict[str, Any]:
    required, optional = _SECTIONS[(problem_type, subcommand)]
    all_sections = required + optional
    properties: dict[str, Any] = {}
    merged_defs: dict[str, Any] = {}
    for section in all_sections:
        if section == "model":
            assert model_name is not None  # guaranteed by validate_deck
            fragment = _load_fragment(f"models/{model_name}.yaml")
        elif section == "qoi":
            assert qoi_name is not None  # guaranteed by validate_deck
            fragment = _load_fragment(f"qois/{qoi_name}.yaml")
        else:
            filename = section.replace(" ", "_") + ".yaml"
            fragment = _load_fragment(filename)
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
        "required": required,
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
