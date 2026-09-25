"""Schema validator for the CMAD deck driver.

Loads YAML-encoded JSON Schema fragments from ``cmad/io/schemas/``,
composes the (problem-type, subcommand)-specific schema (stitching in
the registered model's fragment for MP decks and the registered QoI's
fragment when the subcommand uses one), and validates a parsed deck
with aggregated errors formatted as ``path: reason`` lines per the
driver error-reporting convention.

Section names with spaces (FE convention: ``dirichlet bcs``,
``surface flux bcs``, ``volumetric sources``) map to underscored fragment
filenames (``dirichlet_bcs.yaml``, ...) at fragment-load time.

Two normalization passes run before validation, mirroring the same
helpers in :mod:`cmad.io.deck`: optional top-level wrapper auto-unwrap
and Calibr8-only-section strip. Both are idempotent so calling
``validate_deck`` directly on a not-yet-normalized deck produces the
same result as calling ``apply_deck_defaults`` first.

An FE input file with a ``specimens`` section holds the sections that
describe the material and the solve once at the top level, and the
sections a specimen entry holds (:data:`_FE_SPECIMEN_SECTIONS`) inside
each entry, with an optional ``weight``. Each entry is validated against
the same fragments a single specimen file uses.

The checks here resolve the deck's model, QoI, and global residual
names through :mod:`cmad.io.registry`, which imports only the named
module; an unknown name raises with a listing of the modules actually
present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from cmad.io.deck import strip_calibr8_only, unwrap_top_level
from cmad.io.registry import (
    resolve_global_residual,
    resolve_model,
    resolve_qoi,
)

_SCHEMAS_DIR = Path(__file__).parent / "schemas"

# (problem_type, subcommand) -> (required_sections, optional_sections)
_SECTIONS: dict[tuple[str, str], tuple[list[str], list[str]]] = {
    ("material_point", "primal"): (
        ["problem", "model", "parameters", "deformation", "solver"],
        ["output"],
    ),
    ("material_point", "objective"): (
        ["problem", "model", "parameters",
         "deformation", "qoi", "solver"],
        ["output"],
    ),
    ("material_point", "gradient"): (
        ["problem", "model", "parameters",
         "deformation", "qoi", "sensitivity", "solver"],
        ["output"],
    ),
    ("material_point", "hessian"): (
        ["problem", "model", "parameters",
         "deformation", "qoi", "sensitivity", "solver"],
        ["output"],
    ),
    ("material_point", "calibrate"): (
        ["problem", "model", "parameters",
         "deformation", "qoi", "sensitivity", "solver", "optimizer"],
        ["output"],
    ),
    ("fe", "primal"): (
        ["problem", "discretization", "residuals"],
        ["output", "dirichlet bcs", "surface flux bcs", "volumetric sources",
         "initial conditions", "convection bcs", "radiation bcs",
         "linear solver", "qoi"],
    ),
    ("fe", "objective"): (
        ["problem", "discretization", "residuals", "qoi"],
        ["output", "dirichlet bcs", "surface flux bcs", "volumetric sources",
         "initial conditions", "convection bcs", "radiation bcs",
         "linear solver"],
    ),
    ("fe", "gradient"): (
        ["problem", "discretization", "residuals", "qoi"],
        ["output", "dirichlet bcs", "surface flux bcs", "volumetric sources",
         "initial conditions", "convection bcs", "radiation bcs",
         "linear solver"],
    ),
    ("fe", "hessian"): (
        ["problem", "discretization", "residuals", "qoi"],
        ["output", "dirichlet bcs", "surface flux bcs", "volumetric sources",
         "initial conditions", "convection bcs", "radiation bcs",
         "linear solver"],
    ),
    ("fe", "calibrate"): (
        ["problem", "discretization", "residuals", "qoi", "optimizer"],
        ["output", "dirichlet bcs", "surface flux bcs", "volumetric sources",
         "initial conditions", "convection bcs", "radiation bcs",
         "linear solver"],
    ),
}

# The FE sections a specimen entry holds in a multispecimen file; the
# others describe the material and the solve, shared by every specimen.
_FE_SPECIMEN_SECTIONS: tuple[str, ...] = (
    "discretization", "dirichlet bcs", "surface flux bcs",
    "volumetric sources", "initial conditions", "convection bcs",
    "radiation bcs", "qoi",
)


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
    if problem_type == "fe" and "specimens" in deck:
        qoi_name_by_tag = _check_specimens(deck, subcommand, all_sections)
        composed = _compose_multispecimen_schema(subcommand, qoi_name_by_tag)
    else:
        qoi_name: str | None = None
        if "qoi" in all_sections and "qoi" in deck:
            _check_qoi_registered(deck)
            qoi_name = deck["qoi"]["name"]
        composed = _compose_schema(problem_type, subcommand, qoi_name=qoi_name)
    errors = list(Draft202012Validator(composed).iter_errors(deck))
    if errors:
        joined = "\n".join(_format_error(e) for e in errors)
        raise ValueError(f"deck validation failed:\n{joined}")


def _check_specimens(
        deck: dict[str, Any], subcommand: str, all_sections: list[str],
) -> dict[str, str | None]:
    """Reject a multispecimen file whose layout is wrong, and return each
    specimen's QoI name (``None`` for an entry without a ``qoi`` section)."""
    if subcommand == "primal":
        raise ValueError(
            "specimens: cmad primal runs one specimen; put the shared "
            "sections into that specimen's own input file",
        )
    at_top = [s for s in _FE_SPECIMEN_SECTIONS if s in deck]
    if at_top:
        raise ValueError(
            f"specimens: the section(s) {at_top} belong inside each "
            "specimen entry when the file has a specimens section",
        )
    specimens = deck["specimens"]
    if not isinstance(specimens, dict) or not specimens:
        raise ValueError(
            "specimens: expected a mapping with one entry per specimen",
        )
    qoi_name_by_tag: dict[str, str | None] = {}
    for tag, entry in specimens.items():
        if not isinstance(entry, dict):
            raise ValueError(f"specimens.{tag}: expected a mapping of sections")
        qoi_name: str | None = None
        if "qoi" in all_sections and "qoi" in entry:
            _check_qoi_registered(entry, where=f"specimens.{tag}.qoi")
            qoi_name = entry["qoi"]["name"]
        qoi_name_by_tag[str(tag)] = qoi_name
    return qoi_name_by_tag


def _check_model_registered(
        deck: dict[str, Any], problem_type: str,
) -> None:
    """Verify the deck's model name resolves to a module.

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

    resolve_model(name, where=path)


def _check_global_residual_registered(deck: dict[str, Any]) -> None:
    """Verify the FE deck's GR name resolves to a module."""
    residuals = deck.get("residuals")
    if not isinstance(residuals, dict):
        raise ValueError("residuals: missing or not a mapping")
    glob = residuals.get("global residual")
    if not isinstance(glob, dict) or "type" not in glob:
        raise ValueError("residuals.global residual: missing 'type' field")
    resolve_global_residual(glob["type"])


def _check_qoi_registered(
        section: dict[str, Any], where: str = "qoi",
) -> None:
    qoi_section = section.get("qoi")
    if not isinstance(qoi_section, dict) or "name" not in qoi_section:
        raise ValueError(f"{where}: missing 'name' field")
    resolve_qoi(qoi_section["name"])


def _compose_schema(
        problem_type: str,
        subcommand: str,
        qoi_name: str | None = None,
) -> dict[str, Any]:
    required, optional = _SECTIONS[(problem_type, subcommand)]
    merged_defs: dict[str, Any] = {}
    properties = _fragments(required + optional, qoi_name, merged_defs)
    return _object_schema(required, properties, merged_defs)


def _compose_multispecimen_schema(
        subcommand: str, qoi_name_by_tag: dict[str, str | None],
) -> dict[str, Any]:
    """The schema of an FE file with a ``specimens`` section: the shared
    sections at the top level, and under ``specimens`` one object per tag
    holding the specimen sections and an optional ``weight``."""
    required, optional = _SECTIONS[("fe", subcommand)]
    shared_required = [s for s in required if s not in _FE_SPECIMEN_SECTIONS]
    shared_optional = [s for s in optional if s not in _FE_SPECIMEN_SECTIONS]
    specimen_required = [s for s in required if s in _FE_SPECIMEN_SECTIONS]
    specimen_optional = [s for s in optional if s in _FE_SPECIMEN_SECTIONS]
    merged_defs: dict[str, Any] = {}
    properties = _fragments(
        shared_required + shared_optional, None, merged_defs,
    )
    entries: dict[str, Any] = {}
    for tag, qoi_name in qoi_name_by_tag.items():
        entry_properties = _fragments(
            specimen_required + specimen_optional, qoi_name, merged_defs,
        )
        entry_properties["weight"] = {"type": "number", "exclusiveMinimum": 0}
        entries[tag] = {
            "type": "object",
            "required": specimen_required,
            "additionalProperties": False,
            "properties": entry_properties,
        }
    properties["specimens"] = {
        "type": "object",
        "minProperties": 1,
        "additionalProperties": False,
        "properties": entries,
    }
    return _object_schema(
        [*shared_required, "specimens"], properties, merged_defs,
    )


def _fragments(
        sections: list[str],
        qoi_name: str | None,
        merged_defs: dict[str, Any],
) -> dict[str, Any]:
    """The schema fragment of each section, keyed by section, each
    fragment's ``$defs`` moved into ``merged_defs``. The ``qoi`` fragment
    is the named QoI's, skipped when there is none."""
    properties: dict[str, Any] = {}
    for section in sections:
        if section == "model":
            fragment = _load_fragment("model.yaml")
        elif section == "qoi":
            if qoi_name is None:
                continue
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
    return properties


def _object_schema(
        required: list[str],
        properties: dict[str, Any],
        merged_defs: dict[str, Any],
) -> dict[str, Any]:
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
    path = _SCHEMAS_DIR / relative_path
    if not path.exists():
        raise RuntimeError(
            f"no schema fragment for '{relative_path}' under {_SCHEMAS_DIR}",
        )
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(
            f"schema fragment '{relative_path}' is empty or not a mapping",
        )
    return cast(dict[str, Any], data)


def _format_error(error: ValidationError) -> str:
    path = ".".join(str(p) for p in error.absolute_path) or "<root>"
    return f"{path}: {error.message}"
