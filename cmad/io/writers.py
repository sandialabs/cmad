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

``cmad calibrate`` adds three structured outputs:

- ``opt_history.json`` — per-fun-call ``{J, grad_norm, params}`` entries;
  ``params`` and the top-level ``active_param_paths`` are included only
  when ``optimizer.log_params`` is ``true``.
- ``opt_params.yaml`` — the deck's ``parameters:`` subtree with active
  leaves rewritten to their raw-coordinate final values (directly
  substitutable into a follow-up deck).
- ``opt_status.json`` — scipy ``OptimizeResult`` fields (``success``,
  ``status``, ``message``, ``fun``, ``nfev``, ``njev``, ``nit``); the
  canonical optimum ``x`` is deliberately omitted — raw final values
  live in ``opt_params.yaml``.

FE-side trajectories are written via :func:`write_fe_exodus` to a single
Exodus II file; the nodal + per-block element fields come from a
:class:`FEOutputPlan` (built by :func:`resolve_fe_output_plan` from the
deck's source-grouped ``output.global residual`` / ``output.local
residual`` selection). FE primal additionally writes ``J.json`` when the
deck supplies a ``qoi`` section.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from cmad.fem.fe_problem import FEProblem, FEState
from cmad.fem.postprocess import (
    DERIVED_OUTPUT_REGISTRY,
    evaluate_state_var_at_ips,
)
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.io.exodus import ExodusWriter
from cmad.io.results import FieldSpec, ip_average_to_element
from cmad.models.var_types import VarType

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


@dataclass(frozen=True)
class ResolvedNodalField:
    """A nodal (GR) output field resolved against the GR's
    ``primary_output_fields()`` catalog; the writer materializes it via
    ``gr.evaluate_nodal_field(name, ...)``.
    """

    name: str
    var_type: VarType


@dataclass(frozen=True)
class ResolvedElementField:
    """An element (Model) output field resolved against a block's catalog.

    ``evaluator`` returns per-IP values ``(n_elems, n_ip, n_comp)`` for one
    block at one step; the writer reduces them with
    :func:`cmad.io.results.ip_average_to_element`. State variables bind
    :func:`cmad.fem.postprocess.evaluate_state_var_at_ips` at a fixed
    ``resid_idx``; derived fields carry their
    :data:`cmad.fem.postprocess.DERIVED_OUTPUT_REGISTRY` evaluator.
    """

    name: str
    var_type: VarType
    evaluator: Callable[
        [FEProblem, FEState, int, str], NDArray[np.floating]
    ]


@dataclass(frozen=True)
class FEOutputPlan:
    """The fully resolved Exodus output selection for an FE primal run.

    ``nodal`` are the GR fields written per vertex; ``element_by_block``
    are the per-block model fields written per element (IP-averaged). Built
    by :func:`resolve_fe_output_plan` from the deck ``output`` section and
    the catalogs the GR + per-block Models advertise.
    """

    nodal: list[ResolvedNodalField]
    element_by_block: dict[str, list[ResolvedElementField]]


def _element_catalog_for_block(
        fe_problem: FEProblem, block: str,
) -> dict[str, ResolvedElementField]:
    """The block's selectable element-output fields, keyed by name.

    State variables (meaningful only on a COUPLED block -- CLOSED_FORM
    never solves xi) bind ``evaluate_state_var_at_ips`` at the residual
    index; derived quantities the model advertises via
    ``derived_output_field_names()`` carry their ``DERIVED_OUTPUT_REGISTRY``
    evaluator. Raises if a state-variable name collides with a derived name.
    """
    model = fe_problem.models_by_block[block]
    mode = fe_problem.modes_by_block[block]

    catalog: dict[str, ResolvedElementField] = {}
    if mode == GlobalResidualMode.COUPLED:
        for resid_idx, (name, var_type) in enumerate(
                model.state_output_fields(),
        ):
            catalog[name] = ResolvedElementField(
                name, var_type,
                partial(evaluate_state_var_at_ips, resid_idx=resid_idx),
            )

    for name in model.derived_output_field_names():
        if name in catalog:
            raise ValueError(
                f"block '{block}': derived output {name!r} collides with a "
                f"local state-variable name (state vars: {sorted(catalog)}); "
                f"rename one",
            )
        derived = DERIVED_OUTPUT_REGISTRY.get(name)
        if derived is None:
            raise ValueError(
                f"block '{block}': model advertises derived output {name!r} "
                f"absent from DERIVED_OUTPUT_REGISTRY "
                f"({sorted(DERIVED_OUTPUT_REGISTRY)})",
            )
        catalog[name] = ResolvedElementField(
            name, derived.var_type, derived.evaluator,
        )
    return catalog


def resolve_fe_output_plan(
        output_section: dict[str, Any],
        fe_problem: FEProblem,
) -> FEOutputPlan:
    """Resolve the deck ``output`` selection into an :class:`FEOutputPlan`.

    Output is grouped by source: ``output["global residual"]`` lists GR
    field names (resolved against ``gr.primary_output_fields()``);
    ``output["local residual"]`` is ``{block: [names]}`` (resolved against
    each block's state + derived catalog). Either group may be omitted, in
    which case that source's full advertised set is written (GR: every
    var_name; per block: every state variable + derived field). An unknown
    field name or unknown block raises -- the build-time guarantee that a
    requested output is producible.
    """
    gr = fe_problem.gr
    nodal_catalog = dict(gr.primary_output_fields())
    nodal_sel = output_section.get("global residual")
    if nodal_sel is None:
        nodal_names = list(nodal_catalog)
    else:
        for name in nodal_sel:
            if name not in nodal_catalog:
                raise ValueError(
                    f"output.global residual: unknown field {name!r}; the "
                    f"GR exposes {sorted(nodal_catalog)}",
                )
        nodal_names = list(nodal_sel)
    nodal = [
        ResolvedNodalField(name, nodal_catalog[name])
        for name in nodal_names
    ]

    blocks = list(fe_problem.mesh.element_blocks)
    element_sel = output_section.get("local residual")
    if element_sel is not None:
        unknown_blocks = set(element_sel) - set(blocks)
        if unknown_blocks:
            raise ValueError(
                f"output.local residual: unknown block(s) "
                f"{sorted(unknown_blocks)}; mesh element blocks are "
                f"{sorted(blocks)}",
            )

    element_by_block: dict[str, list[ResolvedElementField]] = {}
    for block in blocks:
        catalog = _element_catalog_for_block(fe_problem, block)
        if element_sel is None or block not in element_sel:
            names = list(catalog)
        else:
            for name in element_sel[block]:
                if name not in catalog:
                    raise ValueError(
                        f"output.local residual['{block}']: unknown field "
                        f"{name!r}; the block exposes {sorted(catalog)}",
                    )
            names = list(element_sel[block])
        element_by_block[block] = [catalog[name] for name in names]

    return FEOutputPlan(nodal=nodal, element_by_block=element_by_block)


def write_fe_exodus(
        out_dir: Path,
        prefix: str,
        fe_problem: FEProblem,
        fe_state: FEState,
        output_plan: FEOutputPlan,
        exodus_filename: str,
) -> None:
    """Write FE results to an Exodus II file from a resolved output plan.

    Opens an :class:`ExodusWriter` at
    ``out_dir / f"{prefix}{exodus_filename}"`` with the plan's nodal +
    per-block element :class:`FieldSpec` lists, then iterates over
    ``fe_state.t_history`` writing each field per step: nodal via
    ``gr.evaluate_nodal_field`` and element via the field's per-IP
    evaluator reduced with
    :func:`cmad.io.results.ip_average_to_element`. The
    :func:`resolve_fe_output_plan` selection is validated at build time,
    so every name here is known.
    """
    out_path = out_dir / f"{prefix}{exodus_filename}"
    gr = fe_problem.gr
    nodal_specs = [
        FieldSpec(f.name, f.var_type) for f in output_plan.nodal
    ]
    element_specs_by_block: dict[str, Sequence[FieldSpec]] = {
        block: [FieldSpec(f.name, f.var_type) for f in fields]
        for block, fields in output_plan.element_by_block.items()
    }
    with ExodusWriter(
            out_path,
            mesh=fe_problem.mesh,
            nodal_field_specs=nodal_specs,
            element_field_specs=element_specs_by_block,
    ) as writer:
        for step in range(len(fe_state.t_history)):
            nodal_data = {
                f.name: gr.evaluate_nodal_field(
                    f.name, fe_problem, fe_state, step,
                )
                for f in output_plan.nodal
            }
            element_data = {
                block: {
                    f.name: ip_average_to_element(
                        f.evaluator(fe_problem, fe_state, step, block),
                        fe_problem.geometry_cache, block,
                    )
                    for f in fields
                }
                for block, fields in output_plan.element_by_block.items()
            }
            writer.write_step(
                fe_state.t_history[step],
                nodal_data=nodal_data,
                element_data=element_data,
            )


def write_opt_history(
        out_dir: Path,
        prefix: str,
        history: list[dict[str, Any]],
        active_param_paths: list[str] | None = None,
) -> None:
    """Write the per-fun-call optimization trace.

    ``history`` entries carry ``{J, grad_norm}`` (and ``params`` when
    parameter logging is enabled). ``grad_norm`` is the L2 norm of the
    canonical-coordinate gradient — matches scipy's internal convergence
    metric. ``params`` is raw-coordinate and ordered per
    ``active_param_paths``, which is included only when param logging is
    enabled.
    """
    payload: dict[str, Any] = {"history": history}
    if active_param_paths is not None:
        payload["active_param_paths"] = active_param_paths
    with (out_dir / f"{prefix}opt_history.json").open("w") as f:
        json.dump(payload, f, indent=2)


def write_opt_params(
        out_dir: Path,
        prefix: str,
        deck_parameters: dict[str, Any],
        current_values: Any,
) -> None:
    """Write the optimized parameters as a substitutable deck subtree.

    Takes the deck's original ``parameters:`` subtree and rewrites each
    leaf value to the current raw-coordinate value in ``current_values``
    (which should be ``Parameters.values`` after the optimizer's final
    ``set_active_values_from_flat(x_opt, are_canonical=True)``). Leaf
    envelope metadata (``active``, ``transform``) and inactive-leaf
    values pass through unchanged.
    """
    updated = _inject_values(copy.deepcopy(deck_parameters), current_values)
    with (out_dir / f"{prefix}opt_params.yaml").open("w") as f:
        yaml.safe_dump(
            {"parameters": updated},
            f, default_flow_style=False, sort_keys=False,
        )


def write_opt_status(
        out_dir: Path,
        prefix: str,
        status: dict[str, Any],
) -> None:
    """Write the scipy ``OptimizeResult`` status fields as JSON."""
    with (out_dir / f"{prefix}opt_status.json").open("w") as f:
        json.dump(status, f, indent=2)


def _inject_values(deck_node: Any, values_node: Any) -> Any:
    """Rewrite each leaf in ``deck_node`` with the matching value from
    ``values_node`` (parallel pytree). Leaves in ``deck_node`` are either
    bare scalars/lists or ``{value, active, transform}`` envelopes; both
    are handled uniformly, with envelope metadata preserved.
    """
    if isinstance(deck_node, dict) and "value" in deck_node:
        deck_node["value"] = _to_yaml(values_node)
        return deck_node
    if isinstance(deck_node, dict):
        return {k: _inject_values(v, values_node[k])
                for k, v in deck_node.items()}
    return _to_yaml(values_node)


def _to_yaml(x: Any) -> Any:
    """Coerce a pytree leaf (numpy/jax/python scalar or array) to a YAML-
    safe Python-native type so ``yaml.safe_dump`` can serialize it.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        try:
            return x.tolist()
        except (TypeError, AttributeError):
            pass
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x
