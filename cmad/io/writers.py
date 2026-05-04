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

**F1+ extension point.** When ``GlobalResidual`` lands, it will have its
own block structure (displacement, pressure, possibly more). A sibling
``write_global_fields`` (or similarly named) should mirror :func:`write_xi`'s
per-block file convention so FE-side trajectories follow the same naming
pattern and read-back idiom.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from cmad.fem.fe_problem import FEProblem, FEState
from cmad.io.exodus import ExodusWriter
from cmad.io.results import FieldSpec
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


_VAR_TYPE_BY_DECK_NAME: dict[str, VarType] = {
    "scalar": VarType.SCALAR,
    "vector": VarType.VECTOR,
    "sym_tensor": VarType.SYM_TENSOR,
    "tensor": VarType.TENSOR,
}


def _field_spec_from_deck(d: dict[str, Any]) -> FieldSpec:
    return FieldSpec(
        name=d["name"],
        var_type=_VAR_TYPE_BY_DECK_NAME[d["var_type"]],
    )


def _resolve_fe_field_specs(
        output_section: dict[str, Any],
        fe_problem: FEProblem,
) -> tuple[list[FieldSpec], dict[str, Sequence[FieldSpec]]]:
    """Resolve writer field specs from deck or GR defaults.

    Each (nodal, element) bucket independently falls back to
    ``fe_problem.gr.default_output_fields()`` when the deck omits it.
    Element defaults are replicated across every key in
    ``fe_problem.mesh.element_blocks``; deck-supplied element specs
    are taken as-is, letting the user emit different fields on
    different blocks.
    """
    gr_defaults = fe_problem.gr.default_output_fields()
    blocks = list(fe_problem.mesh.element_blocks)

    nodal_deck = output_section.get("nodal fields")
    nodal_specs = (
        list(gr_defaults["nodal"]) if nodal_deck is None
        else [_field_spec_from_deck(d) for d in nodal_deck]
    )

    element_specs_by_block: dict[str, Sequence[FieldSpec]]
    element_deck = output_section.get("element fields by block")
    if element_deck is None:
        element_specs_by_block = {
            block: list(gr_defaults["element"]) for block in blocks
        }
    else:
        element_specs_by_block = {
            block: [_field_spec_from_deck(d) for d in specs]
            for block, specs in element_deck.items()
        }
    return nodal_specs, element_specs_by_block


def write_fe_exodus(
        out_dir: Path,
        prefix: str,
        fe_problem: FEProblem,
        fe_state: FEState,
        output_section: dict[str, Any],
) -> None:
    """Write FE primal results to an Exodus II file.

    Resolves nodal + per-block element :class:`FieldSpec` lists from
    the deck's ``output`` section; when either bucket is omitted,
    falls back to ``fe_problem.gr.default_output_fields()`` (element
    defaults replicated across every mesh element block). Opens an
    :class:`ExodusWriter` at ``out_dir / f"{prefix}primal.exo"`` and
    iterates over ``fe_state.t_history`` calling
    ``gr.evaluate_nodal_field`` and ``gr.evaluate_element_field`` per
    declared spec at each step.
    """
    nodal_specs, element_specs_by_block = _resolve_fe_field_specs(
        output_section, fe_problem,
    )
    out_path = out_dir / f"{prefix}primal.exo"
    gr = fe_problem.gr
    with ExodusWriter(
            out_path,
            mesh=fe_problem.mesh,
            nodal_field_specs=nodal_specs,
            element_field_specs=element_specs_by_block,
    ) as writer:
        for step in range(len(fe_state.t_history)):
            nodal_data = {
                spec.name: gr.evaluate_nodal_field(
                    spec.name, fe_problem, fe_state, step,
                )
                for spec in nodal_specs
            }
            element_data = {
                block: {
                    spec.name: gr.evaluate_element_field(
                        spec.name, fe_problem, fe_state, step, block,
                    )
                    for spec in specs
                }
                for block, specs in element_specs_by_block.items()
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
