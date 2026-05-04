"""Shared helpers for CMAD subcommand orchestrators.

The MP subcommands share a deck-load / defaults / schema / model /
parameters / deformation-history / (optional) QoI construction prelude
and an output-location resolution tail. The FE primal subcommand has
its own builder (:func:`build_fe_problem_from_deck`) that mirrors the
shape: deck → mesh → GR → per-block Models → DBCs / NBCs / forcing →
:class:`cmad.fem.fe_problem.FEProblem` plus a time schedule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.bcs import DirichletBC, NeumannBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import P1_TET, Q1_HEX, FiniteElement
from cmad.fem.quadrature import (
    QuadratureRule,
    hex_quadrature,
    quad_quadrature,
    tet_quadrature,
    tri_quadrature,
)
from cmad.global_residuals.global_residual import GlobalResidual
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.io.deck import apply_deck_defaults, load_deck
from cmad.io.deformation import load_history
from cmad.io.exodus import read_mesh
from cmad.io.expressions import parse_scalar_expression
from cmad.io.params_builder import build_parameters
from cmad.io.qoi_data import load_qoi_data
from cmad.io.registry import (
    resolve_global_residual,
    resolve_model,
    resolve_qoi,
)
from cmad.io.schema import validate_deck
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.qois.qoi import QoI
from cmad.typing import JaxArray


@dataclass(frozen=True)
class MPProblem:
    resolved: dict[str, Any]
    parameters: Parameters
    model: Model
    F: NDArray[np.float64]
    qoi: QoI | None


def build_mp_problem(
        deck_path: Path, subcommand: str,
) -> MPProblem:
    """Build the material-point problem shared by all subcommands.

    Runs deck load + defaults + schema validation, resolves the
    registered model and (for all subcommands except ``primal``) QoI,
    builds parameters, and loads the deformation history. The returned
    problem's ``qoi`` is ``None`` iff ``subcommand == "primal"``.
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

    return MPProblem(
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


_DEFAULT_FE_PER_FAMILY: dict[ElementFamily, FiniteElement] = {
    ElementFamily.HEX_LINEAR: Q1_HEX,
    ElementFamily.TET_LINEAR: P1_TET,
}

_FE_BY_NAME: dict[str, FiniteElement] = {
    "Q1": Q1_HEX,
    "Q1_HEX": Q1_HEX,
    "P1": P1_TET,
    "P1_TET": P1_TET,
}

_BC_COORD_NAMES: tuple[str, ...] = ("x", "y", "z", "t")


@dataclass(frozen=True)
class FEProblemBundle:
    resolved: dict[str, Any]
    fe_problem: FEProblem
    t_schedule: NDArray[np.float64]


def build_fe_problem_from_deck(
        deck_path: Path, subcommand: str,
) -> FEProblemBundle:
    """Build the FE problem shared by FE-track subcommands.

    Mirrors :func:`build_mp_problem`'s shape: deck load + defaults +
    schema validation, then resolves the registered GR / per-block
    Models, parses BC value expressions to JAX-traceable callables,
    builds the :class:`cmad.fem.dof.GlobalDofMap` and
    :class:`cmad.fem.fe_problem.FEProblem`, and assembles the time
    schedule. The mode-per-block dispatch (``CLOSED_FORM`` vs
    ``COUPLED``) is decided here from each Model's
    ``supports_closed_form_cauchy`` flag and threaded explicitly into
    :func:`build_fe_problem`.
    """
    deck = load_deck(deck_path)
    resolved = apply_deck_defaults(deck)
    validate_deck(resolved, subcommand)

    mesh_path = Path(resolved["discretization"]["mesh file"])
    if not mesh_path.is_absolute():
        mesh_path = deck_path.parent / mesh_path
    mesh = read_mesh(mesh_path)
    ndims = int(mesh.nodes.shape[1])

    gr_section = resolved["residuals"]["global residual"]
    gr_cls = resolve_global_residual(gr_section["type"])
    gr = gr_cls.from_deck(gr_section, ndims=ndims)

    local_section = resolved["residuals"]["local residual"]
    models_by_block = _build_models_by_block(local_section, mesh)
    modes_by_block = {
        block: (
            GlobalResidualMode.CLOSED_FORM
            if model.supports_closed_form_cauchy
            else GlobalResidualMode.COUPLED
        )
        for block, model in models_by_block.items()
    }

    field_layouts = _build_field_layouts(
        resolved["discretization"], gr, mesh.element_family,
    )
    components_by_field = {
        str(gr.var_names[r]): int(gr._num_eqs[r])
        for r in range(gr.num_residuals)
    }

    dirichlet_bcs = _build_dirichlet_bcs(
        resolved.get("dirichlet bcs"), gr,
    )
    dof_map = build_dof_map(
        mesh, field_layouts, dirichlet_bcs, components_by_field,
    )

    neumann_bcs = _build_neumann_bcs(
        resolved.get("surface flux bcs"), gr,
    )
    forcing_fns = _build_forcing_fns(resolved.get("body forces"), gr)

    assembly_quadrature, side_quadrature = _build_quadrature_overrides(
        resolved["discretization"], mesh.element_family,
    )

    fe_problem = build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=gr,
        models_by_block=models_by_block,
        modes_by_block=modes_by_block,
        forcing_fns_by_block_idx=forcing_fns or None,
        assembly_quadrature=assembly_quadrature,
        neumann_bcs=neumann_bcs,
        side_quadrature=side_quadrature,
    )

    t_schedule = _load_t_schedule(
        resolved["discretization"], deck_path.parent,
    )

    return FEProblemBundle(
        resolved=resolved, fe_problem=fe_problem, t_schedule=t_schedule,
    )


def _build_field_layouts(
        disc_section: dict[str, Any],
        gr: GlobalResidual,
        family: ElementFamily,
) -> list[GlobalFieldLayout]:
    """One layout per GR residual block, FE looked up per ``var_name``.

    Per-var FE choice comes from ``discretization.finite elements``
    (deck-side discretization decision); omitted var_names fall back to
    family-matched linear Lagrange (Q1_HEX / P1_TET). Stray override
    keys that don't match any GR var_name raise — silent typos in the
    deck would otherwise apply nothing.
    """
    if family not in _DEFAULT_FE_PER_FAMILY:
        raise ValueError(
            f"unsupported mesh element family for FE deck: {family.name}; "
            f"supported families: "
            f"{sorted(f.name for f in _DEFAULT_FE_PER_FAMILY)}",
        )
    overrides = disc_section.get("finite elements") or {}
    var_names = {str(gr.var_names[r]) for r in range(gr.num_residuals)}
    unknown = set(overrides) - var_names
    if unknown:
        raise ValueError(
            f"discretization.finite elements references unknown "
            f"var_name(s) {sorted(unknown)}; GR var_names: "
            f"{sorted(var_names)}",
        )
    layouts: list[GlobalFieldLayout] = []
    for r in range(gr.num_residuals):
        var_name = str(gr.var_names[r])
        fe_name = overrides.get(var_name)
        fe = (
            _resolve_fe_name(fe_name, family, var_name)
            if fe_name is not None
            else _DEFAULT_FE_PER_FAMILY[family]
        )
        layouts.append(GlobalFieldLayout(name=var_name, finite_element=fe))
    return layouts


def _resolve_fe_name(
        name: str, family: ElementFamily, var_name: str,
) -> FiniteElement:
    fe = _FE_BY_NAME.get(name)
    if fe is None:
        raise ValueError(
            f"discretization.finite elements['{var_name}']: unknown "
            f"finite element '{name}'; known: {sorted(set(_FE_BY_NAME))}",
        )
    if fe.element_family != family:
        raise ValueError(
            f"discretization.finite elements['{var_name}']: '{name}' has "
            f"family {fe.element_family.name} but the mesh is "
            f"{family.name}",
        )
    return fe


def _build_quadrature_overrides(
        disc_section: dict[str, Any],
        family: ElementFamily,
) -> tuple[
    dict[ElementFamily, QuadratureRule] | None,
    dict[ElementFamily, QuadratureRule] | None,
]:
    """Resolve scalar volume/surface degrees from the deck (or ``(None,
    None)`` to inherit ``build_fe_problem``'s family defaults).

    The deck shape is intentionally scalar-per-kind for now; per-IP-set
    granularity (e.g. distinct quadrature for a stabilization term) is
    a future runtime+schema change tracked as a deferred design note.
    """
    quad_section = disc_section.get("quadrature") or {}
    vol_deg = quad_section.get("volume degree")
    surf_deg = quad_section.get("surface degree")
    assembly = (
        {family: _quad_rule(family, "volume", int(vol_deg))}
        if vol_deg is not None else None
    )
    side = (
        {family: _quad_rule(family, "surface", int(surf_deg))}
        if surf_deg is not None else None
    )
    return assembly, side


def _quad_rule(
        family: ElementFamily, kind: str, degree: int,
) -> QuadratureRule:
    if kind == "volume":
        if family == ElementFamily.HEX_LINEAR:
            return hex_quadrature(degree=degree)
        if family == ElementFamily.TET_LINEAR:
            return tet_quadrature(degree=degree)
    elif kind == "surface":
        if family == ElementFamily.HEX_LINEAR:
            return quad_quadrature(degree=degree)
        if family == ElementFamily.TET_LINEAR:
            return tri_quadrature(degree=degree)
    raise ValueError(
        f"_quad_rule: unsupported (family={family.name}, kind={kind})",
    )


def _build_models_by_block(
        local_section: dict[str, Any], mesh: Any,
) -> dict[str, Model]:
    materials = local_section["materials"]
    mesh_blocks = set(mesh.element_blocks.keys())
    deck_blocks = set(materials.keys())
    if mesh_blocks != deck_blocks:
        raise ValueError(
            "residuals.local residual.materials keys "
            f"({sorted(deck_blocks)}) must match mesh element blocks "
            f"({sorted(mesh_blocks)})",
        )
    model_cls = resolve_model(local_section["type"])
    return {
        block: model_cls.from_deck(
            local_section, build_parameters(materials[block]),
        )
        for block in materials
    }


def _resolve_resid_idx(
        resid_name: str, gr: GlobalResidual, where: str,
) -> int:
    try:
        return gr.resid_names.index(resid_name)
    except ValueError as e:
        raise ValueError(
            f"{where}: residual '{resid_name}' is not declared by GR "
            f"(known: {gr.resid_names})",
        ) from e


def _build_dirichlet_bcs(
        dbc_section: dict[str, Any] | None, gr: GlobalResidual,
) -> list[DirichletBC]:
    if not dbc_section:
        return []
    bcs: list[DirichletBC] = []
    for entry_name, entry in dbc_section.get("expression", {}).items():
        resid_name, eq, sideset, value_expr = entry
        where = f"dirichlet bcs.expression.{entry_name}"
        r = _resolve_resid_idx(resid_name, gr, where)
        num_eqs = int(gr._num_eqs[r])
        if not (0 <= int(eq) < num_eqs):
            raise ValueError(
                f"{where}: eq {eq} out of range for residual "
                f"'{resid_name}' (num_eqs={num_eqs})",
            )
        scalar_fn = parse_scalar_expression(value_expr, _BC_COORD_NAMES)
        bcs.append(DirichletBC(
            sideset_names=[str(sideset)],
            field_name=str(gr.var_names[r]),
            dofs=[int(eq)],
            values=_make_dbc_value_callable(scalar_fn),
        ))
    return bcs


def _build_neumann_bcs(
        sfb_section: dict[str, Any] | None, gr: GlobalResidual,
) -> list[NeumannBC]:
    if not sfb_section:
        return []
    bcs: list[NeumannBC] = []
    for entry_name, entry in sfb_section.get("expression", {}).items():
        where = f"surface flux bcs.expression.{entry_name}"
        resid_name = entry[0]
        sideset = entry[1]
        component_exprs = entry[2:]
        r = _resolve_resid_idx(resid_name, gr, where)
        num_components = int(gr._num_eqs[r])
        if len(component_exprs) != num_components:
            raise ValueError(
                f"{where}: residual '{resid_name}' takes {num_components} "
                f"components, got {len(component_exprs)}",
            )
        component_fns = [
            parse_scalar_expression(e, _BC_COORD_NAMES)
            for e in component_exprs
        ]
        bcs.append(NeumannBC(
            sideset_names=[str(sideset)],
            field_name=str(gr.var_names[r]),
            values=_make_nbc_value_callable(component_fns),
        ))
    return bcs


def _build_forcing_fns(
        body_section: dict[str, Any] | None, gr: GlobalResidual,
) -> dict[int, Callable[
    [NDArray[np.floating] | JaxArray, float],
    NDArray[np.floating] | JaxArray,
]]:
    if not body_section:
        return {}
    fns_by_idx: dict[int, Callable[
        [NDArray[np.floating] | JaxArray, float],
        NDArray[np.floating] | JaxArray,
    ]] = {}
    for entry_name, entry in body_section.get("expression", {}).items():
        where = f"body forces.expression.{entry_name}"
        resid_name = entry[0]
        component_exprs = entry[1:]
        r = _resolve_resid_idx(resid_name, gr, where)
        num_components = int(gr._num_eqs[r])
        if len(component_exprs) != num_components:
            raise ValueError(
                f"{where}: residual '{resid_name}' takes {num_components} "
                f"components, got {len(component_exprs)}",
            )
        if r in fns_by_idx:
            raise ValueError(
                f"{where}: residual '{resid_name}' already has a body-"
                "force entry; one forcing fn per residual block",
            )
        component_fns = [
            parse_scalar_expression(e, _BC_COORD_NAMES)
            for e in component_exprs
        ]
        fns_by_idx[r] = _make_body_force_callable(component_fns)
    return fns_by_idx


def _make_dbc_value_callable(
        scalar_fn: Callable[..., Any],
) -> Callable[[NDArray[np.floating], float], NDArray[np.floating]]:
    """Wrap a scalar ``f(x, y, z, t)`` into a DBC ``(coords, t) ->
    (N_set, 1)`` callable.

    DBC values materialize at the Python boundary (see
    :meth:`cmad.fem.dof.GlobalDofMap.evaluate_prescribed_values`), so
    the closure coerces JAX scalars back to NumPy via ``np.asarray``
    and broadcasts constants up to the BC's vertex count.
    """
    def fn(coords: NDArray[np.floating], t: float) -> NDArray[np.floating]:
        n_set = int(coords.shape[0])
        val = np.asarray(scalar_fn(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], t=t,
        ), dtype=np.float64)
        return np.broadcast_to(val, (n_set,)).reshape(n_set, 1)
    return fn


def _make_nbc_value_callable(
        component_fns: list[Callable[..., Any]],
) -> Callable[
    [NDArray[np.floating] | JaxArray, float],
    JaxArray,
]:
    """Stack per-component scalar callables into an NBC ``(coords_ip, t)
    -> (N_side_ips, num_components)`` callable.

    Inputs may be JAX tracers (the surface-scatter step in
    :mod:`cmad.fem.neumann` runs under jit/vmap), so the closure stays
    in ``jax.numpy`` and broadcasts constants up to the per-call IP
    count via ``jnp.broadcast_to``.
    """
    def fn(
            coords_ip: NDArray[np.floating] | JaxArray, t: float,
    ) -> JaxArray:
        n_ips = coords_ip.shape[0]
        cols = []
        for c in component_fns:
            val = jnp.asarray(c(
                x=coords_ip[:, 0], y=coords_ip[:, 1],
                z=coords_ip[:, 2], t=t,
            ))
            cols.append(jnp.broadcast_to(val, (n_ips,)))
        return jnp.stack(cols, axis=-1)
    return fn


def _make_body_force_callable(
        component_fns: list[Callable[..., Any]],
) -> Callable[
    [NDArray[np.floating] | JaxArray, float],
    JaxArray,
]:
    """Stack per-component scalar callables into a forcing
    ``(coords_ip, t) -> (num_eqs,)`` callable.

    ``coords_ip`` is the per-IP physical coord ``(ndims,)`` (single
    point in, vector out). The assembly layer vmap-traces this, so the
    closure stays in ``jax.numpy``.
    """
    def fn(
            coords_ip: NDArray[np.floating] | JaxArray, t: float,
    ) -> JaxArray:
        cols = [
            jnp.asarray(c(
                x=coords_ip[0], y=coords_ip[1], z=coords_ip[2], t=t,
            ))
            for c in component_fns
        ]
        return jnp.stack(cols)
    return fn


def _load_t_schedule(
        disc_section: dict[str, Any], base_dir: Path,
) -> NDArray[np.float64]:
    """Materialize the time schedule from the ``discretization`` section.

    Three branches mirror the schema's ``oneOf`` in
    ``discretization.yaml``: ``num steps`` + ``step size`` produces an
    arithmetic sweep including the initial time; ``times file`` reads a
    1D array from disk (``.npy`` via ``np.load``, ``.txt`` / ``.csv``
    via ``np.loadtxt``); ``times`` consumes an inline list.
    """
    if "times" in disc_section:
        return np.asarray(
            disc_section["times"], dtype=np.float64,
        ).ravel()
    if "times file" in disc_section:
        path = Path(disc_section["times file"])
        if not path.is_absolute():
            path = base_dir / path
        suffix = path.suffix.lower()
        if suffix == ".npy":
            data = np.load(path)
        elif suffix in (".txt", ".csv"):
            data = np.loadtxt(path)
        else:
            raise ValueError(
                f"discretization.times file: unsupported extension "
                f"'{suffix}' for path {path}; expected .npy/.txt/.csv",
            )
        return np.asarray(data, dtype=np.float64).ravel()
    n = int(disc_section["num steps"])
    dt = float(disc_section["step size"])
    return np.arange(n + 1, dtype=np.float64) * dt
