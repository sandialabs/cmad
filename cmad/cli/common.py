"""Shared helpers for CMAD subcommand orchestrators.

The MP subcommands share a deck-load / defaults / schema / model /
parameters / deformation-history / (optional) QoI construction prelude
and an output-location resolution tail. The FE primal subcommand has
its own builder (:func:`build_fe_problem_from_deck`) that mirrors the
shape: deck → mesh → GR → per-block Models → DBCs / NBCs / forcing →
:class:`cmad.fem.fe_problem.FEProblem` plus a time schedule.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeAlias

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.fem.bcs import (
    DirichletBC,
    NeumannBC,
    make_nodal_field_values,
)
from cmad.fem.dof import (
    GlobalFieldLayout,
    build_dof_map,
    sideset_basis_fns,
)
from cmad.fem.driver import StateInit, build_fe_quasistatic_trajectory
from cmad.fem.element_family import ElementFamily
from cmad.fem.fe_problem import FEProblem, FEState, build_fe_problem
from cmad.fem.finite_element import P1_TET, P1_TRI, Q1_HEX, Q1_QUAD, FiniteElement
from cmad.fem.kernel_arrays import FEKernelArrays
from cmad.fem.mesh import Mesh, coordinate_side_sets
from cmad.fem.quadrature import (
    QuadratureRule,
    hex_quadrature,
    quad_quadrature,
    tet_quadrature,
    tri_quadrature,
)
from cmad.fem.sharding import place_element_leaves
from cmad.global_residuals.global_residual import GlobalResidual
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.io.calibration_data import CalibrationData, is_calibration_data
from cmad.io.deck import apply_deck_defaults, load_deck
from cmad.io.deformation import load_history
from cmad.io.expressions import parse_scalar_expression
from cmad.io.mesh_io import read_mesh_file
from cmad.io.params_builder import build_parameters
from cmad.io.qoi_data import load_displacement_data, load_qoi_data
from cmad.io.registry import (
    resolve_global_residual,
    resolve_model,
    resolve_qoi,
)
from cmad.io.schema import validate_deck
from cmad.models.deformation_types import DefType
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.qois.fe_qoi import FEQoI
from cmad.qois.qoi import QoI
from cmad.typing import JaxArray, Scalar


@dataclass(frozen=True)
class MPProblem:
    resolved: dict[str, Any]
    parameters: Parameters
    model: Model
    F: NDArray[np.float64]
    qoi: QoI | None


def _with_material_defaults(
        params_section: dict[str, Any], model_cls: type[Model],
) -> dict[str, Any]:
    """Return a copy of ``params_section`` with the model's
    ``material_defaults()`` setdefault-merged in, so the deck builder fills
    omitted top-level material parameters such as ``rotation matrix`` before
    :func:`build_parameters` splits them into the parallel parameter trees.
    """
    merged = dict(params_section)
    for key, default in model_cls.material_defaults().items():
        merged.setdefault(key, default)
    return merged


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
    parameters = build_parameters(
        _with_material_defaults(resolved["parameters"], model_cls),
    )
    def_type = DefType[resolved["model"]["def_type"].upper()]
    model = model_cls.from_deck(resolved["model"], parameters, def_type)

    F = load_history(
        resolved["deformation"], expected_ndims=model.ndims,
    )

    qoi: QoI | None = None
    if subcommand != "primal":
        qoi_cls = resolve_qoi(resolved["qoi"]["name"])
        if qoi_cls.problem_type != "material_point":
            raise ValueError(
                f"qoi.name '{resolved['qoi']['name']}' is registered "
                f"for problem_type='{qoi_cls.problem_type}', but the "
                f"deck has problem.type='material_point'"
            )
        assert issubclass(qoi_cls, QoI)
        data, weight = load_qoi_data(resolved["qoi"])
        qoi = qoi_cls.from_deck(resolved["qoi"], model, data, weight)

    return MPProblem(
        resolved=resolved, parameters=parameters,
        model=model, F=F, qoi=qoi,
    )


def resolve_output(
        resolved: dict[str, Any],
) -> tuple[Path, str, str]:
    """Resolve ``out_dir``, ``prefix``, and ``format`` from a validated deck.

    The output block is optional: when it is absent the directory defaults
    to the current working directory and the prefix to ``""``, so the
    subcommands that always emit their results (objective, gradient,
    hessian, calibrate) work without one. A given path is taken relative to
    the current working directory when not absolute, then created.
    ``format`` defaults to ``npy`` (filled into the deck only for MP;
    callers that don't emit array outputs can discard it).
    """
    output = resolved.get("output", {})
    out_dir = Path(output.get("path", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    return (
        out_dir,
        output.get("prefix", ""),
        output.get("format", "npy"),
    )


def nonlinear_solver_settings(
        gr_section: dict[str, Any], print_global_convergence: bool,
) -> dict[str, Any]:
    """The global Newton settings an input file's ``global residual``
    section gives, in the solver's keys; the rest take the solver's
    defaults."""
    names = {
        "nonlinear max iters": "max iters",
        "nonlinear absolute tol": "abs tol",
        "nonlinear relative tol": "rel tol",
        "line search": "line search",
        "initial guess": "initial guess",
        "extrapolation max ratio": "extrapolation max ratio",
    }
    settings = {
        key: gr_section[name] for name, key in names.items()
        if name in gr_section
    }
    settings["print convergence"] = print_global_convergence
    return settings


TrajectoryCost: TypeAlias = Callable[
    [JaxArray, StateInit, FEKernelArrays, JaxArray],
    tuple[JaxArray, tuple[JaxArray, JaxArray]],
]
"""``cost(params_flat, state_init, fe_arrays, t_schedule_jax)`` returning
``(J, (first_failed_step, iters_per_step))``: the QoI accumulated over
the forward solve on the given schedule and the solve's status, so a
caller can refine the schedule and evaluate again."""


def build_fe_trajectory_cost(
        bundle: FEProblemBundle,
        print_global_convergence: bool = False,
) -> tuple[JaxArray, StateInit, TrajectoryCost]:
    """Build ``(params_flat_init, state_init, cost)`` for the FE objective.

    ``params_flat`` is the concatenation of each mesh element block's
    flat-active canonical parameter vector
    (``Parameters.flat_active_values(return_canonical=True)``); inactive
    parameters are held at their stored values and don't appear in
    ``params_flat``. AD only flows through the active components: each
    block's ``Parameters.get_params_pytree_from_flat_canonical_active``
    overlays the traced active values onto the closure-captured full
    init vector before reconstructing the pytree, so inactive entries
    are JAX constants. Hessians are therefore
    ``(num_active, num_active)``, not ``(num_total, num_total)``.

    The :data:`TrajectoryCost` closure reconstructs ``params_by_block``
    per block, invokes :meth:`FEQoI.step_contribution` to build the
    per-step QoI closure against those reconstructed params, and runs
    the FE forward solve on the schedule it is given via the
    ``trajectory`` closure from
    :func:`cmad.fem.driver.build_fe_quasistatic_trajectory`; the
    schedule is an argument so that a refined one only retraces.
    ``state_init`` is the ``(U_init, xi_init_by_block)`` pair the time
    loop starts from; callers source ``fe_arrays`` (the
    :class:`FEKernelArrays` carrier) from
    ``bundle.fe_problem.kernel_arrays``.

    ``bundle.qoi`` must be non-None; the FE-side
    ``build_fe_problem_from_deck`` populates it for ``"objective"``,
    ``"gradient"``, ``"hessian"``, and ``"calibrate"`` subcommands.
    """
    fe_problem = bundle.fe_problem
    qoi = bundle.qoi
    if qoi is None:
        raise ValueError(
            "build_fe_trajectory_cost requires bundle.qoi to be set; "
            "FEProblemBundle from a 'primal' build has no QoI"
        )
    gr_section = bundle.resolved["residuals"]["global residual"]

    state = FEState.from_problem(
        fe_problem, t_init=float(bundle.t_schedule[0]),
    )
    U_init = jnp.asarray(state.U_at(0), dtype=jnp.float64)
    xi_init: dict[str, JaxArray] = place_element_leaves(
        {b: jnp.asarray(state.xi_at(0, b)) for b in fe_problem.models_by_block},
        fe_problem.device_mesh,
    )
    state_init: StateInit = (U_init, xi_init)

    dbc_arrays = fe_problem.kernel_arrays.dbc_arrays
    for t in bundle.t_schedule[1:]:
        fe_problem.dof_map.evaluate_prescribed_values(dbc_arrays, float(t))

    block_names = list(fe_problem.models_by_block.keys())
    per_block_init: list[JaxArray] = []
    per_block_lengths: list[int] = []
    for b in block_names:
        params_obj = fe_problem.models_by_block[b].parameters
        flat_active = params_obj.flat_active_values(return_canonical=True)
        per_block_init.append(jnp.asarray(flat_active, dtype=jnp.float64))
        per_block_lengths.append(int(flat_active.shape[0]))
    params_flat_init = (
        jnp.concatenate(per_block_init).astype(jnp.float64)
        if per_block_init else jnp.zeros((0,), dtype=jnp.float64)
    )
    boundaries = np.cumsum([0, *per_block_lengths])

    trajectory = build_fe_quasistatic_trajectory(
        fe_problem,
        nonlinear_solver_settings=nonlinear_solver_settings(
            gr_section, print_global_convergence,
        ),
        linear_solver_settings=bundle.resolved["linear solver"],
    )

    def cost(
            params_flat: JaxArray,
            state_init: StateInit,
            fe_arrays: FEKernelArrays,
            t_schedule_jax: JaxArray,
    ) -> tuple[JaxArray, tuple[JaxArray, JaxArray]]:
        params_by_block: dict[str, Any] = {}
        for i, b in enumerate(block_names):
            sub_flat = params_flat[boundaries[i]:boundaries[i + 1]]
            params_obj = fe_problem.models_by_block[b].parameters
            params_by_block[b] = (
                params_obj.get_params_pytree_from_flat_canonical_active(
                    sub_flat,
                )
            )
        qoi_step_contribution = qoi.step_contribution(
            params_by_block, fe_arrays,
        )
        # This runs under AD, so a non-converged step cannot raise from
        # here; the objective is built from what the trajectory reached
        # and the status says whether that was every step.
        _, _, J, first_failed_step, _rel_norm, iters_per_step = trajectory(
            fe_arrays,
            params_by_block,
            state_init,
            t_schedule_jax,
            qoi_step_contribution=qoi_step_contribution,
        )
        return J, (first_failed_step, iters_per_step)

    return params_flat_init, state_init, cost


def build_fe_J_of_params_flat(
        bundle: FEProblemBundle,
        print_global_convergence: bool = False,
) -> tuple[
    JaxArray,
    StateInit,
    Callable[[JaxArray, StateInit, FEKernelArrays], JaxArray],
]:
    """``(params_flat_init, state_init, J_of_params_flat)`` for ``cmad
    objective`` / ``gradient`` / ``hessian``: :func:`build_fe_trajectory_cost`
    on the bundle's schedule, returning ``J`` alone."""
    params_flat_init, state_init, cost = build_fe_trajectory_cost(
        bundle, print_global_convergence,
    )
    t_schedule_jax = jnp.asarray(bundle.t_schedule, dtype=jnp.float64)

    def J_of_params_flat(
            params_flat: JaxArray,
            state_init: StateInit,
            fe_arrays: FEKernelArrays,
    ) -> JaxArray:
        return cost(params_flat, state_init, fe_arrays, t_schedule_jax)[0]

    return params_flat_init, state_init, J_of_params_flat


_DEFAULT_FE_PER_FAMILY: dict[ElementFamily, FiniteElement] = {
    ElementFamily.HEX_LINEAR: Q1_HEX,
    ElementFamily.TET_LINEAR: P1_TET,
    ElementFamily.QUAD_LINEAR: Q1_QUAD,
    ElementFamily.TRI_LINEAR: P1_TRI,
}

_FE_BY_NAME: dict[str, FiniteElement] = {
    "Q1": Q1_HEX,
    "Q1_HEX": Q1_HEX,
    "Q1_QUAD": Q1_QUAD,
    "P1": P1_TET,
    "P1_TET": P1_TET,
    "P1_TRI": P1_TRI,
}

_BC_COORD_NAMES: tuple[str, ...] = ("x", "y", "z", "t")


@dataclass(frozen=True)
class FEProblemBundle:
    resolved: dict[str, Any]
    fe_problem: FEProblem
    t_schedule: NDArray[np.float64]
    qoi: FEQoI | None = None


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
    ``supports_closed_form`` flag and threaded explicitly into
    :func:`build_fe_problem`.
    """
    deck = load_deck(deck_path)
    resolved = apply_deck_defaults(deck)
    validate_deck(resolved, subcommand)

    mesh_path = Path(resolved["discretization"]["mesh file"])
    mesh = read_mesh_file(mesh_path)
    if resolved["discretization"].get("build coordinate sidesets", False):
        built = coordinate_side_sets(mesh)
        clash = sorted(set(built) & set(mesh.side_sets))
        if clash:
            raise ValueError(
                "discretization.build coordinate sidesets would redefine "
                f"side set(s) already in the mesh: {clash}; remove the option "
                "or rename the existing set(s)"
            )
        mesh = replace(mesh, side_sets={**mesh.side_sets, **built})
    ndims = int(mesh.nodes.shape[1])

    gr_section = resolved["residuals"]["global residual"]
    gr_cls = resolve_global_residual(gr_section["type"])
    gr = gr_cls.from_deck(gr_section, ndims=ndims)

    is_mixed = bool(gr_section.get("mixed", False))
    ls_section = resolved["linear solver"]
    ls_type = ls_section["type"]
    precon_section = ls_section.get("preconditioner", {})
    precon_type = precon_section.get("type")
    if is_mixed and ls_type != "direct" and not (
        ls_type == "gmres" and precon_type == "block"
    ):
        raise ValueError(
            "residuals.global residual: mixed requires linear solver "
            "type 'direct', or 'gmres' with a 'block' preconditioner "
            f"(the mixed tangent is indefinite); got '{ls_type}'",
        )
    if ls_section.get("operator", "assembled") == "element":
        jax_native = (
            (ls_type in ("cg", "gmres") and precon_type == "jacobi")
            or (
                ls_type == "gmres" and precon_type == "block"
                and precon_section.get("inner", "jacobi") in ("jacobi", "chebyshev")
            )
        )
        if not jax_native:
            raise ValueError(
                "linear solver: operator 'element' needs a jax native "
                "solver, cg + jacobi, gmres + jacobi, or gmres + block with "
                "a jacobi or chebyshev inner solve; got "
                f"'{ls_type}' with preconditioner '{precon_type}'",
            )

    def_type = DefType[gr_section["def_type"].upper()]
    local_section = resolved["residuals"]["local residual"]
    models_by_block = _build_models_by_block(local_section, mesh, def_type)
    modes_by_block = {
        block: (
            GlobalResidualMode.CLOSED_FORM
            if model.supports_closed_form
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

    t_schedule = _load_t_schedule(resolved["discretization"])
    dirichlet_bcs = _build_dirichlet_bcs(
        resolved.get("dirichlet bcs"), gr, mesh, field_layouts,
        t_schedule,
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
    if is_mixed:
        quad_section = resolved["discretization"].get("quadrature") or {}
        vol_deg = quad_section.get("volume degree")
        if vol_deg is not None and int(vol_deg) < 2:
            raise ValueError(
                "residuals.global residual: mixed requires volume "
                f"quadrature degree >= 2; got {vol_deg}",
            )
        if assembly_quadrature is None:
            assembly_quadrature = {
                mesh.element_family: _quad_rule(
                    mesh.element_family, "volume", 2),
            }

    local_newton_settings = {
        "max_iters": int(local_section["nonlinear max iters"]),
        "abs_tol": float(local_section["nonlinear absolute tol"]),
        "rel_tol": float(local_section["nonlinear relative tol"]),
        "line_search_settings": local_section.get("line search", {}),
    }

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
        print_local_convergence=bool(
            local_section.get("print convergence", False),
        ),
        local_newton_settings=local_newton_settings,
        thickness=resolved["discretization"].get("thickness"),
    )

    qoi: FEQoI | None = None
    if "qoi" in resolved:
        qoi_cls = resolve_qoi(resolved["qoi"]["name"])
        if qoi_cls.problem_type != "fe":
            raise ValueError(
                f"qoi.name '{resolved['qoi']['name']}' is registered "
                f"for problem_type='{qoi_cls.problem_type}', but the "
                f"deck has problem.type='fe'"
            )
        assert issubclass(qoi_cls, FEQoI)
        qoi = qoi_cls.from_deck(
            resolved["qoi"], fe_problem, t_schedule.tolist(),
        )

    return FEProblemBundle(
        resolved=resolved, fe_problem=fe_problem, t_schedule=t_schedule,
        qoi=qoi,
    )


def _build_field_layouts(
        disc_section: dict[str, Any],
        gr: GlobalResidual,
        family: ElementFamily,
) -> list[GlobalFieldLayout]:
    """One layout per GR residual block, FE looked up per ``var_name``.

    Per-var FE choice comes from ``discretization.finite elements``
    (deck-side discretization decision); omitted var_names fall back to
    family-matched linear Lagrange (Q1_HEX / P1_TET in 3D, Q1_QUAD /
    P1_TRI in 2D). Stray override keys that don't match any GR var_name
    raise — silent typos in the deck would otherwise apply nothing.
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
        local_section: dict[str, Any], mesh: Any, def_type: int,
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
            local_section,
            build_parameters(
                _with_material_defaults(materials[block], model_cls),
            ),
            def_type,
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
        dbc_section: dict[str, Any] | None,
        gr: GlobalResidual,
        mesh: Mesh,
        field_layouts: Sequence[GlobalFieldLayout],
        t_schedule: NDArray[np.float64],
) -> list[DirichletBC]:
    if not dbc_section:
        return []
    bcs: list[DirichletBC] = []
    for entry_name, entry in dbc_section.get("expression", {}).items():
        resid_name, eq, sideset, value_expr = entry
        where = f"dirichlet bcs.expression.{entry_name}"
        r = _resolve_resid_idx(resid_name, gr, where)
        _check_bc_eq(eq, r, resid_name, gr, where)
        scalar_fn = parse_scalar_expression(value_expr, _BC_COORD_NAMES)
        bcs.append(DirichletBC(
            sideset_names=[str(sideset)],
            field_name=str(gr.var_names[r]),
            dofs=[int(eq)],
            values=_make_dbc_value_callable(scalar_fn),
        ))

    fe_by_field = {fl.name: fl.finite_element for fl in field_layouts}
    field_entries = dbc_section.get("field", {})
    data_file = dbc_section.get("field data file")
    if field_entries and data_file is None:
        raise KeyError(
            "dirichlet bcs: 'field data file' is required when 'field' "
            "conditions are given; every component reads the same measured "
            "field, so the file is named once",
        )
    # The file is either a calibration data archive, which carries every
    # measured frame on the sideset nodes, or a plain nodal field with
    # one frame per schedule entry.
    store: CalibrationData | None = None
    data: NDArray[np.float64] | None = None
    if field_entries and is_calibration_data(str(data_file)):
        store = CalibrationData.read(str(data_file))
        _check_calibration_bc_data(store, t_schedule, mesh, str(data_file))
    elif field_entries:
        data = np.asarray(
            load_displacement_data({"data_file": str(data_file)}),
            dtype=np.float64,
        )
    for entry_name, entry in field_entries.items():
        resid_name, eq, sideset = entry
        where = f"dirichlet bcs.field.{entry_name}"
        r = _resolve_resid_idx(resid_name, gr, where)
        _check_bc_eq(eq, r, resid_name, gr, where)
        field_name = str(gr.var_names[r])
        if str(sideset) not in mesh.side_sets:
            raise KeyError(
                f"{where}: unknown sideset '{sideset}'; known sidesets: "
                f"{sorted(mesh.side_sets)}",
            )
        node_ids = sideset_basis_fns(
            mesh, fe_by_field[field_name], [str(sideset)],
        )
        if store is not None:
            if int(eq) >= store.num_components:
                raise ValueError(
                    f"{where}: eq {eq} exceeds the {store.num_components} "
                    f"components in '{data_file}'",
                )
            values_by_step = store.rows(
                np.arange(store.num_frames), node_ids,
            )[:, :, int(eq):int(eq) + 1]
            data_times = store.times
        else:
            assert data is not None
            _check_field_bc_data(
                data, t_schedule, mesh, eq, str(data_file), where,
            )
            values_by_step = data[:, node_ids, int(eq):int(eq) + 1]
            data_times = t_schedule
        bcs.append(DirichletBC(
            sideset_names=[str(sideset)],
            field_name=field_name,
            dofs=[int(eq)],
            values=make_nodal_field_values(values_by_step, data_times),
        ))
    return bcs


def calibration_data_times(
        resolved: dict[str, Any],
) -> NDArray[np.float64] | None:
    """The frame times of the archive named by ``dirichlet bcs.field data
    file``, ``None`` when the file is not one."""
    path = (resolved.get("dirichlet bcs") or {}).get("field data file")
    if path is None or not is_calibration_data(str(path)):
        return None
    return CalibrationData.read(str(path)).times


def _check_calibration_bc_data(
        store: CalibrationData,
        t_schedule: NDArray[np.float64],
        mesh: Mesh,
        data_file: str,
) -> None:
    """Reject an archive built for another mesh or not covering the schedule."""
    store.check_mesh(int(mesh.nodes.shape[0]))
    tol = 1.0e-8 * float(store.times[-1] - store.times[0])
    if (
        float(t_schedule[0]) < float(store.times[0]) - tol
        or float(t_schedule[-1]) > float(store.times[-1]) + tol
    ):
        raise ValueError(
            f"dirichlet bcs.field data file: the schedule spans "
            f"{float(t_schedule[0]):g} to {float(t_schedule[-1]):g} but "
            f"'{data_file}' covers {float(store.times[0]):g} to "
            f"{float(store.times[-1]):g}",
        )


def _check_bc_eq(
        eq: int, r: int, resid_name: str, gr: GlobalResidual, where: str,
) -> None:
    num_eqs = int(gr._num_eqs[r])
    if not (0 <= int(eq) < num_eqs):
        raise ValueError(
            f"{where}: eq {eq} out of range for residual "
            f"'{resid_name}' (num_eqs={num_eqs})",
        )


def _check_field_bc_data(
        data: NDArray[np.float64],
        t_schedule: NDArray[np.float64],
        mesh: Mesh,
        eq: int,
        data_file: str,
        where: str,
) -> None:
    """Reject nodal field data that cannot line up with mesh and schedule."""
    if data.ndim != 3:
        raise ValueError(
            f"{where}: '{data_file}' has shape {data.shape}; expected "
            f"(num_frames, num_nodes, num_components)",
        )
    if data.shape[0] != t_schedule.shape[0]:
        raise ValueError(
            f"{where}: '{data_file}' has {data.shape[0]} frames but the "
            f"time schedule has {t_schedule.shape[0]} entries; one frame "
            f"per entry is required, the first being the reference state",
        )
    num_nodes = int(mesh.nodes.shape[0])
    if data.shape[1] != num_nodes:
        raise ValueError(
            f"{where}: '{data_file}' covers {data.shape[1]} nodes but the "
            f"mesh has {num_nodes}",
        )
    if int(eq) >= data.shape[2]:
        raise ValueError(
            f"{where}: eq {eq} exceeds the {data.shape[2]} components in "
            f"'{data_file}'",
        )


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
    [NDArray[np.floating] | JaxArray, Scalar],
    NDArray[np.floating] | JaxArray,
]]:
    if not body_section:
        return {}
    fns_by_idx: dict[int, Callable[
        [NDArray[np.floating] | JaxArray, Scalar],
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
) -> Callable[
    [NDArray[np.floating] | JaxArray, Scalar],
    JaxArray,
]:
    """Wrap a scalar ``f(x, y, z, t)`` into a DBC ``(coords, t) ->
    (N_set, 1)`` callable.

    Inputs may be JAX tracers (the traced quasi-static driver in
    :mod:`cmad.fem.driver` evaluates DBC values inside ``lax.scan``
    where ``t`` is traced), so the closure stays in ``jax.numpy``
    and broadcasts constants up to the BC's vertex count.
    """
    def fn(
            coords: NDArray[np.floating] | JaxArray,
            t: Scalar,
    ) -> JaxArray:
        n_set = coords.shape[0]
        val = jnp.asarray(scalar_fn(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], t=t,
        ), dtype=jnp.float64)
        return jnp.broadcast_to(val, (n_set,)).reshape(n_set, 1)
    return fn


def _make_nbc_value_callable(
        component_fns: list[Callable[..., Any]],
) -> Callable[
    [NDArray[np.floating] | JaxArray, Scalar],
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
            coords_ip: NDArray[np.floating] | JaxArray,
            t: Scalar,
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
    [NDArray[np.floating] | JaxArray, Scalar],
    JaxArray,
]:
    """Stack per-component scalar callables into a forcing
    ``(coords_ip, t) -> (num_eqs,)`` callable.

    ``coords_ip`` is the per-IP physical coord ``(ndims,)`` (single
    point in, vector out). The assembly layer vmap-traces this, so the
    closure stays in ``jax.numpy``.
    """
    def fn(
            coords_ip: NDArray[np.floating] | JaxArray,
            t: Scalar,
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
        disc_section: dict[str, Any],
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
