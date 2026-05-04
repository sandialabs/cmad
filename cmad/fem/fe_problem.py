"""FEProblem + FEState dataclasses for the FE forward problem."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from jax.flatten_util import ravel_pytree
from numpy.typing import NDArray

from cmad.fem.bcs import NeumannBC
from cmad.fem.dof import GlobalDofMap, GlobalFieldLayout
from cmad.fem.element_family import ElementFamily
from cmad.fem.mesh import Mesh
from cmad.fem.neumann import ResolvedNeumannBC, resolve_neumann_bcs
from cmad.fem.precompute import (
    BlockIPGeometryCache,
    precompute_block_geometry,
)
from cmad.fem.quadrature import (
    QuadratureRule,
    hex_quadrature,
    quad_quadrature,
    tet_quadrature,
    tri_quadrature,
)
from cmad.global_residuals.global_residual import GlobalResidual
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.model import Model
from cmad.typing import GREvaluators, JaxArray, StateList

_DEFAULT_ASSEMBLY_QUADRATURE: dict[ElementFamily, QuadratureRule] = {
    ElementFamily.HEX_LINEAR: hex_quadrature(degree=2),
    ElementFamily.TET_LINEAR: tet_quadrature(degree=1),
}

_DEFAULT_SIDE_QUADRATURE: dict[ElementFamily, QuadratureRule] = {
    ElementFamily.HEX_LINEAR: quad_quadrature(degree=2),
    ElementFamily.TET_LINEAR: tri_quadrature(degree=2),
}


@dataclass(frozen=True)
class FEProblem:
    """Immutable durable state for an FE forward problem.

    Sibling to ``MPProblem`` in ``cmad/cli/common.py`` — same
    builder/runner separation. CLI dispatch wraps :func:`build_fe_problem`
    and the assembly+solve operations without modifying this
    dataclass.

    The ``evaluators_by_block`` dict is the (block_name → evaluator-
    dict) result of binding ``gr.for_model(models_by_block[block],
    mode=modes_by_block[block])`` per element block, so each
    (GR, Model, mode) triple compiles once at construction.

    ``forcing_fns_by_block_idx`` is a sparse per-residual-block dict
    of body-force / source-term callables. Absence of a residual-
    block index means no forcing on that block (e.g. the pressure
    residual of a mixed u-p method, or a divergence-free thermal
    problem). Each callable returns a vector of shape
    ``(num_eqs[block_idx],)``.

    ``neumann_bcs`` is the list of natural-BC (surface-flux)
    declarations applied via the per-side evaluator in
    :mod:`cmad.fem.neumann`. ``side_quadrature`` is the per-family
    quadrature rule consumed by that evaluator (defaults
    degree-2 quad / tri for hex / tet faces). NBCs resolve at
    construction into ``resolved_neumann_bcs`` — per-NBC
    ``(family, local_side_id) -> elem_ids`` groups plus
    materialized values — which the surface-scatter step in
    :func:`cmad.fem.assembly.assemble_global` consumes per call.

    ``field_layouts_per_block`` and ``field_idx_per_block`` resolve
    the ``gr.var_names[r]`` ↔ ``dof_map.field_layouts`` dispatch once
    at construction (parallel to ``gr.var_names``, length
    ``gr.num_residuals``). Consumers — assembly, the adjoint
    builder — read directly from these instead of rebuilding the
    name → layout dict on every call. Population happens in
    :meth:`__post_init__` and raises ``ValueError`` on a missing or
    unmatched ``var_name``.

    ``unravel_xi_by_block`` caches each COUPLED block's
    ``ravel_pytree(model._init_xi)[1]`` unravel callable, which
    bridges the FEState's flat-trailing per-IP xi storage and the
    pytree-keyed ``StateList`` consumed by ``model._residual``'s
    block-by-block contract. Built once in :meth:`__post_init__` so
    the COUPLED kernel's per-element vmap doesn't re-derive the
    pytree treedef every Newton iteration. CLOSED_FORM blocks have
    no entry; the dict is empty for CLOSED_FORM-only problems.

    ``geometry_cache`` holds per-element-block reference-frame geometry
    (``iso_jac_det``, physical-frame field-shape gradients, IP
    coordinates, and the shared per-IP shape values + quadrature
    weights). Computed once at construction by
    :func:`cmad.fem.precompute.precompute_block_geometry` and consumed
    by the FE assembly kernels in lieu of recomputing from
    interpolants on every call. For total-Lagrangian formulations
    (cmad's current pipeline) this geometry is solution-independent;
    updated-Lagrangian kernels would bypass the cache.
    """
    mesh: Mesh
    dof_map: GlobalDofMap
    gr: GlobalResidual
    models_by_block: dict[str, Model]
    modes_by_block: dict[str, GlobalResidualMode]
    evaluators_by_block: dict[str, GREvaluators]
    forcing_fns_by_block_idx: dict[int, Callable[
        [NDArray[np.floating] | JaxArray, float],
        NDArray[np.floating] | JaxArray,
    ]] | None
    assembly_quadrature: dict[ElementFamily, QuadratureRule]
    neumann_bcs: Sequence[NeumannBC]
    side_quadrature: dict[ElementFamily, QuadratureRule]

    field_layouts_per_block: list[GlobalFieldLayout] = field(
        init=False, default_factory=list,
    )
    field_idx_per_block: list[int] = field(
        init=False, default_factory=list,
    )
    resolved_neumann_bcs: list[ResolvedNeumannBC] = field(
        init=False, default_factory=list,
    )
    unravel_xi_by_block: dict[str, Callable[[JaxArray], StateList]] = field(
        init=False, default_factory=dict,
    )
    geometry_cache: dict[str, BlockIPGeometryCache] = field(
        init=False, default_factory=dict,
    )

    def __post_init__(self) -> None:
        name_to_idx = {
            fl.name: i for i, fl in enumerate(self.dof_map.field_layouts)
        }
        layouts: list[GlobalFieldLayout] = []
        idxs: list[int] = []
        for r in range(self.gr.num_residuals):
            var_name = self.gr.var_names[r]
            if var_name is None:
                raise ValueError(
                    f"GR var_names[{r}] is None; fully-initialized GRs "
                    "must populate every var_names entry"
                )
            if var_name not in name_to_idx:
                raise ValueError(
                    f"GR var_names[{r}]='{var_name}' has no matching "
                    f"GlobalFieldLayout (known: {sorted(name_to_idx)})"
                )
            idx = name_to_idx[var_name]
            gr_neq = int(self.gr._num_eqs[r])
            dm_neq = int(self.dof_map.num_dofs_per_basis_fn[idx])
            if gr_neq != dm_neq:
                raise ValueError(
                    f"GR _num_eqs[{r}]={gr_neq} disagrees with "
                    f"dof_map.num_dofs_per_basis_fn[{idx}]={dm_neq} for "
                    f"field '{var_name}'; the components_by_field passed "
                    f"to build_dof_map must match gr._num_eqs"
                )
            idxs.append(idx)
            layouts.append(self.dof_map.field_layouts[idx])
        object.__setattr__(self, "field_layouts_per_block", layouts)
        object.__setattr__(self, "field_idx_per_block", idxs)

        resolved = resolve_neumann_bcs(
            self.mesh, self.dof_map, self.neumann_bcs,
        )
        object.__setattr__(self, "resolved_neumann_bcs", resolved)

        unravel_xi_by_block: dict[
            str, Callable[[JaxArray], StateList],
        ] = {}
        for block, mode in self.modes_by_block.items():
            if mode == GlobalResidualMode.COUPLED:
                model = self.models_by_block[block]
                unravel_xi_by_block[block] = ravel_pytree(model._init_xi)[1]
        object.__setattr__(
            self, "unravel_xi_by_block", unravel_xi_by_block,
        )

        geometry_cache = precompute_block_geometry(
            self.mesh, self.assembly_quadrature, layouts,
        )
        object.__setattr__(self, "geometry_cache", geometry_cache)

    @property
    def ndims(self) -> int:
        return int(self.mesh.nodes.shape[1])

    @property
    def block_shapes(self) -> list[tuple[int, int]]:
        """Per-residual-block ``(num_basis_fns, num_eqs)`` tuples.

        ``num_basis_fns`` is the per-element basis-fn count of the
        field bound to residual block ``r``, sourced from
        ``field_layouts_per_block[r].finite_element.num_dofs_per_element``
        (resolved in :meth:`__post_init__`). ``num_eqs`` is
        ``gr._num_eqs[r]``. Together they shape the per-element
        residual and tangent buffers the assembly layer allocates.
        """
        return [
            (
                self.field_layouts_per_block[r]
                .finite_element.num_dofs_per_element,
                self.gr._num_eqs[r],
            )
            for r in range(self.gr.num_residuals)
        ]


@dataclass
class FEState:
    """Time-indexed mutable state companion to :class:`FEProblem`.

    Holds the U history (full nodal displacement at each step), per-
    block xi history (per-element per-IP state arrays), and t history.
    The driver builds an FEState via :meth:`from_problem` before the
    time loop, appends one ``(U, xi_by_block, t)`` tuple per converged
    step via :meth:`append`, and the writer / QoI / adjoint stages
    consume the resulting trajectory.

    Single-step quasi-static use: ``U_history`` holds
    ``[U_init=0, U_solved]`` after one ``fe_newton_solve`` call;
    multi-step use extends additively.

    Mutable-by-design: long simulations rebuild less if history grows
    in place. Initialization-time content (zeros U, per-block xi tiled
    from ``model._init_xi``) is reproducible from FEProblem alone.
    """
    U_history: list[NDArray[np.floating]]
    xi_history_by_block: dict[str, list[NDArray[np.floating]]]
    t_history: list[float]

    @classmethod
    def from_problem(
            cls,
            fe_problem: FEProblem,
            t_init: float = 0.0,
            U_init: NDArray[np.floating] | None = None,
    ) -> "FEState":
        """Build an FEState seeded with the initial step.

        ``U_init`` defaults to zeros over all dofs. xi history seeds
        from each Model's ``_init_xi`` tiled across the block's
        elements and IPs.
        """
        n_dofs = fe_problem.dof_map.num_total_dofs
        U_init_arr = (
            np.zeros(n_dofs, dtype=np.float64)
            if U_init is None else U_init.copy()
        )
        xi_init_by_block: dict[str, list[NDArray[np.floating]]] = {}
        for block, model in fe_problem.models_by_block.items():
            elem_indices = fe_problem.mesh.element_blocks[block]
            n_elems = len(elem_indices)
            quad_rule = fe_problem.assembly_quadrature[
                fe_problem.mesh.element_family
            ]
            n_ips = int(quad_rule.xi.shape[0])
            init_xi_flat = np.concatenate(
                [np.asarray(b) for b in model._init_xi],
            )
            xi_init = np.tile(init_xi_flat, (n_elems, n_ips, 1))
            xi_init_by_block[block] = [xi_init]
        return cls(
            U_history=[U_init_arr],
            xi_history_by_block=xi_init_by_block,
            t_history=[float(t_init)],
        )

    def append(
            self,
            U_new: NDArray[np.floating],
            xi_by_block: dict[str, NDArray[np.floating]],
            t_new: float,
    ) -> None:
        """Append one converged step's ``(U, xi_by_block, t)`` tuple."""
        self.U_history.append(np.asarray(U_new).copy())
        for block, xi in xi_by_block.items():
            self.xi_history_by_block[block].append(np.asarray(xi).copy())
        self.t_history.append(float(t_new))

    @property
    def step_idx(self) -> int:
        """Index of the most recently appended step (0 = initial)."""
        return len(self.U_history) - 1

    def U_at(self, step: int) -> NDArray[np.floating]:
        return self.U_history[step]

    def xi_at(self, step: int, block: str) -> NDArray[np.floating]:
        return self.xi_history_by_block[block][step]


def build_fe_problem(
        mesh: Mesh,
        dof_map: GlobalDofMap,
        gr: GlobalResidual,
        models_by_block: dict[str, Model],
        modes_by_block: dict[str, GlobalResidualMode] | None = None,
        forcing_fns_by_block_idx: dict[int, Callable[
            [NDArray[np.floating] | JaxArray, float],
            NDArray[np.floating] | JaxArray,
        ]] | None = None,
        assembly_quadrature: dict[ElementFamily, QuadratureRule] | None = None,
        neumann_bcs: Sequence[NeumannBC] = (),
        side_quadrature: dict[ElementFamily, QuadratureRule] | None = None,
) -> FEProblem:
    """Validate FE inputs and build an immutable :class:`FEProblem`.

    Element-block names in ``mesh.element_blocks`` must match the keys
    in ``models_by_block`` and ``modes_by_block`` (when supplied); the
    builder raises ``ValueError`` on mismatch. ``modes_by_block``
    defaults to all-``CLOSED_FORM``. ``assembly_quadrature``
    defaults to a per-family table with degree-2 hex Gauss-Legendre
    and 1-pt tet rules. ``side_quadrature`` defaults to degree-2
    quad / tri rules for hex / tet face integration consumed by
    :mod:`cmad.fem.neumann`.

    Each (block, model, mode) triple is bound via
    ``gr.for_model(model, mode=mode)`` at construction so the
    per-block evaluator dicts compile once.

    ``forcing_fns_by_block_idx`` keys must lie in ``[0, gr.num_residuals)``;
    the builder also probes each callable at the origin (best-effort)
    and verifies the returned shape matches ``(gr._num_eqs[block_idx],)``
    so common mistakes (wrong vector length, wrong block index) surface
    eagerly rather than at jit trace time. Probes that fail to evaluate
    outside a jit context are silently skipped — the trace-time error
    will still catch the shape mismatch.

    ``neumann_bcs`` is forwarded to :func:`FEProblem` for resolution
    in :meth:`__post_init__`; resolution failures (unknown field /
    sideset, non-VERTEX FE, sequence-values length mismatch) raise
    eagerly with diagnostic messages from
    :func:`cmad.fem.neumann.resolve_neumann_bcs`.
    """
    if modes_by_block is None:
        modes_by_block = {
            b: GlobalResidualMode.CLOSED_FORM for b in models_by_block
        }
    if assembly_quadrature is None:
        assembly_quadrature = dict(_DEFAULT_ASSEMBLY_QUADRATURE)
    if side_quadrature is None:
        side_quadrature = dict(_DEFAULT_SIDE_QUADRATURE)

    block_names_mesh = set(mesh.element_blocks.keys())
    block_names_models = set(models_by_block.keys())
    if block_names_mesh != block_names_models:
        raise ValueError(
            f"models_by_block keys ({sorted(block_names_models)}) must "
            f"match mesh.element_blocks keys ({sorted(block_names_mesh)})"
        )
    if set(modes_by_block.keys()) != block_names_models:
        raise ValueError(
            f"modes_by_block keys ({sorted(modes_by_block.keys())}) must "
            f"match models_by_block keys ({sorted(block_names_models)})"
        )

    if forcing_fns_by_block_idx is not None:
        num_blocks = gr.num_residuals
        ndims = int(mesh.nodes.shape[1])
        for block_idx, fn in forcing_fns_by_block_idx.items():
            if not (0 <= block_idx < num_blocks):
                raise ValueError(
                    f"forcing_fns_by_block_idx has block_idx={block_idx} "
                    f"out of range [0, {num_blocks}); GR has "
                    f"{num_blocks} residual blocks"
                )
            try:
                probe = np.asarray(fn(np.zeros(ndims), 0.0))
            except Exception:
                continue
            expected = (int(gr._num_eqs[block_idx]),)
            if probe.shape != expected:
                raise ValueError(
                    f"forcing_fns_by_block_idx[{block_idx}] returned "
                    f"shape {probe.shape}; expected {expected} "
                    f"(gr._num_eqs[{block_idx}])"
                )

    evaluators_by_block: dict[str, GREvaluators] = {
        b: gr.for_model(models_by_block[b], mode=modes_by_block[b])
        for b in models_by_block
    }

    return FEProblem(
        mesh=mesh,
        dof_map=dof_map,
        gr=gr,
        models_by_block=models_by_block,
        modes_by_block=modes_by_block,
        evaluators_by_block=evaluators_by_block,
        forcing_fns_by_block_idx=forcing_fns_by_block_idx,
        assembly_quadrature=assembly_quadrature,
        neumann_bcs=neumann_bcs,
        side_quadrature=side_quadrature,
    )
