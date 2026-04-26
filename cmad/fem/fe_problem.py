"""FEProblem + FEState dataclasses for the FE forward problem."""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cmad.fem.dof import GlobalDofMap
from cmad.fem.interpolants import hex_linear, tet_linear
from cmad.fem.mesh import ElementFamily, Mesh
from cmad.fem.quadrature import QuadratureRule, hex_quadrature, tet_quadrature
from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.global_residuals.global_residual import GlobalResidual
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.model import Model
from cmad.typing import JaxArray, PyTree

_DEFAULT_INTERPOLANT_FN: dict[
    ElementFamily, Callable[[JaxArray], ShapeFunctionsAtIP],
] = {
    ElementFamily.HEX_LINEAR: hex_linear,
    ElementFamily.TET_LINEAR: tet_linear,
}

_DEFAULT_ASSEMBLY_QUADRATURE: dict[ElementFamily, QuadratureRule] = {
    ElementFamily.HEX_LINEAR: hex_quadrature(degree=2),
    ElementFamily.TET_LINEAR: tet_quadrature(degree=1),
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
    ``(num_eqs[block_idx],)``. Surface tractions / Neumann BCs are
    not handled here — see future per-face evaluator design.
    """
    mesh: Mesh
    dof_map: GlobalDofMap
    gr: GlobalResidual
    models_by_block: dict[str, Model]
    modes_by_block: dict[str, GlobalResidualMode]
    evaluators_by_block: dict[str, dict[str, Callable[..., PyTree]]]
    forcing_fns_by_block_idx: dict[int, Callable[
        [NDArray[np.floating] | JaxArray, float],
        NDArray[np.floating] | JaxArray,
    ]] | None
    assembly_quadrature: dict[ElementFamily, QuadratureRule]
    interpolant_fn: dict[
        ElementFamily, Callable[[JaxArray], ShapeFunctionsAtIP],
    ]

    @property
    def ndims(self) -> int:
        return int(self.mesh.nodes.shape[1])


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
        interpolant_fn: dict[
            ElementFamily, Callable[[JaxArray], ShapeFunctionsAtIP],
        ] | None = None,
) -> FEProblem:
    """Validate FE inputs and build an immutable :class:`FEProblem`.

    Element-block names in ``mesh.element_blocks`` must match the keys
    in ``models_by_block`` and ``modes_by_block`` (when supplied); the
    builder raises ``ValueError`` on mismatch. ``modes_by_block``
    defaults to all-``CLOSED_FORM``. ``assembly_quadrature`` and
    ``interpolant_fn`` default to a hex / tet table using degree-2
    hex Gauss-Legendre and 1-pt tet rules.

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
    """
    if modes_by_block is None:
        modes_by_block = {
            b: GlobalResidualMode.CLOSED_FORM for b in models_by_block
        }
    if assembly_quadrature is None:
        assembly_quadrature = dict(_DEFAULT_ASSEMBLY_QUADRATURE)
    if interpolant_fn is None:
        interpolant_fn = dict(_DEFAULT_INTERPOLANT_FN)

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

    evaluators_by_block: dict[str, dict[str, Callable[..., PyTree]]] = {
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
        interpolant_fn=interpolant_fn,
    )
