"""JAX-traced FE forward solve: ``jax.grad`` vs central-difference smoke tests.

Five tests verifying ``jax.grad(J)(p)`` matches central-difference FD on
a JAX-traced FE primal that mirrors the production data flow:

- Per-element kernels: ``per_element_R_and_K`` /
  ``per_element_R_and_K_coupled`` reused verbatim (already pure-JAX,
  per-IP/per-element AD-tested in ``tests/fem/test_per_element_coupled``).
- Per-IP local Newton (COUPLED): ``make_newton_solve`` wired via
  ``GR.for_model(model, COUPLED)``; ``@custom_jvp`` IFT rule unchanged.
- Strong DBC enforcement: row/col zero + scaled-identity on the
  diagonal + ``R[prescribed] = 0``, expressed in JAX (mirrors
  ``cmad.fem.nonlinear_solver.apply_strong_dirichlet`` step-for-step).
- Outer Newton: ``lax.custom_root`` — opaque-to-AD imperative forward
  via ``lax.while_loop``; ``tangent_solve`` provides IFT-correct
  gradients via dense ``jnp.linalg.solve`` on the converged enforced
  Jacobian. (Production solves with ``scipy.sparse.linalg.spsolve``;
  a JAX-side equivalent can swap that in via ``jax.pure_callback``
  later — same matvec contract, different backend.)

The tests are arranged as boundary-isolation diagnostics: each adds
one new AD boundary on top of the previous, so a failure localizes
which boundary broke. The final test is the everything-composes
capstone — if individual boundaries pass and the capstone fails, the
cross-paths between boundaries are wrong, not the boundaries themselves.
"""
import unittest
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from jax.tree_util import tree_map

from cmad.fem.assembly import (
    _element_eq_indices,
    _gather_element_U,
    per_element_R_and_K,
    per_element_R_and_K_coupled,
)
from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.parameters.parameters import Parameters
from cmad.typing import PyTreeDict
from tests.support.test_problems import J2AnalyticalProblem


# ============================================================
# Module-level fixtures (cribbed from existing 2x2x2 tests)
# ============================================================

def _uniaxial_dbcs(slope: float) -> list[DirichletBC]:
    """Symmetry pins on -x/-y/-z faces, t-ramped u_x on +x face.

    Cribbed from ``tests/fem/test_fe_quasistatic_drive.py:_uniaxial_dbcs``.
    """
    def u_x_at_t(coords, t):
        return jnp.full((coords.shape[0], 1), slope * t)
    return [
        DirichletBC(sideset_names=["xmin_sides"], field_name="u",
                    dofs=(0,), values=None),
        DirichletBC(sideset_names=["ymin_sides"], field_name="u",
                    dofs=(1,), values=None),
        DirichletBC(sideset_names=["zmin_sides"], field_name="u",
                    dofs=(2,), values=None),
        DirichletBC(sideset_names=["xmax_sides"], field_name="u",
                    dofs=(0,), values=u_x_at_t),
    ]


def _make_elastic_model(kappa: float = 100.0, mu: float = 50.0) -> Elastic:
    values = cast(PyTreeDict,
                  {"elastic": {"kappa": float(kappa), "mu": float(mu)}})
    active = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Elastic(Parameters(values, active, transforms),
                   def_type=DefType.FULL_3D)


def _make_J2_model() -> SmallElasticPlastic:
    """SmallElasticPlastic with the J2-Voce parameter set from
    ``J2AnalyticalProblem`` (E=200e3, nu=0.3, Y=200, Voce S=200,
    D=20). Yield strain ``Y/E = 1e-3`` sets the load-magnitude
    reference for the plastic regime."""
    return SmallElasticPlastic(
        J2AnalyticalProblem().J2_parameters,
        def_type=DefType.FULL_3D,
    )


def _build_fe_problem_2x2x2(model, mode: GlobalResidualMode, slope: float):
    """2x2x2 hex (8 elements, 27 nodes, 81 DOFs) with uniaxial DBCs."""
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    dof_map = build_dof_map(
        mesh, [layout], _uniaxial_dbcs(slope),
        components_by_field={"u": 3},
    )
    gr = SmallDispEquilibrium(ndims=3)
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": mode},
    )


# ============================================================
# JAX-traced global assembly + strong-DBC enforcement
# ============================================================

def _assemble_R_K_closed_jax(fe_problem, params, U, U_prev, t):
    """Dense JAX mirror of ``assemble_global`` for CLOSED_FORM blocks.

    Vmaps the production ``per_element_R_and_K`` per element, then
    scatters into dense ``(n_dofs,)`` R and ``(n_dofs, n_dofs)`` K via
    ``jnp.zeros(...).at[idx].add(...)`` (no scipy.sparse, no
    ``np.add.at``). Body forces from ``forcing_fns_by_block_idx``
    accumulate inside the per-element kernel as in production.
    """
    dof_map = fe_problem.dof_map
    mesh = fe_problem.mesh
    block_shapes = fe_problem.block_shapes
    n_dofs = dof_map.num_total_dofs

    R_global = jnp.zeros(n_dofs)
    K_global = jnp.zeros((n_dofs, n_dofs))
    forcing_fns = fe_problem.forcing_fns_by_block_idx or {}

    for block_name in fe_problem.evaluators_by_block:
        elem_indices = mesh.element_blocks[block_name]
        connectivity_block = mesh.connectivity[elem_indices]
        U_elem = _gather_element_U(U, dof_map, connectivity_block)
        U_prev_elem = _gather_element_U(U_prev, dof_map, connectivity_block)
        evaluators = fe_problem.evaluators_by_block[block_name]
        geom_cache = fe_problem.geometry_cache[block_name]

        R_per_elem, K_per_elem = vmap(
            lambda U_e, Up_e, geom: per_element_R_and_K(
                U_e, Up_e, params, geom, geom_cache.shared,
                evaluators["R_and_dR_dU"],
                forcing_fns, block_shapes, t,
            ),
            in_axes=(0, 0, 0),
        )(U_elem, U_prev_elem, geom_cache.per_elem)

        eq_indices = jnp.asarray(_element_eq_indices(
            connectivity_block, dof_map, field_idx=0,
        ))
        n_elems, n_dofs_elem = eq_indices.shape

        # R: list[Array(n_elems, num_basis_fns, num_eqs)]; single block.
        R_flat = R_per_elem[0].reshape(n_elems, n_dofs_elem)
        R_global = R_global.at[eq_indices.ravel()].add(R_flat.ravel())

        # K: list[list[Array(n_elems, nb, neq, nb, neq)]]; single block.
        K_flat = K_per_elem[0][0].reshape(n_elems, n_dofs_elem, n_dofs_elem)
        rows_2d = jnp.broadcast_to(
            eq_indices[:, :, None],
            (n_elems, n_dofs_elem, n_dofs_elem),
        ).ravel()
        cols_2d = jnp.broadcast_to(
            eq_indices[:, None, :],
            (n_elems, n_dofs_elem, n_dofs_elem),
        ).ravel()
        K_global = K_global.at[rows_2d, cols_2d].add(K_flat.ravel())

    return R_global, K_global


def _free_diag_scale(K, prescribed_indices):
    """``mean(|diag(K)|[free])`` — production's spectrum-balancing scale.

    Used as the prescribed-row weight in two places that must agree:
    ``_apply_strong_dirichlet_jax`` (production-style enforcement for
    the forward solve) and the embedded-BC residual in
    ``_solve_step_closed.f`` (AD-visible). Computing both from the same
    ``K(U)`` keeps them numerically identical.
    """
    n = K.shape[0]
    p_mask = jnp.zeros(n, dtype=bool).at[prescribed_indices].set(True)
    free_mask = (~p_mask).astype(K.dtype)
    return (jnp.sum(jnp.abs(jnp.diag(K)) * free_mask)
            / jnp.sum(free_mask))


def _apply_strong_dirichlet_jax(K, R, prescribed_indices, scale):
    """JAX mirror of ``cmad.fem.nonlinear_solver.apply_strong_dirichlet``.

    Zeros prescribed rows + columns of ``K``, sets the prescribed
    diagonal entries to ``scale``, and zeros ``R[prescribed]``
    (homogeneous DBC residual since ``U`` is pre-loaded with BC values).
    """
    n = R.shape[0]
    p_mask = jnp.zeros(n, dtype=bool).at[prescribed_indices].set(True)
    K_enf = jnp.where(p_mask[:, None] | p_mask[None, :], 0.0, K)
    K_enf = K_enf.at[prescribed_indices, prescribed_indices].set(scale)
    R_enf = R.at[prescribed_indices].set(0.0)
    return K_enf, R_enf


# ============================================================
# JAX-traced single-step CLOSED_FORM forward solve
# ============================================================

def _solve_step_closed(fe_problem, params, U_prev, t):
    """One quasi-static step in CLOSED_FORM mode.

    Outer Newton via ``lax.custom_root``: forward pass is opaque to AD
    (imperative ``lax.while_loop`` over Newton iterations);
    ``tangent_solve`` provides IFT-correct gradients via dense
    ``jnp.linalg.solve`` on the converged enforced Jacobian.

    Mirrors ``cmad.fem.nonlinear_solver.fe_newton_solve`` step-for-step
    except for: dense JAX K assembly (vs scipy.sparse COO),
    ``jnp.linalg.solve`` (vs ``scipy.sparse.linalg.spsolve``), and
    ``lax.while_loop`` (vs Python ``for`` loop). All per-IP / per-element
    physics + strong-DBC enforcement is identical to production.
    """
    dof_map = fe_problem.dof_map
    n_dofs = dof_map.num_total_dofs
    prescribed_indices = jnp.asarray(dof_map.prescribed_indices)
    prescribed_vals = jnp.asarray(
        dof_map.evaluate_prescribed_values(t=float(t)),
    )

    max_iters = 20
    abs_tol = 1e-12
    rel_tol = 1e-12

    def f(U):
        """Embedded-BC residual.

        ``f[free] = R[free]``, ``f[prescribed] =
        scale * (U[prescribed] - prescribed_vals)``. The scale factor
        ``mean(|diag(K)|[free])`` matches production's
        ``apply_strong_dirichlet`` and keeps the prescribed block of
        ``df/dU`` non-singular (without it, ``df/dU`` has zero rows at
        prescribed dofs → singular → IFT yields NaN).
        """
        R, K = _assemble_R_K_closed_jax(
            fe_problem, params, U, U_prev, t,
        )
        scale = _free_diag_scale(K, prescribed_indices)
        return R.at[prescribed_indices].set(
            scale * (U[prescribed_indices] - prescribed_vals)
        )

    def solve(_f, U_init):
        def cond(state):
            i, _, R_norm, R_norm_0 = state
            return jnp.logical_and(
                i < max_iters,
                jnp.logical_and(
                    R_norm >= abs_tol,
                    R_norm >= rel_tol * R_norm_0,
                ),
            )

        def body(state):
            i, U, _, R_norm_0 = state
            R, K = _assemble_R_K_closed_jax(
                fe_problem, params, U, U_prev, t,
            )
            scale = _free_diag_scale(K, prescribed_indices)
            K_enf, R_enf = _apply_strong_dirichlet_jax(
                K, R, prescribed_indices, scale,
            )
            dU = jnp.linalg.solve(K_enf, -R_enf)
            U_new = U + dU
            return i + 1, U_new, jnp.linalg.norm(_f(U_new)), R_norm_0

        R0_norm = jnp.maximum(jnp.linalg.norm(_f(U_init)), 1.0)
        _, U_star, _, _ = lax.while_loop(
            cond, body, (0, U_init, R0_norm, R0_norm),
        )
        return U_star

    def tangent_solve(g, y):
        """Solve ``(df/dU)(U_star) @ x = y`` via dense ``jnp.linalg.solve``."""
        K_lin = jax.jacfwd(g)(jnp.zeros_like(y))
        return jnp.linalg.solve(K_lin, y)

    U_init = jnp.zeros(n_dofs).at[prescribed_indices].set(prescribed_vals)
    return lax.custom_root(f, U_init, solve, tangent_solve)


# ============================================================
# JAX-traced global assembly + single-step solve, COUPLED
# ============================================================

def _assemble_R_K_coupled_jax(
        fe_problem, params, U, U_prev, xi_prev_by_block, t,
):
    """Dense JAX mirror of ``assemble_global`` for COUPLED blocks.

    Vmaps the production ``per_element_R_and_K_coupled`` per element
    (which runs the per-IP local Newton via the
    ``R_and_dR_dU_and_xi`` evaluator from
    ``GR.for_model(model, COUPLED)``; ``make_newton_solve``'s
    ``@custom_jvp`` rule supplies the IFT-correct local tangent),
    then scatters into dense ``(n_dofs,)`` R and ``(n_dofs, n_dofs)``
    K. Also returns the converged xi per block, shape
    ``(n_elems_block, n_ips, total_xi_dofs)``.
    """
    dof_map = fe_problem.dof_map
    mesh = fe_problem.mesh
    block_shapes = fe_problem.block_shapes
    n_dofs = dof_map.num_total_dofs

    R_global = jnp.zeros(n_dofs)
    K_global = jnp.zeros((n_dofs, n_dofs))
    xi_solved_by_block: dict[str, jax.Array] = {}
    forcing_fns = fe_problem.forcing_fns_by_block_idx or {}

    for block_name in fe_problem.evaluators_by_block:
        elem_indices = mesh.element_blocks[block_name]
        connectivity_block = mesh.connectivity[elem_indices]
        U_elem = _gather_element_U(U, dof_map, connectivity_block)
        U_prev_elem = _gather_element_U(U_prev, dof_map, connectivity_block)
        evaluators = fe_problem.evaluators_by_block[block_name]
        geom_cache = fe_problem.geometry_cache[block_name]
        unravel_xi = fe_problem.unravel_xi_by_block[block_name]
        xi_prev_block = xi_prev_by_block[block_name]

        R_per_elem, K_per_elem, xi_per_elem = vmap(
            lambda U_e, Up_e, geom, xi_prev: per_element_R_and_K_coupled(
                U_e, Up_e, params, xi_prev,
                geom, geom_cache.shared,
                evaluators["R_and_dR_dU_and_xi"],
                unravel_xi,
                forcing_fns, block_shapes, t,
            ),
            in_axes=(0, 0, 0, 0),
        )(U_elem, U_prev_elem, geom_cache.per_elem, xi_prev_block)

        xi_solved_by_block[block_name] = xi_per_elem

        eq_indices = jnp.asarray(_element_eq_indices(
            connectivity_block, dof_map, field_idx=0,
        ))
        n_elems, n_dofs_elem = eq_indices.shape

        R_flat = R_per_elem[0].reshape(n_elems, n_dofs_elem)
        R_global = R_global.at[eq_indices.ravel()].add(R_flat.ravel())

        K_flat = K_per_elem[0][0].reshape(n_elems, n_dofs_elem, n_dofs_elem)
        rows_2d = jnp.broadcast_to(
            eq_indices[:, :, None],
            (n_elems, n_dofs_elem, n_dofs_elem),
        ).ravel()
        cols_2d = jnp.broadcast_to(
            eq_indices[:, None, :],
            (n_elems, n_dofs_elem, n_dofs_elem),
        ).ravel()
        K_global = K_global.at[rows_2d, cols_2d].add(K_flat.ravel())

    return R_global, K_global, xi_solved_by_block


def _solve_step_coupled(
        fe_problem, params, U_prev, xi_prev_by_block, t,
):
    """One quasi-static step in COUPLED mode.

    Same outer-Newton + ``lax.custom_root`` shape as
    ``_solve_step_closed``; the f closure here calls
    ``_assemble_R_K_coupled_jax``, which fires the inner per-IP
    local Newton (its ``@custom_jvp`` IFT rule contributes to the
    outer ``df/dU`` automatically under JAX tracing). Returns
    ``(U_star, xi_solved_by_block)`` — xi is re-extracted at
    ``U_star`` for path-continuity into the next step.
    """
    dof_map = fe_problem.dof_map
    n_dofs = dof_map.num_total_dofs
    prescribed_indices = jnp.asarray(dof_map.prescribed_indices)
    prescribed_vals = jnp.asarray(
        dof_map.evaluate_prescribed_values(t=float(t)),
    )

    max_iters = 30
    abs_tol = 1e-10
    rel_tol = 1e-10

    def f(U):
        # Prescribed-row scale is forward-solve conditioning only —
        # AD-irrelevant at convergence (the prescribed block of
        # ``df/dU`` is decoupled and prescribed values don't depend on
        # parameters, so ``dU*/dp`` is scale-independent). Hardcoding
        # ``s=1`` here keeps K out of f's output, so the production
        # kernel's internal ``dR_dU = jacfwd(coupled_r_total)`` branch
        # has zero cotangent and JAX skips its transpose synthesis.
        R, _, _ = _assemble_R_K_coupled_jax(
            fe_problem, params, U, U_prev, xi_prev_by_block, t,
        )
        return R.at[prescribed_indices].set(
            U[prescribed_indices] - prescribed_vals
        )

    def solve(_f, U_init):
        def cond(state):
            i, _, R_norm, R_norm_0 = state
            return jnp.logical_and(
                i < max_iters,
                jnp.logical_and(
                    R_norm >= abs_tol,
                    R_norm >= rel_tol * R_norm_0,
                ),
            )

        def body(state):
            i, U, _, R_norm_0 = state
            R, K, _ = _assemble_R_K_coupled_jax(
                fe_problem, params, U, U_prev, xi_prev_by_block, t,
            )
            scale = _free_diag_scale(K, prescribed_indices)
            K_enf, R_enf = _apply_strong_dirichlet_jax(
                K, R, prescribed_indices, scale,
            )
            dU = jnp.linalg.solve(K_enf, -R_enf)
            U_new = U + dU
            return i + 1, U_new, jnp.linalg.norm(_f(U_new)), R_norm_0

        R0_norm = jnp.maximum(jnp.linalg.norm(_f(U_init)), 1.0)
        _, U_star, _, _ = lax.while_loop(
            cond, body, (0, U_init, R0_norm, R0_norm),
        )
        return U_star

    def tangent_solve(g, y):
        K_lin = jax.jacfwd(g)(jnp.zeros_like(y))
        return jnp.linalg.solve(K_lin, y)

    U_init = jnp.zeros(n_dofs).at[prescribed_indices].set(prescribed_vals)
    U_star = lax.custom_root(f, U_init, solve, tangent_solve)

    _, _, xi_solved = _assemble_R_K_coupled_jax(
        fe_problem, params, U_star, U_prev, xi_prev_by_block, t,
    )
    return U_star, xi_solved


def _initial_xi_by_block(fe_problem) -> dict[str, jax.Array]:
    """Per-block xi at step 0: flatten ``model._init_xi`` and tile
    across ``(n_elems_block, n_ips, total_xi_dofs)`` — same pattern
    as ``FEState.from_problem``, expressed in JAX so it can flow
    into traced calls."""
    from jax.flatten_util import ravel_pytree
    xi_by_block: dict[str, jax.Array] = {}
    for block, model in fe_problem.models_by_block.items():
        elem_indices = fe_problem.mesh.element_blocks[block]
        n_elems = len(elem_indices)
        quad = fe_problem.assembly_quadrature[
            fe_problem.mesh.element_family
        ]
        n_ips = int(quad.xi.shape[0])
        init_xi_flat, _ = ravel_pytree(model._init_xi)
        xi_by_block[block] = jnp.tile(init_xi_flat, (n_elems, n_ips, 1))
    return xi_by_block


def _solve_history_coupled(fe_problem, params, ts):
    """Quasi-static time loop, COUPLED.

    Threads ``(U_prev, xi_prev_by_block)`` forward through
    ``len(ts)`` calls to ``_solve_step_coupled``. Returns parallel
    lists of solved displacements and converged xi-by-block (one
    entry per step).
    """
    n_dofs = fe_problem.dof_map.num_total_dofs
    U_history: list[jax.Array] = []
    xi_history: list[dict[str, jax.Array]] = []
    U_prev = jnp.zeros(n_dofs)
    xi_prev = _initial_xi_by_block(fe_problem)
    for t in ts:
        U_n, xi_n = _solve_step_coupled(
            fe_problem, params, U_prev, xi_prev, float(t),
        )
        U_history.append(U_n)
        xi_history.append(xi_n)
        U_prev = U_n
        xi_prev = xi_n
    return U_history, xi_history


def _solve_history_closed(fe_problem, params, ts):
    """Quasi-static time loop, CLOSED_FORM.

    Threads ``U_prev`` forward through ``len(ts)`` calls to
    ``_solve_step_closed``. Python ``for`` loop unrolls under JAX
    tracing — each step's ``lax.custom_root`` AD rule fires
    independently, and reverse-mode composes them by walking the
    history backward through standard JAX VJP rules.

    Returns a list of ``len(ts)`` solved displacement vectors.
    """
    n_dofs = fe_problem.dof_map.num_total_dofs
    U_history: list[jax.Array] = []
    U_prev = jnp.zeros(n_dofs)
    for t in ts:
        U_n = _solve_step_closed(fe_problem, params, U_prev, float(t))
        U_history.append(U_n)
        U_prev = U_n
    return U_history


# ============================================================
# CLOSED_FORM single step
# ============================================================

class TestClosedFormSingleStep(unittest.TestCase):
    """``jax.grad`` of a scalar QoI through a single-step CLOSED_FORM
    Elastic forward solve matches central-difference. Validates the
    outer ``lax.custom_root`` AD rule alone — no inner ``custom_jvp``
    invoked, no time history.
    """

    def test_grad_kappa_mu_matches_fd(self) -> None:
        slope = 5e-4
        t = 1.0

        # Build problem once with concrete (irrelevant for AD) params;
        # parameter dependence flows through the explicit ``params``
        # arg into ``_solve_step_closed``.
        fe_problem = _build_fe_problem_2x2x2(
            _make_elastic_model(),
            GlobalResidualMode.CLOSED_FORM,
            slope,
        )
        n_dofs = fe_problem.dof_map.num_total_dofs

        def J(params: dict[str, Any]) -> jax.Array:
            U_prev = jnp.zeros(n_dofs)
            U_star = _solve_step_closed(fe_problem, params, U_prev, t)
            return jnp.sum(U_star ** 2)

        params_at = {"elastic": {"kappa": 100.0, "mu": 50.0}}
        grad_J = jax.grad(J)(params_at)

        eps = 1e-6
        for key in ("kappa", "mu"):
            params_plus = {"elastic": dict(params_at["elastic"])}
            params_plus["elastic"][key] = params_at["elastic"][key] + eps
            params_minus = {"elastic": dict(params_at["elastic"])}
            params_minus["elastic"][key] = params_at["elastic"][key] - eps
            J_plus = float(J(params_plus))
            J_minus = float(J(params_minus))
            fd = (J_plus - J_minus) / (2 * eps)
            ad = float(grad_J["elastic"][key])
            np.testing.assert_allclose(
                ad, fd, rtol=1e-5, atol=1e-7,
                err_msg=(
                    f"AD vs FD mismatch for {key}: "
                    f"AD={ad}, FD={fd}, rel_err={abs(ad-fd)/abs(fd):.2e}"
                ),
            )


# ============================================================
# CLOSED_FORM multi-step
# ============================================================

class TestClosedFormMultiStep(unittest.TestCase):
    """``jax.grad`` of a sum-over-steps QoI through a 3-step
    CLOSED_FORM Elastic forward solve matches central-difference.
    Validates that vjp through the time loop composes with the outer
    ``lax.custom_root`` AD rule at each step.
    """

    def test_grad_kappa_mu_matches_fd(self) -> None:
        slope = 5e-4
        ts = (0.5, 1.0, 1.5)

        fe_problem = _build_fe_problem_2x2x2(
            _make_elastic_model(),
            GlobalResidualMode.CLOSED_FORM,
            slope,
        )

        def J(params: dict[str, Any]) -> jax.Array:
            U_history = _solve_history_closed(fe_problem, params, ts)
            return sum(jnp.sum(U_n ** 2) for U_n in U_history)

        params_at = {"elastic": {"kappa": 100.0, "mu": 50.0}}
        grad_J = jax.grad(J)(params_at)

        eps = 1e-6
        for key in ("kappa", "mu"):
            params_plus = {"elastic": dict(params_at["elastic"])}
            params_plus["elastic"][key] = params_at["elastic"][key] + eps
            params_minus = {"elastic": dict(params_at["elastic"])}
            params_minus["elastic"][key] = params_at["elastic"][key] - eps
            J_plus = float(J(params_plus))
            J_minus = float(J(params_minus))
            fd = (J_plus - J_minus) / (2 * eps)
            ad = float(grad_J["elastic"][key])
            np.testing.assert_allclose(
                ad, fd, rtol=1e-5, atol=1e-7,
                err_msg=(
                    f"AD vs FD mismatch for {key}: "
                    f"AD={ad}, FD={fd}, rel_err={abs(ad-fd)/abs(fd):.2e}"
                ),
            )


# ============================================================
# COUPLED — helpers shared by 0c / 0d / 0e
# ============================================================

def _set_param_at(params, path, value):
    """Return a copy of ``params`` with leaf at ``path`` replaced.

    ``path`` is a tuple of dict keys; only used by FD perturbation
    here so the structure mirrors the J2 ``Parameters`` tree.
    """
    out = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in params.items()}
    cur = out
    for k in path[:-1]:
        cur[k] = ({kk: (dict(vv) if isinstance(vv, dict) else vv)
                   for kk, vv in cur[k].items()})
        cur = cur[k]
    cur[path[-1]] = value
    return out


def _get_param_at(params, path):
    cur = params
    for k in path:
        cur = cur[k]
    return float(cur)


_J2_FD_PARAM_PATHS = (
    ("elastic", "E"),
    ("plastic", "flow stress", "initial yield", "Y"),
)


def _compare_ad_vs_fd(test_case, J, params_at, paths, eps=1e-3,
                      rtol=5e-4, atol=1e-6):
    """Central-difference vs ``jax.grad`` along each path in
    ``paths``. The default ``eps`` is loose because the J2 return-map
    is mildly non-smooth near the yield surface; tightening too far
    surfaces FD truncation noise rather than genuine AD/FD
    disagreement."""
    grad_J = jax.grad(J)(params_at)
    for path in paths:
        v0 = _get_param_at(params_at, path)
        params_plus = _set_param_at(params_at, path, v0 + eps)
        params_minus = _set_param_at(params_at, path, v0 - eps)
        J_plus = float(J(params_plus))
        J_minus = float(J(params_minus))
        fd = (J_plus - J_minus) / (2 * eps)
        ad_leaf = grad_J
        for k in path:
            ad_leaf = ad_leaf[k]
        ad = float(ad_leaf)
        np.testing.assert_allclose(
            ad, fd, rtol=rtol, atol=atol,
            err_msg=(
                f"AD vs FD mismatch for {'.'.join(path)}: "
                f"AD={ad}, FD={fd}, "
                f"rel_err={abs(ad-fd)/max(abs(fd), 1e-30):.2e}"
            ),
        )


# ============================================================
# COUPLED single step
# ============================================================

class TestCoupledSingleStep(unittest.TestCase):
    """``jax.grad`` of a scalar QoI through a single-step COUPLED
    SmallElasticPlastic forward solve matches central-difference.
    Validates that the outer ``lax.custom_root`` AD rule composes
    with the inner per-IP ``@custom_jvp`` (its tangent rule fires
    twice in one gradient call: once during forward K-assembly, once
    during reverse via JAX-synthesized vjp from the same custom_jvp).
    """

    def test_grad_matches_fd(self) -> None:
        slope = 2e-3  # peak eps_x = 2e-3 = 2x yield (Y/E = 1e-3)
        t = 1.0
        model = _make_J2_model()
        fe_problem = _build_fe_problem_2x2x2(
            model, GlobalResidualMode.COUPLED, slope,
        )
        params_at = model.parameters.values

        def J(params: dict[str, Any]) -> jax.Array:
            n_dofs = fe_problem.dof_map.num_total_dofs
            U_prev = jnp.zeros(n_dofs)
            xi_prev = _initial_xi_by_block(fe_problem)
            U_star, _ = _solve_step_coupled(
                fe_problem, params, U_prev, xi_prev, t,
            )
            return jnp.sum(U_star ** 2)

        _compare_ad_vs_fd(self, J, params_at, _J2_FD_PARAM_PATHS)


# ============================================================
# COUPLED multi-step, simple QoI
# ============================================================

class TestCoupledMultiStepSimple(unittest.TestCase):
    """``jax.grad`` of a sum-over-steps QoI (depends only on each
    step's U) through a 3-step COUPLED forward solve matches
    central-difference. Step times chosen so steps 2 and 3 fall in
    the plastic regime — exercises the cross-step xi-history adjoint
    chain (``xi_{n-1}`` cotangents flowing backward across the inner
    custom_jvp boundary).
    """

    def test_grad_matches_fd(self) -> None:
        slope = 2e-3
        ts = (0.5, 1.0, 1.5)  # eps_x peaks ~ [1e-3, 2e-3, 3e-3]
        model = _make_J2_model()
        fe_problem = _build_fe_problem_2x2x2(
            model, GlobalResidualMode.COUPLED, slope,
        )
        params_at = model.parameters.values

        def J(params: dict[str, Any]) -> jax.Array:
            U_history, _ = _solve_history_coupled(
                fe_problem, params, ts,
            )
            return sum(jnp.sum(U_n ** 2) for U_n in U_history)

        _compare_ad_vs_fd(self, J, params_at, _J2_FD_PARAM_PATHS)


# ============================================================
# COUPLED multi-step, all-paths QoI capstone
# ============================================================

class TestCoupledMultiStepAllPaths(unittest.TestCase):
    """``jax.grad`` of an all-five-inputs QoI through a 3-step
    COUPLED forward solve matches central-difference. The QoI
    couples to ``U_n``, ``U_{n-1}``, ``xi_n``, ``xi_{n-1}``, and
    ``p`` directly, so every cotangent path contributes
    non-trivially. If 0c–0d pass and this fails, the boundaries
    individually work but the cross-path summation is wrong.
    """

    def test_grad_matches_fd(self) -> None:
        slope = 2e-3
        ts = (0.5, 1.0, 1.5)
        model = _make_J2_model()
        fe_problem = _build_fe_problem_2x2x2(
            model, GlobalResidualMode.COUPLED, slope,
        )
        params_at = model.parameters.values

        c_U, c_Up, c_xi, c_xp, c_p = 1.0, 0.7, 0.3, 0.5, 1e-4

        def J(params: dict[str, Any]) -> jax.Array:
            n_dofs = fe_problem.dof_map.num_total_dofs
            U_history, xi_history = _solve_history_coupled(
                fe_problem, params, ts,
            )
            U_prev_seq = [jnp.zeros(n_dofs)] + U_history[:-1]
            xi_prev_seq = ([_initial_xi_by_block(fe_problem)]
                           + xi_history[:-1])
            total = jnp.array(0.0)
            for U_n, U_p, xi_n, xi_p in zip(
                    U_history, U_prev_seq,
                    xi_history, xi_prev_seq, strict=True):
                total = total + c_U * jnp.sum(U_n ** 2)
                total = total + c_Up * jnp.sum(U_p ** 2)
                for block in xi_n:
                    total = total + c_xi * jnp.sum(xi_n[block] ** 2)
                    total = total + c_xp * jnp.sum(xi_p[block] ** 2)
            # Direct-through-parameters term — exercises the path
            # that bypasses both implicit solves.
            total = total + c_p * (params["elastic"]["E"] ** 2
                                   + params["plastic"]["flow stress"][
                                       "initial yield"]["Y"] ** 2)
            return total

        _compare_ad_vs_fd(self, J, params_at, _J2_FD_PARAM_PATHS)


if __name__ == "__main__":
    unittest.main()
