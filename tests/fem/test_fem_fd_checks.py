"""FD-vs-AD checks for ``jax.grad`` and ``jax.hessian`` over the
sparse FE forward solve.

Five test classes (CLOSED_FORM single/multi-step + COUPLED single/
multi-step simple/all-paths), each with a ``test_grad_matches_fd``
and a ``test_hessian_matches_fd``. Each test compares AD against a
directional-difference FD across a logspace ``hs`` range, asserting
the FD-error log10 drop exceeds a threshold (clean V-shaped FD
convergence). Mirrors the MP-side ``tests/objectives/test_J2_fd_checks.py``
pattern.

Pieces under test — no test-only fixtures of the forward solve:

- Per-element kernels: ``per_element_R_and_K`` /
  ``per_element_R_and_K_coupled`` from ``cmad.fem.assembly``.
- Per-IP local Newton (COUPLED): ``make_newton_solve`` wired via
  ``GR.for_model(model, COUPLED)``; ``@custom_jvp`` IFT rule.
- Global assembly: ``cmad.fem.assembly.assemble_global``
  (BCOO + scipy.coo bridge → sparse CSR).
- Sparse linear solve: ``cmad.fem.sparse_solve.spsolve_jax`` via
  ``lax.custom_linear_solve`` over a ``pure_callback`` to scipy
  ``spsolve``; the ``K``-side cotangent flows via
  ``custom_linear_solve``'s VJP rule.
- Outer FE Newton: ``cmad.fem.nonlinear_solver.fe_newton_solve``
  with its ``@custom_jvp`` rule (IFT linear sensitivity equation
  with ``∂r/∂p · p_dot`` from ``jax.jvp(r, p, p_dot)`` at fixed
  ``U_star``, solved through ``spsolve_jax``).
- Strong-DBC enforcement: ``cmad.fem.sparse_solve._embedded_bc_enforce``
  (asymmetric: prescribed rows zeroed, identity 1.0 on the
  prescribed diagonal).
- Multi-step time loop: ``cmad.fem.driver.fe_quasistatic_trajectory``,
  ``lax.scan`` over the time schedule with carry
  ``(U_prev, xi_prev_by_block)``.

The tests are arranged as boundary-isolation diagnostics: each adds
one new AD boundary on top of the previous, so a failure localizes
which boundary broke. The final test is the everything-composes
capstone: if individual boundaries pass and the capstone fails, the
cross-paths between boundaries are wrong, not the boundaries themselves.
"""
import unittest
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.driver import fe_quasistatic_trajectory
from cmad.fem.fe_problem import build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.fem.nonlinear_solver import fe_newton_solve
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


def _initial_xi_by_block(fe_problem) -> dict[str, jax.Array]:
    """Per-block xi at step 0: flatten ``model._init_xi`` and tile
    across ``(n_elems_block, n_ips, total_xi_dofs)`` — same pattern
    as ``FEState.from_problem``, expressed in JAX so it can flow
    into traced calls. CLOSED_FORM blocks get an empty trailing
    dim (``ravel_pytree`` of an empty xi is a 0-length array),
    which the FE-Newton driver handles uniformly with COUPLED
    blocks."""
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


# ============================================================
# Shared FD / Hessian helpers
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


def _apply_step(params_at, paths, steps):
    """Return a copy of ``params_at`` with ``steps[k]`` added to leaf
    at ``paths[k]`` for each k.
    """
    new_params = params_at
    for path, step in zip(paths, steps, strict=True):
        v0 = _get_param_at(new_params, path)
        new_params = _set_param_at(new_params, path, v0 + step)
    return new_params


def _hessian_to_dense(hess_pytree, paths):
    """Flatten a ``jax.hessian`` pytree to dense ``(n, n)`` over ``paths``.

    ``jax.hessian(J)(params)`` carries the params pytree structure
    twice nested; ``H[i, j]`` is the second mixed partial along
    ``paths[i]`` then ``paths[j]``.
    """
    n = len(paths)
    H = np.zeros((n, n))
    for i, path_i in enumerate(paths):
        sub = hess_pytree
        for k in path_i:
            sub = sub[k]
        for j, path_j in enumerate(paths):
            H[i, j] = _get_param_at(sub, path_j)
    return H


def _param_scales(params_at, paths):
    """Per-path scaling factor: ``|params_at[path]|`` (or 1.0 if zero).

    Multiplying the FD step by this scaling makes ``h`` a *relative*
    perturbation regardless of the parameter's raw magnitude, so a
    single ``hs = logspace(-2, -10, 9)`` range gives clean FD
    convergence behavior across params with very different scales
    (e.g. ``E ~ 2e5`` vs ``Y ~ 2e2``).
    """
    raw = np.array([abs(_get_param_at(params_at, p)) for p in paths])
    return np.where(raw > 0, raw, 1.0)


def _fd_grad_dir_deriv_errors(J_fn, grad_fn, params_at, paths,
                              hs, seed):
    """FD-vs-AD directional-derivative errors across an ``hs`` range.

    Picks a random direction ``d`` in scaled space, then for each
    ``h`` computes the central-difference of ``J_fn`` along
    ``scales * d`` and compares to the AD reference
    ``(scales * d) @ grad``.
    """
    n = len(paths)
    rng = np.random.default_rng(seed)
    d = rng.uniform(-1.0, 1.0, size=n)
    scales = _param_scales(params_at, paths)
    sd = scales * d

    grad_at = grad_fn(params_at)
    grad_flat = np.array([_get_param_at(grad_at, p) for p in paths])
    dir_deriv_ref = float(sd @ grad_flat)

    fd_errors = np.zeros(len(hs))
    for ii, h in enumerate(hs):
        params_plus = _apply_step(params_at, paths, h * sd)
        params_minus = _apply_step(params_at, paths, -h * sd)
        J_plus = float(J_fn(params_plus))
        J_minus = float(J_fn(params_minus))
        fd_dir_deriv = (J_plus - J_minus) / (2 * h)
        fd_errors[ii] = abs(fd_dir_deriv - dir_deriv_ref)
    return fd_errors


def _fd_hessian_dir_deriv_errors(J_fn, hess_fn, params_at, paths,
                                 hs, seed):
    """FD-vs-AD Hessian-directional-derivative errors across ``hs``.

    Picks a random direction ``d`` in scaled space, then for each
    ``h`` computes the central second-difference
    ``(J(p + h*scales*d) + J(p - h*scales*d) - 2*J(p)) / h**2``
    and compares to the AD reference ``(scales*d) @ H @ (scales*d)``.
    """
    n = len(paths)
    rng = np.random.default_rng(seed)
    d = rng.uniform(-1.0, 1.0, size=n)
    scales = _param_scales(params_at, paths)
    sd = scales * d

    H_dense = _hessian_to_dense(hess_fn(params_at), paths)
    dir_deriv_ref = float(sd @ H_dense @ sd)
    J_ref = float(J_fn(params_at))

    fd_errors = np.zeros(len(hs))
    for ii, h in enumerate(hs):
        params_plus = _apply_step(params_at, paths, h * sd)
        params_minus = _apply_step(params_at, paths, -h * sd)
        J_plus = float(J_fn(params_plus))
        J_minus = float(J_fn(params_minus))
        fd_dir_deriv = (J_plus + J_minus - 2 * J_ref) / h**2
        fd_errors[ii] = abs(fd_dir_deriv - dir_deriv_ref)
    return fd_errors


_DEFAULT_HS = np.logspace(0, -10, 20)


def _assert_fd_drop(test_case, fd_errors, hs, error_drop_tol):
    """Assert the log10 drop of FD errors across ``hs`` exceeds
    ``error_drop_tol``. A clean V-shaped FD-error curve has a wide
    drop; a flat curve indicates AD/FD disagreement (truncation
    error not converging) and fails."""
    log10_drop = float(np.log10(np.max(fd_errors) / np.min(fd_errors)))
    test_case.assertGreater(
        log10_drop, error_drop_tol,
        msg=(
            f"FD log10 error drop {log10_drop:.2f} <= {error_drop_tol}; "
            f"FD didn't converge well. "
            f"hs={hs.tolist()}, errors={fd_errors.tolist()}"
        ),
    )


def _compare_ad_vs_fd(test_case, J_fn, grad_fn, params_at, paths,
                      hs=_DEFAULT_HS, seed=22, error_drop_tol=5.0):
    """Range-based directional-derivative FD check on AD gradient."""
    fd_errors = _fd_grad_dir_deriv_errors(
        J_fn, grad_fn, params_at, paths, hs, seed,
    )
    _assert_fd_drop(test_case, fd_errors, hs, error_drop_tol)


def _compare_hessian_ad_vs_fd(test_case, J_fn, hess_fn, params_at,
                              paths, hs=_DEFAULT_HS, seed=22,
                              error_drop_tol=5.0):
    """Range-based directional-derivative FD check on AD Hessian."""
    fd_errors = _fd_hessian_dir_deriv_errors(
        J_fn, hess_fn, params_at, paths, hs, seed,
    )
    _assert_fd_drop(test_case, fd_errors, hs, error_drop_tol)


# ============================================================
# CLOSED_FORM single step
# ============================================================

class TestClosedFormSingleStep(unittest.TestCase):
    """``jax.grad`` and ``jax.hessian`` of a scalar QoI through a
    single-step CLOSED_FORM Elastic forward solve via
    ``fe_newton_solve`` match central-difference. Validates the
    outer FE Newton's ``@custom_jvp`` rule + sparse linear-solve
    VJP at one time step. No inner local Newton invoked
    (CLOSED_FORM); no time-history adjoint (single step).
    """

    J: Callable[[PyTreeDict], jax.Array]
    grad_J: Callable[[PyTreeDict], PyTreeDict]
    hess_J: Callable[[PyTreeDict], PyTreeDict]
    params_at: PyTreeDict
    paths: tuple[tuple[str, ...], ...]

    @classmethod
    def setUpClass(cls) -> None:
        slope = 5e-4
        t = 1.0
        # Build problem once with concrete (irrelevant for AD) params;
        # parameter dependence flows through the explicit ``params``
        # arg into ``fe_newton_solve``.
        fe_problem = _build_fe_problem_2x2x2(
            _make_elastic_model(),
            GlobalResidualMode.CLOSED_FORM,
            slope,
        )
        n_dofs = fe_problem.dof_map.num_total_dofs

        def _J(params: dict[str, Any]) -> jax.Array:
            U_prev = jnp.zeros(n_dofs)
            xi_prev = _initial_xi_by_block(fe_problem)
            U_star, _ = fe_newton_solve(
                fe_problem, {"all": params},
                U_prev=U_prev, t=t, xi_prev_by_block=xi_prev,
                max_iters=20, abs_tol=1e-12, rel_tol=1e-12,
            )
            return jnp.sum(U_star ** 2)

        cls.J = staticmethod(jax.jit(_J))
        cls.grad_J = staticmethod(jax.jit(jax.grad(_J)))
        cls.hess_J = staticmethod(jax.jit(jax.hessian(_J)))
        cls.params_at = {"elastic": {"kappa": 100.0, "mu": 50.0}}
        cls.paths = (("elastic", "kappa"), ("elastic", "mu"))

    def test_grad_matches_fd(self) -> None:
        _compare_ad_vs_fd(
            self, self.J, self.grad_J, self.params_at, self.paths,
        )

    def test_hessian_matches_fd(self) -> None:
        _compare_hessian_ad_vs_fd(
            self, self.J, self.hess_J, self.params_at, self.paths,
        )


# ============================================================
# CLOSED_FORM multi-step
# ============================================================

class TestClosedFormMultiStep(unittest.TestCase):
    """``jax.grad`` and ``jax.hessian`` of a sum-over-steps QoI
    through a 3-step CLOSED_FORM Elastic forward solve via
    ``fe_quasistatic_trajectory`` match central-difference.
    Validates that vjp through the ``lax.scan`` time loop composes
    with the outer FE Newton's ``@custom_jvp`` rule at each step.
    """

    J: Callable[[PyTreeDict], jax.Array]
    grad_J: Callable[[PyTreeDict], PyTreeDict]
    hess_J: Callable[[PyTreeDict], PyTreeDict]
    params_at: PyTreeDict
    paths: tuple[tuple[str, ...], ...]

    @classmethod
    def setUpClass(cls) -> None:
        slope = 5e-4
        ts = (0.5, 1.0, 1.5)
        fe_problem = _build_fe_problem_2x2x2(
            _make_elastic_model(),
            GlobalResidualMode.CLOSED_FORM,
            slope,
        )
        n_dofs = fe_problem.dof_map.num_total_dofs

        def _J(params: dict[str, Any]) -> jax.Array:
            U_init = jnp.zeros(n_dofs)
            xi_init = _initial_xi_by_block(fe_problem)
            t_schedule_jax = jnp.asarray([0.0, *ts], dtype=jnp.float64)
            U_steps, _, _ = fe_quasistatic_trajectory(
                fe_problem, {"all": params},
                U_init, xi_init, t_schedule_jax,
                max_iters=20, abs_tol=1e-12, rel_tol=1e-12,
            )
            # U_steps has shape (N, n_dofs); summing over both axes
            # gives the same scalar as the per-step-sum-then-sum form.
            return jnp.sum(U_steps ** 2)

        cls.J = staticmethod(jax.jit(_J))
        cls.grad_J = staticmethod(jax.jit(jax.grad(_J)))
        cls.hess_J = staticmethod(jax.jit(jax.hessian(_J)))
        cls.params_at = {"elastic": {"kappa": 100.0, "mu": 50.0}}
        cls.paths = (("elastic", "kappa"), ("elastic", "mu"))

    def test_grad_matches_fd(self) -> None:
        _compare_ad_vs_fd(
            self, self.J, self.grad_J, self.params_at, self.paths,
        )

    def test_hessian_matches_fd(self) -> None:
        # Trim the small-h tail where FD signal drops below the
        # Newton-convergence floor and reads as a constant plateau.
        _compare_hessian_ad_vs_fd(
            self, self.J, self.hess_J, self.params_at, self.paths,
            hs=np.logspace(0, -7, 15),
        )


# ============================================================
# COUPLED single step
# ============================================================

class TestCoupledSingleStep(unittest.TestCase):
    """``jax.grad`` and ``jax.hessian`` of a scalar QoI through a
    single-step COUPLED SmallElasticPlastic forward solve via
    ``fe_newton_solve`` match central-difference. Validates that
    the outer FE Newton's ``@custom_jvp`` rule composes with the
    inner per-IP local Newton's ``@custom_jvp``.
    """

    J: Callable[[PyTreeDict], jax.Array]
    grad_J: Callable[[PyTreeDict], PyTreeDict]
    hess_J: Callable[[PyTreeDict], PyTreeDict]
    params_at: PyTreeDict

    @classmethod
    def setUpClass(cls) -> None:
        slope = 2e-3  # peak eps_x = 2e-3 = 2x yield (Y/E = 1e-3)
        t = 1.0
        model = _make_J2_model()
        fe_problem = _build_fe_problem_2x2x2(
            model, GlobalResidualMode.COUPLED, slope,
        )
        cls.params_at = model.parameters.values
        n_dofs = fe_problem.dof_map.num_total_dofs

        def _J(params: dict[str, Any]) -> jax.Array:
            U_prev = jnp.zeros(n_dofs)
            xi_prev = _initial_xi_by_block(fe_problem)
            U_star, _ = fe_newton_solve(
                fe_problem, {"all": params},
                U_prev=U_prev, t=t, xi_prev_by_block=xi_prev,
                max_iters=30, abs_tol=1e-10, rel_tol=1e-10,
            )
            return jnp.sum(U_star ** 2)

        cls.J = staticmethod(jax.jit(_J))
        cls.grad_J = staticmethod(jax.jit(jax.grad(_J)))
        cls.hess_J = staticmethod(jax.jit(jax.hessian(_J)))

    def test_grad_matches_fd(self) -> None:
        _compare_ad_vs_fd(
            self, self.J, self.grad_J, self.params_at,
            _J2_FD_PARAM_PATHS,
        )

    def test_hessian_matches_fd(self) -> None:
        _compare_hessian_ad_vs_fd(
            self, self.J, self.hess_J, self.params_at,
            _J2_FD_PARAM_PATHS,
        )


# ============================================================
# COUPLED multi-step, simple QoI
# ============================================================

class TestCoupledMultiStepSimple(unittest.TestCase):
    """``jax.grad`` and ``jax.hessian`` of a sum-over-steps QoI
    (depends only on each step's U) through a multi-step COUPLED
    forward solve via ``fe_quasistatic_trajectory`` match
    central-difference. Step times span elastic and plastic
    regimes, exercising the cross-step xi-history adjoint chain
    (``xi_{n-1}`` cotangents flowing backward through the
    ``lax.scan`` carry across the inner-Newton ``@custom_jvp``
    boundary).
    """

    J: Callable[[PyTreeDict], jax.Array]
    grad_J: Callable[[PyTreeDict], PyTreeDict]
    hess_J: Callable[[PyTreeDict], PyTreeDict]
    params_at: PyTreeDict

    @classmethod
    def setUpClass(cls) -> None:
        slope = 2e-3
        # 2 elastic + 3 plastic, none on the yield boundary (eps = 1e-3
        # at t = 0.5). Strains: [4e-4, 8e-4, 1.2e-3, 1.6e-3, 2e-3].
        ts = (0.2, 0.4, 0.6, 0.8, 1.0)
        model = _make_J2_model()
        fe_problem = _build_fe_problem_2x2x2(
            model, GlobalResidualMode.COUPLED, slope,
        )
        cls.params_at = model.parameters.values
        n_dofs = fe_problem.dof_map.num_total_dofs

        def _J(params: dict[str, Any]) -> jax.Array:
            U_init = jnp.zeros(n_dofs)
            xi_init = _initial_xi_by_block(fe_problem)
            t_schedule_jax = jnp.asarray([0.0, *ts], dtype=jnp.float64)
            U_steps, _, _ = fe_quasistatic_trajectory(
                fe_problem, {"all": params},
                U_init, xi_init, t_schedule_jax,
                max_iters=30, abs_tol=1e-10, rel_tol=1e-10,
            )
            return jnp.sum(U_steps ** 2)

        cls.J = staticmethod(jax.jit(_J))
        cls.grad_J = staticmethod(jax.jit(jax.grad(_J)))
        cls.hess_J = staticmethod(jax.jit(jax.hessian(_J)))

    def test_grad_matches_fd(self) -> None:
        _compare_ad_vs_fd(
            self, self.J, self.grad_J, self.params_at,
            _J2_FD_PARAM_PATHS,
        )

    def test_hessian_matches_fd(self) -> None:
        _compare_hessian_ad_vs_fd(
            self, self.J, self.hess_J, self.params_at,
            _J2_FD_PARAM_PATHS,
        )


# ============================================================
# COUPLED multi-step, all-paths QoI capstone
# ============================================================

class TestCoupledMultiStepAllPaths(unittest.TestCase):
    """``jax.grad`` and ``jax.hessian`` of an all-five-inputs QoI
    through a multi-step COUPLED forward solve via
    ``fe_quasistatic_trajectory`` match central-difference.
    The QoI couples to ``U_n``, ``U_{n-1}``, ``xi_n``, ``xi_{n-1}``,
    and ``p`` directly, so every cotangent path contributes
    non-trivially. If the simpler tests pass and this fails, the
    boundaries individually work but the cross-path summation is
    wrong.
    """

    J: Callable[[PyTreeDict], jax.Array]
    grad_J: Callable[[PyTreeDict], PyTreeDict]
    hess_J: Callable[[PyTreeDict], PyTreeDict]
    params_at: PyTreeDict

    @classmethod
    def setUpClass(cls) -> None:
        slope = 2e-3
        # 2 elastic + 3 plastic, none on the yield boundary.
        ts = (0.2, 0.4, 0.6, 0.8, 1.0)
        model = _make_J2_model()
        fe_problem = _build_fe_problem_2x2x2(
            model, GlobalResidualMode.COUPLED, slope,
        )
        cls.params_at = model.parameters.values
        n_dofs = fe_problem.dof_map.num_total_dofs

        c_U, c_Up, c_xi, c_xp, c_p = 1.0, 0.7, 0.3, 0.5, 1e-4

        def _J(params: dict[str, Any]) -> jax.Array:
            U_init = jnp.zeros(n_dofs)
            xi_init_dict = _initial_xi_by_block(fe_problem)
            t_schedule_jax = jnp.asarray([0.0, *ts], dtype=jnp.float64)
            U_steps, xi_steps, _ = fe_quasistatic_trajectory(
                fe_problem, {"all": params},
                U_init, xi_init_dict, t_schedule_jax,
                max_iters=30, abs_tol=1e-10, rel_tol=1e-10,
            )
            # U_prev_seq shape (N, n_dofs): prepend U_init,
            # drop final step. Same shift, applied per block, for xi.
            U_prev_seq = jnp.concatenate(
                [U_init[None, :], U_steps[:-1]], axis=0,
            )
            xi_prev_seq = {
                block: jnp.concatenate(
                    [xi_init_dict[block][None], xi_steps[block][:-1]],
                    axis=0,
                )
                for block in xi_steps
            }
            total = jnp.array(0.0)
            total = total + c_U * jnp.sum(U_steps ** 2)
            total = total + c_Up * jnp.sum(U_prev_seq ** 2)
            for block in xi_steps:
                total = total + c_xi * jnp.sum(xi_steps[block] ** 2)
                total = total + c_xp * jnp.sum(xi_prev_seq[block] ** 2)
            # Direct-through-parameters term — exercises the path
            # that bypasses both implicit solves.
            total = total + c_p * (params["elastic"]["E"] ** 2
                                   + params["plastic"]["flow stress"][
                                       "initial yield"]["Y"] ** 2)
            return total

        cls.J = staticmethod(jax.jit(_J))
        cls.grad_J = staticmethod(jax.jit(jax.grad(_J)))
        cls.hess_J = staticmethod(jax.jit(jax.hessian(_J)))

    def test_grad_matches_fd(self) -> None:
        _compare_ad_vs_fd(
            self, self.J, self.grad_J, self.params_at,
            _J2_FD_PARAM_PATHS,
        )

    def test_hessian_matches_fd(self) -> None:
        # Shift hs upward: the all-paths capstone's V min sits at
        # h~5e-2 with steep noise blowup below — extend down only as
        # far as h=1e-5 (still well into the noise regime) so the V
        # is well-resolved without the noise plateau dragging the
        # drop.
        _compare_hessian_ad_vs_fd(
            self, self.J, self.hess_J, self.params_at,
            _J2_FD_PARAM_PATHS, hs=np.logspace(0, -5, 11),
        )


if __name__ == "__main__":
    unittest.main()
