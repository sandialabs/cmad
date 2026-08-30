"""Element axis sharding of the FE assembly across the process's devices.

JAX fixes its device count when the backend initialises, so the
multidevice cases run :func:`save_sharded_results` in a subprocess with
``jax_num_cpu_devices`` set first (JAX's own tests do the same); it saves
what it computed on four CPU devices, and the test compares that with the
same computation on one device: this process for the Newton solve, device
0 of the subprocess for the assembly. The divisor rule and the one device
path are checked in process.
"""
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import tree_leaves, tree_map

from cmad.fem.assembly import (
    assemble_element_tangent,
    assemble_global,
    params_by_block_from_models,
)
from cmad.fem.fe_problem import FEState
from cmad.fem.mesh import StructuredHexMesh
from cmad.fem.nonlinear_solver import (
    _assemble_tangent_and_residual,
    _tangent_operator,
    fe_newton_solve,
)
from cmad.fem.sharding import (
    ELEMENT_AXIS,
    build_device_mesh,
    element_shard_count,
    place_element_leaves,
)
from cmad.fem.sparse_solve import (
    AssembledOperator,
    ElementOperator,
    _block_precon_apply,
    _chebyshev_field_bounds,
    _embedded_bc_enforce,
    _gmres_loop,
    _jacobi_preconditioner,
    _pcg_loop,
)
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.models.deformation_types import DefType
from cmad.models.global_fields import StepTime
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from tests.fem.test_assembly_coupled import (
    _build_fe_problem,
    _split_into_two_blocks,
)
from tests.fem.test_mixed_up_plastic import (
    _JAX_BLOCK_CHEBYSHEV_SETTINGS,
    MAX_ALPHA,
    NUM_DRIVE_STEPS,
    _build_mixed_fe,
)
from tests.support.test_problems import J2AnalyticalProblem

_ELEMENT_SOLVER_SETTINGS = {**_JAX_BLOCK_CHEBYSHEV_SETTINGS, "operator": "element"}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_N_DEVICES = 4
# The mixed u-p problem's element axis leaves: two U gather index arrays
# (u, p), two R scatter index arrays, and the five per element geometry
# arrays (iso_jac_det, coords_ip, two field gradients, element_size).
_N_ELEMENT_LEAVES = 9

# The device count has to be set before anything initialises the backend.
_SUBPROCESS_SCRIPT = """
import sys
import jax
jax.config.update("jax_num_cpu_devices", int(sys.argv[1]))
from tests.fem.test_sharding import save_sharded_results
save_sharded_results(sys.argv[2])
"""


def drive_mixed_problem():
    """The uniaxial mixed u-p cube of ``test_mixed_up_plastic`` pulled two
    load steps in, so the second step is plastic: ``(fe_problem, params,
    U, U_prev, xi, xi_prev, step_time)`` at the second step, the arrays as
    numpy."""
    problem = J2AnalyticalProblem()
    stress_mask = np.zeros((3, 3))
    stress_mask[0, 0] = 1.0
    _, strain, _ = problem.analytical_solution(
        stress_mask, MAX_ALPHA, num_steps=2,
    )
    axial_strain = float(strain[0, 0, -1])
    model = SmallElasticPlastic(
        problem.J2_parameters, def_type=DefType.FULL_3D,
    )
    fe_problem = _build_mixed_fe(model)
    params = params_by_block_from_models(fe_problem)
    state = FEState.from_problem(fe_problem)
    U = state.U_at(0)
    xi = {"all": state.xi_at(0, "all")}
    t_prev = 0.0
    for step in range(1, 3):
        U_prev, xi_prev = U, xi
        t = axial_strain * step / NUM_DRIVE_STEPS
        U, xi = fe_newton_solve(
            fe_problem, params, U_prev=U_prev, t=t, t_prev=t_prev,
            xi_prev_by_block=xi_prev,
        )
        t_prev = t
    return (
        fe_problem, params, np.asarray(U), np.asarray(U_prev),
        {b: np.asarray(x) for b, x in xi.items()},
        {b: np.asarray(x) for b, x in xi_prev.items()},
        StepTime(t=t, t_prev=t_prev),
    )


def elastic_problem(divisions, two_blocks=False):
    """An elastic Mechanics problem on a structured hex mesh, in one
    element block or two."""
    mesh = StructuredHexMesh(lengths=(1.0, 1.0, 1.0), divisions=divisions)
    if two_blocks:
        mesh = _split_into_two_blocks(mesh)
    return _build_fe_problem(
        mesh,
        {b: GlobalResidualMode.CLOSED_FORM for b in mesh.element_blocks},
    )


def assemble_at_random_state(fe_problem, arrays):
    """``(K.data, R)`` of ``fe_problem`` at a fixed random U, assembled
    through ``arrays``."""
    n = fe_problem.dof_map.num_total_dofs
    U = 0.01 * np.random.default_rng(0).standard_normal(n)
    params = params_by_block_from_models(fe_problem)

    def assemble(arrs):
        K, R, _ = assemble_global(
            fe_problem, arrs, params, U, np.zeros(n), StepTime(1.0, 0.0),
        )
        return K.data, R

    K_data, R = jax.jit(assemble)(arrays)
    return np.asarray(K_data), np.asarray(R)


def _on_device_0(tree):
    """``tree`` with every leaf committed to device 0."""
    device_0 = jax.devices()[0]
    return tree_map(lambda leaf: jax.device_put(leaf, device_0), tree)


_CG_SETTINGS = {"type": "cg", "rtol": 1.0e-10, "max iters": 2000}


def elastic_krylov_problem():
    """``(fe_problem, params, U)``: the elastic 3 x 3 x 3 cube (192 dofs, the
    24 at the interior nodes free) at a random displacement."""
    fe_problem = elastic_problem((3, 3, 3))
    params = params_by_block_from_models(fe_problem)
    n = fe_problem.dof_map.num_total_dofs
    U = 0.01 * np.random.default_rng(27).standard_normal(n)
    return fe_problem, params, U


def _block_chebyshev_preconditioner(op, bounds):
    def precon(v):
        return _block_precon_apply(
            op, v, coupling="lower", diagonal_block="schur",
            inner="chebyshev", transpose=False,
            chebyshev_degree=3, chebyshev_bounds=bounds,
        )
    return precon


def krylov_solves(fe_problem, params, U, U_prev, xi_prev, step_time):
    """One Krylov solve of ``K dU = -r`` at the given state per operator
    kind, through the problem's device mesh: ``{kind: (iterations, x)}``.
    A mixed problem runs block chebyshev GMRES, a single field problem
    CG + jacobi, both unrestarted at rtol 1e-10."""
    arrays = fe_problem.kernel_arrays
    presc_vals = jnp.asarray(fe_problem.dof_map.evaluate_prescribed_values(
        arrays.dbc_arrays, float(step_time.t),
    ))
    mixed = arrays.block_sparsity is not None
    out = {}
    for kind in ("assembled", "element"):
        def solve(U_, U_prev_, xi_prev_, kind=kind):
            r, K, _ = _assemble_tangent_and_residual(
                fe_problem, arrays, params, U_, U_prev_, step_time, xi_prev_,
                presc_vals, kind,
            )
            op = _tangent_operator(K, fe_problem, arrays, kind)
            if mixed:
                precon = _block_chebyshev_preconditioner(
                    op, _chebyshev_field_bounds(op, "schur"),
                )
                return _gmres_loop(op.matvec, precon, -r, 1.0e-10, op.n, 4,
                                   op.device_mesh)
            return _pcg_loop(op.matvec, -r, _jacobi_preconditioner(op),
                             1.0e-10, 10 * op.n, op.device_mesh)

        x, iterations = jax.jit(solve)(
            jnp.asarray(U), jnp.asarray(U_prev),
            place_element_leaves(
                {b: jnp.asarray(v) for b, v in xi_prev.items()},
                fe_problem.device_mesh,
            ),
        )
        out[kind] = (int(iterations), np.asarray(x))
    return out


def _mesh_size(fe_problem) -> int:
    mesh = fe_problem.device_mesh
    return 0 if mesh is None else int(mesh.size)


def save_sharded_results(path: str) -> None:
    """The multidevice half of the test, run in a subprocess with several
    CPU devices: saves what it computes to the ``.npz`` at ``path``."""
    n_devices = len(jax.devices())
    (fe_problem, params, U, U_prev, xi, xi_prev,
     step_time) = drive_mixed_problem()
    mesh = fe_problem.device_mesh
    arrays = fe_problem.kernel_arrays

    element_sharding = NamedSharding(mesh, PartitionSpec(ELEMENT_AXIS))

    def is_element_sharded(x) -> bool:
        return x.ndim >= 1 and x.sharding.is_equivalent_to(
            element_sharding, x.ndim,
        )

    element_leaves = [
        *tree_leaves(arrays.u_gather_eq_by_block),
        *tree_leaves(arrays.r_scatter_eq_by_block),
        *tree_leaves(
            {b: c.per_elem for b, c in arrays.geometry_cache.items()},
        ),
    ]
    other_leaves = [
        *tree_leaves(
            {b: c.shared for b, c in arrays.geometry_cache.items()},
        ),
        arrays.coo_rows, arrays.coo_cols, arrays.coo_dedup_scatter,
        arrays.prescribed_indices,
    ]
    xi_placed = place_element_leaves(xi_prev, mesh)

    # The assembly at the plastic state through the sharded carrier, and
    # through the same arrays committed to device 0.
    def assemble(arrs, xi_p):
        K, R, xi_out = assemble_global(
            fe_problem, arrs, params, U, U_prev, step_time,
            xi_prev_by_block=xi_p,
        )
        return K.data, R, xi_out["all"]

    K_data, R, xi_out = (
        np.asarray(a) for a in jax.jit(assemble)(arrays, xi_placed)
    )
    K_data_ref, R_ref, xi_out_ref = (
        np.asarray(a) for a in jax.jit(assemble)(
            _on_device_0(arrays), _on_device_0(xi_prev),
        )
    )

    # The element operator built from the sharded element blocks against
    # the assembled operator on device 0, and a Newton step through it.
    K_elem = jax.jit(lambda arrs, xi_p: assemble_element_tangent(
        fe_problem, arrays, params, U, U_prev, step_time,
        xi_prev_by_block=xi_p,
    )[0])(arrays, xi_placed)
    element = ElementOperator(
        K_elem, arrays.r_scatter_eq_by_block, fe_problem.field_idx_per_block,
        fe_problem.dof_map.block_offsets, arrays.prescribed_indices,
        fe_problem.dof_map.num_total_dofs,
    )
    arrays_0 = _on_device_0(arrays)
    K_bcoo_0, _, _ = jax.jit(lambda arrs, xi_p: assemble_global(
        fe_problem, arrs, params, U, U_prev, step_time, xi_prev_by_block=xi_p,
    ))(arrays_0, _on_device_0(xi_prev))
    K_enforced_0, _ = _embedded_bc_enforce(K_bcoo_0, arrays_0.prescribed_indices)
    assembled_0 = AssembledOperator(
        K_enforced_0, arrays_0.embedded_sparsity, arrays_0.block_sparsity,
    )
    x = jnp.asarray(np.random.default_rng(5).standard_normal(
        fe_problem.dof_map.num_total_dofs))
    element_matvec = np.asarray(jax.jit(element.matvec)(x))
    element_matvec_ref = np.asarray(jax.jit(assembled_0.matvec)(x))
    U_element_step, _ = fe_newton_solve(
        fe_problem, params, U_prev=U_prev, t=float(step_time.t),
        t_prev=float(step_time.t_prev), xi_prev_by_block=xi_prev,
        linear_solver_settings=_ELEMENT_SOLVER_SETTINGS,
    )

    # Partitioning the Krylov vectors across the devices changes where the
    # arithmetic runs, not its result: same iteration count and the same
    # solution as on one device. Mixed cube (108 dofs) with block chebyshev
    # GMRES at the start of a step (a residual of size one, not the
    # converged one); 27 element elastic cube (192 dofs, 24 free) with CG +
    # jacobi at a random state, and a CG Newton step under both operators.
    mixed_solves = krylov_solves(
        fe_problem, params, U_prev, U_prev, xi_prev, step_time,
    )
    fe_27, params_27, U_27 = elastic_krylov_problem()
    elastic_solves = krylov_solves(
        fe_27, params_27, U_27, np.zeros_like(U_27), {}, StepTime(1.0, 0.0),
    )
    U_cg_step = {}
    for kind in ("assembled", "element"):
        U_cg_step[kind], _ = fe_newton_solve(
            fe_27, params_27, U_prev=U_27, t=1.0, t_prev=0.0,
            linear_solver_settings={**_CG_SETTINGS, "operator": kind},
        )

    # The divisor rule: 7 elements (prime) run on one device with a
    # warning; 6 elements over 4 devices shard over 3; two blocks of 1
    # and 2 elements share no divisor, so no mesh.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fe_7 = elastic_problem((7, 1, 1))
    n_warnings_7 = sum("common divisor" in str(w.message) for w in caught)
    fe_6 = elastic_problem((6, 1, 1))
    fe_1_2 = elastic_problem((3, 1, 1), two_blocks=True)
    K7, R7 = assemble_at_random_state(fe_7, fe_7.kernel_arrays)
    K7_ref, R7_ref = assemble_at_random_state(
        fe_7, _on_device_0(fe_7.kernel_arrays),
    )
    K6, R6 = assemble_at_random_state(fe_6, fe_6.kernel_arrays)
    K6_ref, R6_ref = assemble_at_random_state(
        fe_6, _on_device_0(fe_6.kernel_arrays),
    )

    # num_devices limits the mesh to the first num_devices devices.
    capped_2 = build_device_mesh({"all": 8}, num_devices=2)
    capped_1 = build_device_mesh({"all": 8}, num_devices=1)
    try:
        build_device_mesh({"all": 8}, num_devices=n_devices + 1)
        too_many_raises = False
    except ValueError:
        too_many_raises = True

    np.savez(
        path,
        n_devices=n_devices,
        mesh_size=_mesh_size(fe_problem),
        n_element_leaves=len(element_leaves),
        n_element_leaves_sharded=sum(map(is_element_sharded, element_leaves)),
        n_other_leaves_sharded=sum(map(is_element_sharded, other_leaves)),
        xi_placed_sharded=is_element_sharded(xi_placed["all"]),
        U=U, xi=xi["all"],
        K_data=K_data, R=R, xi_out=xi_out,
        K_data_ref=K_data_ref, R_ref=R_ref, xi_out_ref=xi_out_ref,
        element_matvec=element_matvec, element_matvec_ref=element_matvec_ref,
        U_element_step=np.asarray(U_element_step),
        mixed_iters_assembled=mixed_solves["assembled"][0],
        mixed_x_assembled=mixed_solves["assembled"][1],
        mixed_iters_element=mixed_solves["element"][0],
        mixed_x_element=mixed_solves["element"][1],
        elastic_iters_assembled=elastic_solves["assembled"][0],
        elastic_x_assembled=elastic_solves["assembled"][1],
        elastic_iters_element=elastic_solves["element"][0],
        elastic_x_element=elastic_solves["element"][1],
        U_cg_step_assembled=np.asarray(U_cg_step["assembled"]),
        U_cg_step_element=np.asarray(U_cg_step["element"]),
        mesh_size_7=_mesh_size(fe_7), n_warnings_7=n_warnings_7,
        mesh_size_6=_mesh_size(fe_6), mesh_size_1_2=_mesh_size(fe_1_2),
        capped_2_size=0 if capped_2 is None else int(capped_2.size),
        capped_1_is_none=capped_1 is None, too_many_raises=too_many_raises,
        K7=K7, R7=R7, K7_ref=K7_ref, R7_ref=R7_ref,
        K6=K6, R6=R6, K6_ref=K6_ref, R6_ref=R6_ref,
    )


def _assert_close(actual, reference, rel: float) -> None:
    scale = float(np.abs(reference).max())
    assert scale > 0.0
    np.testing.assert_allclose(actual, reference, rtol=0, atol=rel * scale)


class TestElementShardCount(unittest.TestCase):

    def test_largest_common_divisor_that_fits(self) -> None:
        cases = [
            ({"all": 8}, 4, 4),
            ({"all": 7}, 4, 1),
            ({"all": 6}, 4, 3),
            ({"left": 1, "right": 2}, 4, 1),
            ({"left": 4, "right": 4}, 4, 4),
            ({"all": 8}, 1, 1),
            # The 2D and 3D XT10 meshes: 2387 = 7 * 11 * 31 triangles,
            # 8184 tets.
            ({"all": 2387}, 8, 7),
            ({"all": 2387}, 12, 11),
            ({"all": 8184}, 8, 8),
        ]
        for counts, n_devices, expected in cases:
            with self.subTest(counts=counts, n_devices=n_devices):
                self.assertEqual(
                    element_shard_count(counts, n_devices), expected,
                )


@unittest.skipUnless(
    len(jax.devices()) == 1, "the suite's process has several devices",
)
class TestOneDevice(unittest.TestCase):

    def test_nothing_is_placed(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(build_device_mesh({"all": 8}))
        self.assertEqual(caught, [])
        xi = {"all": jnp.arange(24.0).reshape(4, 3, 2)}
        self.assertIs(place_element_leaves(xi, None), xi)
        fe_problem = elastic_problem((2, 1, 1))
        self.assertIsNone(fe_problem.device_mesh)
        self.assertIs(
            fe_problem.kernel_arrays.geometry_cache, fe_problem.geometry_cache,
        )
        self.assertIsNone(build_device_mesh({"all": 8}, num_devices=1))
        with self.assertRaises(ValueError):
            build_device_mesh({"all": 8}, num_devices=2)


@unittest.skipUnless(
    len(jax.devices()) == 1, "the suite's process has several devices",
)
class TestFourDevices(unittest.TestCase):
    """The results of :func:`save_sharded_results` on four CPU devices."""

    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sharded.npz"
            proc = subprocess.run(
                [sys.executable, "-c", _SUBPROCESS_SCRIPT,
                 str(_N_DEVICES), str(out)],
                cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"the {_N_DEVICES} device subprocess failed:\n"
                    f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
                )
            with np.load(out) as data:
                cls.sharded = {k: data[k] for k in data.files}
        cls.drive = drive_mixed_problem()
        _, _, cls.U_ref, _, cls.xi_ref, _, _ = cls.drive

    def test_placement(self) -> None:
        r = self.sharded
        self.assertEqual(int(r["n_devices"]), _N_DEVICES)
        self.assertEqual(int(r["mesh_size"]), _N_DEVICES)
        self.assertEqual(int(r["n_element_leaves"]), _N_ELEMENT_LEAVES)
        self.assertEqual(
            int(r["n_element_leaves_sharded"]), _N_ELEMENT_LEAVES,
        )
        self.assertEqual(int(r["n_other_leaves_sharded"]), 0)
        self.assertTrue(bool(r["xi_placed_sharded"]))

    def test_assembly_matches_one_device(self) -> None:
        r = self.sharded
        for name in ("K_data", "R", "xi_out"):
            with self.subTest(name=name):
                _assert_close(r[name], r[f"{name}_ref"], 1e-14)

    def test_newton_solve_matches_one_device_process(self) -> None:
        _assert_close(self.sharded["U"], self.U_ref, 1e-12)
        _assert_close(self.sharded["xi"], self.xi_ref["all"], 1e-12)

    def test_element_operator_on_four_devices(self) -> None:
        r = self.sharded
        _assert_close(r["element_matvec"], r["element_matvec_ref"], 1e-12)
        fe_problem, params, _U, U_prev, _xi, xi_prev, step_time = self.drive
        U_ref, _ = fe_newton_solve(
            fe_problem, params, U_prev=U_prev, t=float(step_time.t),
            t_prev=float(step_time.t_prev), xi_prev_by_block=xi_prev,
            linear_solver_settings=_ELEMENT_SOLVER_SETTINGS,
        )
        _assert_close(r["U_element_step"], np.asarray(U_ref), 1e-8)

    def test_partitioned_krylov_solves_match_one_device(self) -> None:
        r = self.sharded
        fe_problem, params, _U, U_prev, _xi, xi_prev, step_time = self.drive
        mixed = krylov_solves(
            fe_problem, params, U_prev, U_prev, xi_prev, step_time,
        )
        fe_27, params_27, U_27 = elastic_krylov_problem()
        elastic = krylov_solves(
            fe_27, params_27, U_27, np.zeros_like(U_27), {}, StepTime(1.0, 0.0),
        )
        for name, solves in (("mixed", mixed), ("elastic", elastic)):
            for kind in ("assembled", "element"):
                with self.subTest(problem=name, operator=kind):
                    iterations, x = solves[kind]
                    self.assertGreater(iterations, 1)
                    self.assertEqual(
                        int(r[f"{name}_iters_{kind}"]), iterations,
                    )
                    _assert_close(r[f"{name}_x_{kind}"], x, 1e-8)
        # The step drives the random start to the exact answer, zero, so the
        # comparison is scaled by the start, not by the answer.
        for kind in ("assembled", "element"):
            U_step, _ = fe_newton_solve(
                fe_27, params_27, U_prev=U_27, t=1.0, t_prev=0.0,
                linear_solver_settings={**_CG_SETTINGS, "operator": kind},
            )
            with self.subTest(operator=kind):
                np.testing.assert_allclose(
                    r[f"U_cg_step_{kind}"], np.asarray(U_step), rtol=0,
                    atol=1e-8 * float(np.abs(U_27).max()),
                )

    def test_divisor_rule(self) -> None:
        r = self.sharded
        self.assertEqual(int(r["mesh_size_7"]), 0)
        self.assertEqual(int(r["n_warnings_7"]), 1)
        _assert_close(r["K7"], r["K7_ref"], 1e-14)
        _assert_close(r["R7"], r["R7_ref"], 1e-14)
        self.assertEqual(int(r["mesh_size_6"]), 3)
        _assert_close(r["K6"], r["K6_ref"], 1e-14)
        _assert_close(r["R6"], r["R6_ref"], 1e-14)
        self.assertEqual(int(r["mesh_size_1_2"]), 0)

    def test_num_devices_limits_the_mesh(self) -> None:
        r = self.sharded
        self.assertEqual(int(r["capped_2_size"]), 2)
        self.assertTrue(bool(r["capped_1_is_none"]))
        self.assertTrue(bool(r["too_many_raises"]))


if __name__ == "__main__":
    unittest.main()
