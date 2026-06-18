"""Block solver defaults sweep on the mixed u-p notch forward solve.

Builds the FE problem once per mesh, then for each linear solver config compiles
the forward solve ahead of time and times one warm run, reporting compile and
warm times separately.

    python benchmarks/block_solver/benchmark.py
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from cmad.cli.common import build_fe_problem_from_deck  # noqa: E402
from cmad.fem.assembly import params_by_block_from_models  # noqa: E402
from cmad.fem.driver import build_fe_quasistatic_trajectory  # noqa: E402
from cmad.fem.fe_problem import FEState  # noqa: E402
from examples.notch_mesh import generate_notch_msh  # noqa: E402

_GMRES = {"type": "gmres", "rtol": 1.0e-8, "restart": 50, "max iters": 500}
_NLS = {
    "max iters": 15, "abs tol": 1.0e-9, "rel tol": 1.0e-9,
    "print convergence": False, "line search": {},
}


def _block(coupling: str, diagonal_block: str, inner: str, **extra: Any) -> dict:
    return {
        **_GMRES,
        "preconditioner": {
            "type": "block", "coupling": coupling,
            "diagonal_block": diagonal_block, "inner": inner, **extra,
        },
    }


# Each entry is (label, linear solver settings). jacobi takes only the assembled
# diagonal block, while chebyshev and amg take either.
CONFIGS: list[tuple[str, dict]] = [
    ("direct", {"type": "direct"}),
    ("block jacobi (assembled)", _block("lower", "assembled", "jacobi")),
    ("block amg (schur)", _block("lower", "schur", "amg")),
    ("block chebyshev d3 (assembled)",
     _block("lower", "assembled", "chebyshev", degree=3)),
    ("block chebyshev d3 (schur)",
     _block("lower", "schur", "chebyshev", degree=3)),
]

MESH_SIZES = (0.12, 0.07)


def mixed_notch_deck(mesh_path: Path, num_steps: int) -> dict:
    """Mixed u-p J2 plastic notch problem on ``mesh_path``. The deck's direct
    solver clears the build guard, and each config overrides it per run.
    """
    return {
        "problem": {"type": "fe", "name": "notch_mixed"},
        "discretization": {
            "mesh file": str(mesh_path),
            "build coordinate sidesets": True,
            "num steps": num_steps,
            "step size": 0.3,
        },
        "linear solver": {"type": "direct"},
        "residuals": {
            "global residual": {
                "type": "small_disp_equilibrium",
                "def_type": "full_3d",
                "mixed": True,
                "stabilization multiplier": 1.0,
                "nonlinear max iters": _NLS["max iters"],
                "nonlinear absolute tol": _NLS["abs tol"],
                "nonlinear relative tol": _NLS["rel tol"],
            },
            "local residual": {
                "type": "small_elastic_plastic",
                "nonlinear max iters": 100,
                "nonlinear absolute tol": 1.0e-12,
                "nonlinear relative tol": 1.0e-12,
                "materials": {
                    "solid": {
                        "elastic": {"E": 200000.0, "nu": 0.3},
                        "plastic": {
                            "effective stress": {"J2": {}},
                            "flow stress": {
                                "initial yield": {"Y": 200.0},
                                "hardening": {"voce": {"S": 200.0, "D": 20.0}},
                            },
                        },
                    },
                },
            },
        },
        "dirichlet bcs": {
            "expression": {
                "sym_x": ["equilibrium", 0, "xmin_sides", "0.0"],
                "sym_y": ["equilibrium", 1, "ymin_sides", "0.0"],
                "sym_z": ["equilibrium", 2, "zmin_sides", "0.0"],
                "load_x": ["equilibrium", 0, "xmax_sides", "0.005 * t"],
            },
        },
        "output": {
            "path": "out",
            "exodus filename": "notch_mixed.exo",
            "global residual": ["u"],
            "local residual": {"solid": ["cauchy"]},
        },
    }


def setup_inputs(fe_problem: Any, t_schedule: list[float]) -> tuple[Any, ...]:
    """Build the JAX trajectory inputs the FE problem reuses across configs:
    ``(params_by_block, state_init, fe_arrays, t_schedule_jax)``.
    """
    state = FEState.from_problem(fe_problem, t_init=t_schedule[0])
    dbc_arrays = fe_problem.kernel_arrays.dbc_arrays
    for t in t_schedule[1:]:
        fe_problem.dof_map.evaluate_prescribed_values(dbc_arrays, t)
    params_by_block = params_by_block_from_models(fe_problem)
    fe_arrays = fe_problem.kernel_arrays
    u_init = jnp.asarray(state.U_at(0), dtype=jnp.float64)
    xi_init = {
        b: jnp.asarray(state.xi_at(0, b)) for b in fe_problem.models_by_block
    }
    t_schedule_jax = jnp.asarray(t_schedule, dtype=jnp.float64)
    return params_by_block, (u_init, xi_init), fe_arrays, t_schedule_jax


def time_config(
        fe_problem: Any, inputs: tuple[Any, ...], linear_solver: dict,
) -> tuple[float, float]:
    """Compile the trajectory for ``linear_solver`` ahead of time, then run it
    once warm. Return ``(compile_seconds, warm_seconds)``.
    """
    params_by_block, state_init, fe_arrays, t_schedule_jax = inputs
    trajectory = build_fe_quasistatic_trajectory(
        fe_problem, nonlinear_solver_settings=_NLS,
        linear_solver_settings=linear_solver,
    )

    def run(params: Any, state: Any, arrays: Any) -> Any:
        return trajectory(arrays, params, state, t_schedule_jax)

    start = time.perf_counter()
    compiled = jax.jit(run).lower(
        params_by_block, state_init, fe_arrays,
    ).compile()
    compile_s = time.perf_counter() - start

    start = time.perf_counter()
    out = compiled(params_by_block, state_init, fe_arrays)
    jax.block_until_ready(out)
    warm_s = time.perf_counter() - start
    return compile_s, warm_s


def run_sweep(
        sizes: tuple[float, ...], configs: list[tuple[str, dict]],
        work_dir: Path, num_steps: int = 3,
) -> None:
    """Generate a notch mesh per size, build the FE problem, and time every
    config on it, printing one row per run.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    deck_path = work_dir / "deck.yaml"
    for h in sizes:
        mesh_path = work_dir / f"notch_h{h:.3f}.msh"
        n_elem = generate_notch_msh(mesh_path, h)
        deck_path.write_text(
            yaml.safe_dump(mixed_notch_deck(mesh_path, num_steps)),
        )
        with redirect_stdout(io.StringIO()):
            bundle = build_fe_problem_from_deck(deck_path, "primal")
            inputs = setup_inputs(bundle.fe_problem, bundle.t_schedule.tolist())
        for label, linear_solver in configs:
            try:
                with redirect_stdout(io.StringIO()):
                    compile_s, warm_s = time_config(
                        bundle.fe_problem, inputs, linear_solver,
                    )
                row = f"warm {warm_s:8.3f}s   compile {compile_s:7.2f}s"
            except Exception as exc:
                row = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"  h={h:.3f} ({n_elem:>6d} tets)  {label:<32s}  {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=float, nargs="+", default=list(MESH_SIZES),
        help="mesh sizes to sweep",
    )
    args = parser.parse_args()
    print("block solver defaults sweep (mixed u-p plastic notch)\n")
    run_sweep(tuple(args.sizes), CONFIGS, _HERE / "run")


if __name__ == "__main__":
    main()
