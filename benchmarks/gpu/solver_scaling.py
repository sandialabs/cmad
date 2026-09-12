"""Mesh refinement sweep of a forward solve on one device.

For each mesh size a notch mesh is generated, the base input file is
rewritten onto it (mesh, element block name, step count, no output
files), the forward solve is compiled ahead of time and run once warm,
and one row reports the element and equation counts, the compile and
warm times, and the Newton iterations per step. The input file decides
the solver and the convergence prints, so with both prints on the log
carries every Newton iteration's Krylov count.

    python benchmarks/gpu/solver_scaling.py [--sizes H ...] [--steps N]
        [--input FILE] [--work-dir DIR]

Paths default relative to the installed cmad source tree, and the
arguments are parsed leniently, so the script also runs as sent to a
remote kernel.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import yaml
from jax import block_until_ready, default_backend, devices, jit

import cmad
from cmad.cli.common import build_fe_problem_from_deck, nonlinear_solver_settings
from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.driver import build_fe_quasistatic_trajectory
from cmad.fem.fe_problem import FEState
from cmad.fem.sharding import place_element_leaves

REPO_ROOT = Path(cmad.__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from examples.notch_mesh import generate_notch_msh  # noqa: E402

MESH_SIZES = (0.06, 0.045, 0.035, 0.028, 0.02)


def rewrite_input(
        base: dict[str, Any], mesh_path: Path, num_steps: int, work_dir: Path,
        linear_solver: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The base input file on ``mesh_path``: its single material block
    renamed to the generated mesh's ``solid``, ``num_steps`` steps, an
    output section that writes nothing, and ``linear_solver`` in place of
    its linear solver section when one is given."""
    deck = yaml.safe_load(yaml.safe_dump(base))
    deck["discretization"]["mesh file"] = str(mesh_path)
    deck["discretization"]["num steps"] = num_steps
    local = deck["residuals"]["local residual"]
    (_, material), = local["materials"].items()
    local["materials"] = {"solid": material}
    deck["output"] = {"path": str(work_dir), "write exodus": False}
    if linear_solver is not None:
        deck["linear solver"] = linear_solver
    return deck


def trajectory_inputs(fe_problem: Any, t_schedule: list[float]) -> tuple[Any, ...]:
    """``(params_by_block, state_init, fe_arrays, t_schedule_jax)`` for the
    trajectory closure, as the driver builds them."""
    state = FEState.from_problem(fe_problem, t_init=t_schedule[0])
    dbc_arrays = fe_problem.kernel_arrays.dbc_arrays
    for t in t_schedule[1:]:
        fe_problem.dof_map.evaluate_prescribed_values(dbc_arrays, t)
    params_by_block = params_by_block_from_models(fe_problem)
    u_init = jnp.asarray(state.U_at(0), dtype=jnp.float64)
    xi_init = place_element_leaves(
        {b: jnp.asarray(state.xi_at(0, b)) for b in fe_problem.models_by_block},
        fe_problem.device_mesh,
    )
    t_schedule_jax = jnp.asarray(t_schedule, dtype=jnp.float64)
    return params_by_block, (u_init, xi_init), fe_problem.kernel_arrays, t_schedule_jax


def run_once(
        fe_problem: Any, inputs: tuple[Any, ...],
        nls: dict[str, Any], lss: dict[str, Any],
) -> tuple[float, float, int, float, list[int]]:
    """Compile the trajectory ahead of time and run it once warm. Returns
    ``(compile_seconds, warm_seconds, first_failed_step,
    first_failed_rel_norm, iters_per_step)``."""
    params_by_block, state_init, fe_arrays, t_schedule_jax = inputs
    trajectory = build_fe_quasistatic_trajectory(
        fe_problem, nonlinear_solver_settings=nls, linear_solver_settings=lss,
    )

    def run(params: Any, state: Any, arrays: Any) -> Any:
        return trajectory(arrays, params, state, t_schedule_jax)

    start = time.perf_counter()
    compiled = jit(run).lower(params_by_block, state_init, fe_arrays).compile()
    compile_s = time.perf_counter() - start
    memory = compiled.memory_analysis()
    if memory is not None:
        gib = 1024.0 ** 3
        print(
            f"executable memory: temporaries {memory.temp_size_in_bytes / gib:.2f} GiB, "
            f"arguments {memory.argument_size_in_bytes / gib:.2f} GiB, "
            f"outputs {memory.output_size_in_bytes / gib:.2f} GiB",
            flush=True,
        )

    start = time.perf_counter()
    out = compiled(params_by_block, state_init, fe_arrays)
    block_until_ready(out)
    warm_s = time.perf_counter() - start
    _, _, _, first_failed_step, first_failed_rel_norm, iters_per_step = out
    return (
        compile_s, warm_s, int(first_failed_step), float(first_failed_rel_norm),
        np.asarray(iters_per_step).tolist(),
    )


def run_sweep(
        sizes: tuple[float, ...], base_path: Path, num_steps: int,
        work_dir: Path, linear_solver_path: Path | None = None,
) -> None:
    """One row per mesh size."""
    work_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(base_path.read_text())
    linear_solver = None
    if linear_solver_path is not None:
        linear_solver = yaml.safe_load(linear_solver_path.read_text())["linear solver"]
    print(f"backend {default_backend()}, devices {devices()}")
    print(f"input file {base_path}, {num_steps} steps")
    print(f"linear solver {linear_solver_path or 'from the input file'}\n")
    for h in sizes:
        mesh_path = work_dir / f"notch_h{h:.3f}.msh"
        n_elem = generate_notch_msh(mesh_path, h)
        deck_path = work_dir / f"notch_h{h:.3f}.yaml"
        deck_path.write_text(yaml.safe_dump(rewrite_input(
            base, mesh_path, num_steps, work_dir, linear_solver,
        )))
        bundle = build_fe_problem_from_deck(deck_path, "primal")
        fe_problem = bundle.fe_problem
        gr_section = bundle.resolved["residuals"]["global residual"]
        nls = nonlinear_solver_settings(
            gr_section, bool(gr_section.get("print convergence", False)),
        )
        lss = bundle.resolved["linear solver"]
        inputs = trajectory_inputs(fe_problem, bundle.t_schedule.tolist())
        n_eq = int(fe_problem.num_dofs_padded)
        print(f"h={h:.3f}: {n_elem} tets, {n_eq} equations", flush=True)
        compile_s, warm_s, failed_step, failed_rel_norm, iters = run_once(
            fe_problem, inputs, nls, lss,
        )
        row = (
            f"h={h:.3f} ({n_elem:>7d} tets, {n_eq:>8d} equations)  "
            f"compile {compile_s:7.2f}s  warm {warm_s:8.2f}s  "
            f"Newton iterations per step {iters}"
        )
        if failed_step >= 0:
            row += (
                f"  FAILED at step {failed_step + 1}, relative residual "
                f"{failed_rel_norm:.3e}"
            )
        print(row, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=float, nargs="+", default=list(MESH_SIZES),
        help="mesh sizes to sweep",
    )
    parser.add_argument(
        "--steps", type=int, default=4, help="load steps per size (default 4)",
    )
    parser.add_argument(
        "--input", type=Path, default=REPO_ROOT / "examples" / "notch_mixed.yaml",
        help="base input file (default examples/notch_mixed.yaml)",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=REPO_ROOT / "benchmarks" / "gpu" / "run",
        help="where the meshes and input files go (default benchmarks/gpu/run)",
    )
    parser.add_argument(
        "--linear-solver", type=Path, default=None,
        help="yaml file whose 'linear solver' section replaces the input file's",
    )
    args, _unknown = parser.parse_known_args()
    run_sweep(
        tuple(args.sizes), args.input, args.steps, args.work_dir,
        args.linear_solver,
    )


if __name__ == "__main__":
    main()
