"""Assembly timing on one device: R plus K against R only.

For each mesh size a notch mesh is generated, step 1 is solved to a
converged plastic state (the direct solver unless a linear solver file
says otherwise; it only produces the state), and the assembly at step
2's extrapolated starting point, the representative Newton iteration,
and at the elastic start is timed three ways: the Newton body's R plus
K assembly with the embedded boundary conditions, the per element
tangent without the dedup scatter, and the residual only assembly.
Each is jitted, run warm, and reported as the median over the repeats
with its executable memory and its flop and byte estimates. Variants
of the local solver settings (no local line search, a clamped local
iteration count) time the assemblies again at the plastic state, and
one profiler trace of the R plus K assembly is summed by op name.

    python benchmarks/gpu/assembly_timing.py [--sizes H ...]
        [--input FILE] [--linear-solver FILE] [--work-dir DIR]
        [--repeats N]

Paths default relative to the installed cmad source tree, and the
arguments are parsed leniently, so the script also runs as sent to a
remote kernel.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import yaml
from jax import block_until_ready, default_backend, devices, jit, profiler

import cmad
from cmad.cli.common import build_fe_problem_from_deck, nonlinear_solver_settings
from cmad.fem.assembly import assemble_element_tangent, assemble_global_residual
from cmad.fem.driver import build_fe_quasistatic_trajectory
from cmad.fem.nonlinear_solver import _assemble_tangent_and_residual, _solve_linear
from cmad.models.global_fields import StepTime

REPO_ROOT = Path(cmad.__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gpu.solver_scaling import rewrite_input, trajectory_inputs  # noqa: E402
from examples.notch_mesh import generate_notch_msh  # noqa: E402

GIB = 1024.0 ** 3
LOCAL_VARIANTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("local line search off", {"line search": {"max evals": 0}}),
    ("local max iters 5", {"nonlinear max iters": 5}),
    ("local max iters 10", {"nonlinear max iters": 10}),
)


def build_problem(deck: dict[str, Any], deck_path: Path) -> tuple[Any, Any, Any]:
    """``(bundle, nls, lss)`` for the input file ``deck`` written to
    ``deck_path``."""
    deck_path.write_text(yaml.safe_dump(deck))
    bundle = build_fe_problem_from_deck(deck_path, "primal")
    gr_section = bundle.resolved["residuals"]["global residual"]
    nls = nonlinear_solver_settings(
        gr_section, bool(gr_section.get("print convergence", False)),
    )
    return bundle, nls, bundle.resolved["linear solver"]


def solve_first_step(
        fe_problem: Any, inputs: tuple[Any, ...], nls: dict[str, Any],
        lss: dict[str, Any], trace_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], list[int], float | None]:
    """``(U1, xi1, iters, traced_wall)`` after the schedule's first step;
    raises when it did not converge. With ``trace_dir`` the warm step
    runs once more under the profiler and ``traced_wall`` is its wall
    time in seconds, ``None`` otherwise."""
    params_by_block, state_init, fe_arrays, t_schedule_jax = inputs
    trajectory = build_fe_quasistatic_trajectory(
        fe_problem, nonlinear_solver_settings=nls, linear_solver_settings=lss,
    )

    def run(params: Any, state: Any, arrays: Any) -> Any:
        return trajectory(arrays, params, state, t_schedule_jax)

    compiled = jit(run).lower(params_by_block, state_init, fe_arrays).compile()
    out = compiled(params_by_block, state_init, fe_arrays)
    block_until_ready(out)
    U_steps, xi_steps, _J, failed_step, failed_rel_norm, iters = out
    if int(failed_step) >= 0:
        raise RuntimeError(
            f"step 1 did not converge: relative residual "
            f"{float(failed_rel_norm):.3e}"
        )
    traced_wall = None
    if trace_dir is not None:
        start = time.perf_counter()
        with profiler.trace(str(trace_dir)):
            block_until_ready(compiled(params_by_block, state_init, fe_arrays))
        traced_wall = time.perf_counter() - start
    return (
        U_steps[0], {b: xi[0] for b, xi in xi_steps.items()},
        np.asarray(iters).tolist(), traced_wall,
    )


def assembly_fns(fe_problem: Any) -> dict[str, Any]:
    """The three assemblies as functions of ``(params, U, U_prev, xi_prev,
    fe_arrays, t, t_prev)``: ``full`` is the Newton body's R plus K with
    the embedded boundary conditions, ``element`` the per element
    tangent, ``residual`` R only."""

    def full(params: Any, U: Any, U_prev: Any, xi_prev: Any, arrays: Any,
             t: Any, t_prev: Any) -> Any:
        presc_vals = jnp.asarray(
            fe_problem.dof_map.evaluate_prescribed_values(arrays.dbc_arrays, t),
        )
        return _assemble_tangent_and_residual(
            fe_problem, arrays, params, U, U_prev, StepTime(t, t_prev),
            xi_prev, presc_vals, "assembled",
        )

    def element(params: Any, U: Any, U_prev: Any, xi_prev: Any, arrays: Any,
                t: Any, t_prev: Any) -> Any:
        return assemble_element_tangent(
            fe_problem, arrays, params, U, U_prev, StepTime(t, t_prev),
            xi_prev_by_block=xi_prev,
        )

    def residual(params: Any, U: Any, U_prev: Any, xi_prev: Any, arrays: Any,
                 t: Any, t_prev: Any) -> Any:
        return assemble_global_residual(
            fe_problem, arrays, params, U, U_prev, StepTime(t, t_prev),
            xi_prev_by_block=xi_prev,
        )

    return {"full": full, "element": element, "residual": residual}


def time_call(
        fn: Any, args: tuple[Any, ...], repeats: int,
) -> tuple[float, Any, Any]:
    """``(median ms, compiled, output)`` of ``fn`` on ``args``, compiled
    ahead of time and run once warm before the timed repeats."""
    compiled = jit(fn).lower(*args).compile()
    out = compiled(*args)
    block_until_ready(out)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = compiled(*args)
        block_until_ready(out)
        times.append(time.perf_counter() - start)
    return 1e3 * float(np.median(times)), compiled, out


def executable_summary(compiled: Any) -> str:
    """Temporaries, flops, and bytes accessed of a compiled executable."""
    parts = []
    memory = compiled.memory_analysis()
    if memory is not None:
        parts.append(f"temporaries {memory.temp_size_in_bytes / GIB:.2f} GiB")
    cost = compiled.cost_analysis()
    if isinstance(cost, list):
        cost = cost[0] if cost else None
    if cost:
        flops = cost.get("flops")
        nbytes = cost.get("bytes accessed")
        if flops is not None:
            parts.append(f"flops {flops:.2e}")
        if nbytes is not None:
            parts.append(f"bytes {nbytes:.2e}")
    return ", ".join(parts)


def trace_summary(trace_dir: Path, top: int) -> list[str]:
    """The ``top`` ops of the newest trace under ``trace_dir`` by summed
    duration, from the device's XLA op events when the trace has them
    and from every event otherwise."""
    files = sorted(trace_dir.rglob("*.trace.json.gz"), key=lambda p: p.stat().st_mtime)
    if not files:
        return ["no trace file found"]
    with gzip.open(files[-1], "rt") as f:
        events = json.load(f)["traceEvents"]
    process_names: dict[int, str] = {}
    thread_names: dict[tuple[int, int], str] = {}
    for e in events:
        if e.get("ph") != "M":
            continue
        if e.get("name") == "process_name":
            process_names[e["pid"]] = e["args"]["name"]
        elif e.get("name") == "thread_name":
            thread_names[(e["pid"], e["tid"])] = e["args"]["name"]
    device_pids = {
        pid for pid, name in process_names.items()
        if "GPU" in name or "device" in name.lower()
    }
    op_threads = {
        key for key, name in thread_names.items()
        if key[0] in device_pids and "XLA Ops" in name
    }
    lines = [
        f"trace {files[-1].name}: device processes "
        f"{[process_names[p] for p in sorted(device_pids)]}, "
        f"{len(op_threads)} XLA op threads",
    ]
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for e in events:
        if e.get("ph") != "X":
            continue
        key = (e.get("pid"), e.get("tid"))
        if op_threads and key not in op_threads:
            continue
        if not op_threads and device_pids and key[0] not in device_pids:
            continue
        name = e["name"]
        totals[name] = totals.get(name, 0.0) + float(e.get("dur", 0.0))
        counts[name] = counts.get(name, 0) + 1
    total_us = sum(totals.values())
    copies = {n: us for n, us in totals.items() if n.startswith("Memcpy")}
    lines.append(
        f"summed op time {total_us / 1e3:.1f} ms, of which copies "
        f"{sum(copies.values()) / 1e3:.1f} ms ("
        + ", ".join(f"{n} {us / 1e3:.1f}" for n, us in sorted(copies.items()))
        + ")",
    )
    for name, us in sorted(totals.items(), key=lambda kv: -kv[1])[:top]:
        lines.append(f"  {us / 1e3:9.2f} ms  {counts[name]:6d} x  {name}")
    lines.extend(copy_details(events, device_pids))
    return lines


def event_bytes(e: dict[str, Any]) -> int | None:
    """The byte count a copy event's arguments record, if any."""
    args = e.get("args") or {}
    for key, value in args.items():
        if "byte" in str(key).lower():
            try:
                return int(float(value))
            except (TypeError, ValueError):
                pass
    text = json.dumps(args)
    marker = text.find("size")
    if marker >= 0:
        digits = ""
        for ch in text[marker + 4:]:
            if ch.isdigit():
                digits += ch
            elif digits:
                break
        if digits:
            return int(digits)
    return None


def copy_details(events: list[dict[str, Any]], device_pids: set[int]) -> list[str]:
    """The copy events by kind and size class, and the ten longest with
    the innermost host Python span (the ``$file:line name`` events) that
    contains each, so a copy can be placed inside or outside a callback."""
    classes = (
        (2 ** 20, "under 1 MiB"), (16 * 2 ** 20, "1 to 16 MiB"),
        (64 * 2 ** 20, "16 to 64 MiB"), (256 * 2 ** 20, "64 to 256 MiB"),
    )

    def size_class(nbytes: int | None) -> str:
        if nbytes is None:
            return "unknown size"
        for bound, label in classes:
            if nbytes < bound:
                return label
        return "over 256 MiB"

    by_class: dict[tuple[str, str], tuple[float, int]] = {}
    copies: list[tuple[float, float, int | None, str]] = []
    host_spans: list[tuple[float, float, float, str]] = []
    t0 = min(
        (float(e["ts"]) for e in events if e.get("ph") == "X" and "ts" in e),
        default=0.0,
    )
    for e in events:
        if e.get("ph") != "X":
            continue
        name = str(e.get("name", ""))
        pid = e.get("pid")
        if name.startswith("Memcpy") and (not device_pids or pid in device_pids):
            nbytes = event_bytes(e)
            dur = float(e.get("dur", 0.0))
            key = (name, size_class(nbytes))
            us, count = by_class.get(key, (0.0, 0))
            by_class[key] = (us + dur, count + 1)
            copies.append((dur, float(e["ts"]), nbytes, name))
        elif name.startswith("$") and pid not in device_pids:
            ts = float(e["ts"])
            dur = float(e.get("dur", 0.0))
            host_spans.append((ts, ts + dur, dur, name))
    lines = ["copies by kind and size:"]
    for (name, label), (us, count) in sorted(by_class.items()):
        lines.append(f"  {name} {label}: {count} copies, {us / 1e3:.1f} ms")
    lines.append("longest copies (start s, ms, MiB, innermost host span):")
    for dur, ts, nbytes, name in sorted(copies, reverse=True)[:10]:
        containing = [
            s for s in host_spans if s[0] <= ts and s[1] >= ts + dur
        ]
        span = min(containing, key=lambda s: s[2])[3] if containing else "none"
        size = "?" if nbytes is None else f"{nbytes / 2 ** 20:.1f}"
        lines.append(
            f"  {name} at {(ts - t0) / 1e6:8.3f} s, {dur / 1e3:7.1f} ms, "
            f"{size} MiB, {span}",
        )
    return lines


def run_size(
        h: float, base: dict[str, Any], work_dir: Path,
        linear_solver: dict[str, Any], repeats: int, trace_step: bool,
) -> None:
    mesh_path = work_dir / f"notch_h{h:.3f}.msh"
    n_elem = generate_notch_msh(mesh_path, h)
    deck = rewrite_input(base, mesh_path, 1, work_dir, linear_solver)
    bundle, nls, lss = build_problem(deck, work_dir / f"assembly_h{h:.3f}.yaml")
    fe_problem = bundle.fe_problem
    t_schedule = bundle.t_schedule.tolist()
    inputs = trajectory_inputs(fe_problem, t_schedule)
    params_by_block, state_init, fe_arrays, _ = inputs
    print(
        f"\nh={h:.3f}: {n_elem} tets, {int(fe_problem.num_dofs_padded)} "
        f"equations", flush=True,
    )

    start = time.perf_counter()
    step_trace_dir = work_dir / f"trace_step_h{h:.3f}" if trace_step else None
    U1, xi1, iters, traced_wall = solve_first_step(
        fe_problem, inputs, nls, lss, step_trace_dir,
    )
    print(
        f"step 1: {iters[0]} Newton iterations, {time.perf_counter() - start:.1f} s "
        f"with the compile{' and the traced run' if trace_step else ''}",
        flush=True,
    )
    if step_trace_dir is not None and traced_wall is not None:
        print(
            f"step 1 traced: wall {traced_wall:.2f} s over {iters[0]} Newton "
            f"iterations", flush=True,
        )
        for line in trace_summary(step_trace_dir, 25):
            print(line, flush=True)
    U0, xi0 = state_init
    t0, t1 = float(t_schedule[0]), float(t_schedule[1])
    t2 = t1 + (t1 - t0)
    # Step 2's starting point as the driver extrapolates it: the previous
    # increment scaled by the stride ratio, 1 for a uniform schedule.
    U_guess2 = U1 + ((t2 - t1) / (t1 - t0)) * (U1 - U0)
    states = {
        "elastic": (U0, U0, xi0, jnp.asarray(t1), jnp.asarray(t0)),
        "plastic": (U_guess2, U1, xi1, jnp.asarray(t2), jnp.asarray(t1)),
    }

    fns = assembly_fns(fe_problem)

    def solve(K: Any, arrays: Any, rhs: Any) -> Any:
        return _solve_linear(K, fe_problem, arrays, rhs, lss)

    base_residual: dict[str, Any] = {}
    for state_name, (U, U_prev, xi_prev, t, t_prev) in states.items():
        full_out: Any = None
        for label, fn in fns.items():
            args = (params_by_block, U, U_prev, xi_prev, fe_arrays, t, t_prev)
            ms, compiled, out = time_call(fn, args, repeats)
            print(
                f"{state_name:8s} {label:9s} {ms:9.1f} ms  "
                f"{executable_summary(compiled)}", flush=True,
            )
            if label == "full":
                full_out = out
            if label == "residual":
                base_residual[state_name] = out
        # The linear solve of the assembled system, the whole call: the
        # operator's segment sum, the copies to and from the host, and the
        # callback, against the phases the callback prints.
        r_full, K_full = full_out[0], full_out[1]
        ms, compiled, _ = time_call(solve, (K_full, fe_arrays, -r_full), repeats)
        print(
            f"{state_name:8s} {'solve':9s} {ms:9.1f} ms  "
            f"{executable_summary(compiled)}", flush=True,
        )

    # The local solver variants, at the plastic state, on a rebuilt problem.
    U, U_prev, xi_prev, t, t_prev = states["plastic"]
    R_base = base_residual["plastic"]
    for label, overrides in LOCAL_VARIANTS:
        variant = copy.deepcopy(deck)
        variant["residuals"]["local residual"].update(copy.deepcopy(overrides))
        v_bundle, _, _ = build_problem(
            variant, work_dir / f"assembly_h{h:.3f}_{label.replace(' ', '_')}.yaml",
        )
        v_fns = assembly_fns(v_bundle.fe_problem)
        v_arrays = v_bundle.fe_problem.kernel_arrays
        for fn_label in ("full", "residual"):
            args = (params_by_block, U, U_prev, xi_prev, v_arrays, t, t_prev)
            ms, compiled, out = time_call(v_fns[fn_label], args, repeats)
            note = ""
            if fn_label == "residual":
                diff = float(jnp.linalg.norm(out - R_base) / jnp.linalg.norm(R_base))
                note = f"  relative R difference {diff:.2e}"
            print(
                f"plastic  {fn_label:9s} {ms:9.1f} ms  {executable_summary(compiled)}"
                f"  [{label}]{note}", flush=True,
            )

    # One traced call of the Newton body's assembly at the plastic state.
    trace_dir = work_dir / f"trace_h{h:.3f}"
    args = (params_by_block, U, U_prev, xi_prev, fe_arrays, t, t_prev)
    compiled = jit(fns["full"]).lower(*args).compile()
    block_until_ready(compiled(*args))
    with profiler.trace(str(trace_dir)):
        block_until_ready(compiled(*args))
    for line in trace_summary(trace_dir, 20):
        print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=float, nargs="+", default=[0.06],
        help="mesh sizes (default 0.06)",
    )
    parser.add_argument(
        "--input", type=Path, default=REPO_ROOT / "examples" / "notch_mixed.yaml",
        help="base input file (default examples/notch_mixed.yaml)",
    )
    parser.add_argument(
        "--linear-solver", type=Path, default=None,
        help="yaml file whose 'linear solver' section solves step 1 "
             "(default: the direct solver)",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=REPO_ROOT / "benchmarks" / "gpu" / "run",
        help="where the meshes, input files, and traces go",
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="timed calls per assembly",
    )
    parser.add_argument(
        "--trace-step", action="store_true",
        help="run the warm step 1 once more under the profiler and sum its "
             "device ops by name",
    )
    args, _unknown = parser.parse_known_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(args.input.read_text())
    if args.linear_solver is None:
        linear_solver: dict[str, Any] = {"type": "direct"}
    else:
        linear_solver = yaml.safe_load(args.linear_solver.read_text())["linear solver"]
    print(f"backend {default_backend()}, devices {devices()}")
    print(f"input file {args.input}, linear solver {linear_solver}")
    for h in args.sizes:
        run_size(
            h, base, args.work_dir, linear_solver, args.repeats, args.trace_step,
        )


if __name__ == "__main__":
    main()
