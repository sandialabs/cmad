"""``cmad`` CLI entry point.

Argparse dispatcher for the driver subcommands. The registries
lazy load concrete Model / QoI modules on first resolution, so no
side effect imports are needed here. The subcommand modules are imported
inside :func:`main` after the command line is parsed, so that
``--devices`` is applied before anything initialises the JAX backend
(the CPU device count can only be set before that).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jax import config as jax_config
from jax import devices as jax_devices

from cmad.fem.sharding import set_device_count


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"expected an integer >= 1, got {text}",
        )
    return value


def _apply_device_count(n_devices: int) -> None:
    """Run on ``n_devices`` devices; the FE assembly is sharded across them
    (:mod:`cmad.fem.sharding`). On CPU, JAX makes that many devices; on a
    machine with accelerators, the run uses the first ``n_devices`` of
    them. JAX reads the CPU count only before its backend initialises.
    When JAX is already running in this process, the count cannot be
    changed, and asking for more devices than JAX has is an error."""
    if n_devices > 1:
        try:
            jax_config.update("jax_num_cpu_devices", n_devices)
        except RuntimeError as exc:
            present = len(jax_devices())
            if n_devices > present:
                raise RuntimeError(
                    f"--devices {n_devices}: JAX is already running with "
                    f"{present} device(s) in this process; set the device "
                    f"count before JAX starts",
                ) from exc
    set_device_count(n_devices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cmad")
    parser.add_argument(
        "--devices", type=_positive_int, default=None, metavar="N",
        help="run on N devices and shard the FE assembly across them: N CPU "
             "devices, or the first N accelerators; without it every device "
             "JAX sees is used",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    primal = sub.add_parser("primal", help="Run a forward (primal) solve.")
    primal.add_argument("deck", type=Path, help="Path to the YAML deck.")

    objective = sub.add_parser(
        "objective",
        help="Run a forward solve and accumulate the QoI J.",
    )
    objective.add_argument("deck", type=Path, help="Path to the YAML deck.")

    gradient = sub.add_parser(
        "gradient",
        help="Compute (J, grad) via the chosen sensitivity strategy.",
    )
    gradient.add_argument("deck", type=Path, help="Path to the YAML deck.")

    hessian = sub.add_parser(
        "hessian",
        help="Compute (J, grad, hess) via direct_adjoint or jvp.",
    )
    hessian.add_argument("deck", type=Path, help="Path to the YAML deck.")

    calibrate = sub.add_parser(
        "calibrate",
        help="Optimize active parameters against the QoI via scipy.",
    )
    calibrate.add_argument("deck", type=Path, help="Path to the YAML deck.")

    args = parser.parse_args(argv)
    if args.devices is not None:
        _apply_device_count(args.devices)

    # Imported here so the device count above precedes the backend
    # initialisation these modules trigger on import.
    from cmad.cli.calibrate import run_calibrate
    from cmad.cli.gradient import run_gradient
    from cmad.cli.hessian import run_hessian
    from cmad.cli.objective import run_objective
    from cmad.cli.primal import run_primal

    if args.subcommand == "primal":
        return run_primal(args.deck)
    if args.subcommand == "objective":
        return run_objective(args.deck)
    if args.subcommand == "gradient":
        return run_gradient(args.deck)
    if args.subcommand == "hessian":
        return run_hessian(args.deck)
    if args.subcommand == "calibrate":
        return run_calibrate(args.deck)
    return 2


if __name__ == "__main__":
    sys.exit(main())
