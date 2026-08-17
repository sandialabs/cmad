"""``cmad`` CLI entry point.

Argparse dispatcher for the driver subcommands. The registries
lazy load concrete Model / QoI modules on first resolution, so no
side effect imports are needed here. The subcommand modules are imported
inside :func:`main` after the command line is parsed, so that
``--cpu-devices`` can set JAX's CPU device count before anything
initialises the JAX backend.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from jax import config as jax_config
from jax import devices as jax_devices


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"expected an integer >= 1, got {text}",
        )
    return value


def _apply_cpu_devices(n_devices: int) -> None:
    """Run JAX with ``n_devices`` CPU devices; the FE assembly is sharded
    across them (:mod:`cmad.fem.sharding`). JAX reads the count only before
    its backend initialises, so a caller that ran JAX in this process before
    ``main`` keeps the devices it has, with a warning. 1 leaves JAX's own
    default alone."""
    if n_devices == 1:
        return
    try:
        jax_config.update("jax_num_cpu_devices", n_devices)
    except RuntimeError:
        warnings.warn(
            f"--cpu-devices {n_devices} ignored: JAX is already running "
            f"with {len(jax_devices())} device(s) in this process",
            stacklevel=2,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cmad")
    parser.add_argument(
        "--cpu-devices", type=_positive_int, default=1, metavar="N",
        help="run JAX with N CPU devices and shard the FE assembly across "
             "them (default 1)",
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
    _apply_cpu_devices(args.cpu_devices)

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
