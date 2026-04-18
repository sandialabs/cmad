"""``cmad`` CLI entry point.

Argparse dispatcher for the driver subcommands. The registries
lazy-load concrete Model / QoI modules on first resolution, so no
side-effect imports are needed here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cmad.cli.gradient import run_gradient
from cmad.cli.objective import run_objective
from cmad.cli.primal import run_primal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cmad")
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

    args = parser.parse_args(argv)

    if args.subcommand == "primal":
        return run_primal(args.deck)
    if args.subcommand == "objective":
        return run_objective(args.deck)
    if args.subcommand == "gradient":
        return run_gradient(args.deck)
    return 2


if __name__ == "__main__":
    sys.exit(main())
