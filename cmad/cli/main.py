"""``cmad`` CLI entry point.

Argparse dispatcher for the driver subcommands. Currently only
``primal`` is wired; M4/M5 add ``objective``, ``gradient``, ``hessian``,
and ``calibrate``. The registry lazy-loads concrete Model modules on
first resolution, so no side-effect imports are needed here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cmad.cli.primal import run_primal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cmad")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    primal = sub.add_parser("primal", help="Run a forward (primal) solve.")
    primal.add_argument("deck", type=Path, help="Path to the YAML deck.")

    args = parser.parse_args(argv)

    if args.subcommand == "primal":
        return run_primal(args.deck)
    return 2


if __name__ == "__main__":
    sys.exit(main())
