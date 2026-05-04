"""Sympy-backed scalar expression parser for deck values.

Used by the FE deck builder to turn string-form DBC / SFB / body-force
values (``"0.01 * t"``, ``"sin(pi*x) * cos(pi*y)"``, etc.) into
JAX-traceable callables. Each parsed expression is compiled once at
parse time via ``sympy.parse_expr`` + ``sympy.lambdify(modules="jax")``;
the returned callable accepts the named variables as keyword
arguments and runs the JAX-native lambdified body per call.

Numeric literals (Python ``int`` / ``float``) short-circuit to a
constant callable that bypasses sympy entirely. Booleans are
rejected (a bool deck value is almost certainly a typo).

Available namespace:

- numpy / jax math functions (``sin``, ``cos``, ``exp``, ``sqrt``,
  etc.) recognized by sympy at parse time and routed to ``jax.numpy``
  by lambdify.
- ``pi``, ``E`` via sympy's standard atoms.
- caller-provided variable names via ``local_dict``.

Names referenced in the expression but not in ``names`` are caught
at parse time via the ``free_symbols`` check — no Python ``eval`` is
ever invoked, since sympy's ``parse_expr`` uses an AST parser rather
than ``eval()``.

Returned callables are JAX-traceable: they work both inside ``vmap``
(NBC + body force) and at the Python boundary (DBC, where
``np.asarray()`` coerces JAX scalars to NumPy at the materialization
site). Vector-shape composition (DBC ``(N_set, len(dofs))``, NBC
``(N_side_ips, num_components)``, body-force ``(num_eqs,)``) is
built inline in FE-builder helpers via ``jnp.stack`` of per-component
scalar callables — this module parses one scalar slot at a time.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import sympy
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import parse_expr, standard_transformations


class ExpressionError(ValueError):
    """Raised when a deck expression cannot be parsed or evaluated."""


def _build_safe_global_dict() -> dict[str, Any]:
    """Sympy-namespace dict with Python ``__builtins__`` masked.

    parse_expr ultimately routes through ``eval()``, so without an
    explicit empty ``__builtins__`` Python auto-inserts real builtins
    and resolves names like ``__import__`` / ``open`` / ``getattr`` to
    callables at parse time. Masking forces those names to be treated
    as undefined, so the ``AppliedUndef`` rejection below catches them.
    """
    sympy_ns: dict[str, Any] = {}
    exec("from sympy import *", sympy_ns)
    sympy_ns["__builtins__"] = {}
    return sympy_ns


_SAFE_GLOBAL_DICT = _build_safe_global_dict()


def parse_scalar_expression(
        expr: str | int | float,
        names: tuple[str, ...],
) -> Callable[..., Any]:
    """Parse a scalar deck expression (string or numeric literal).

    Returns a callable ``f(**kwargs)`` that accepts the variable
    names listed in ``names`` as keyword arguments and evaluates to
    a JAX-traceable scalar. Numeric literals short-circuit to a
    constant callable that ignores its kwargs.

    Parameters
    ----------
    expr
        Either a numeric literal (``int`` / ``float``, which produces
        a constant callable) or a string expression compiled by
        sympy. Booleans are rejected.
    names
        Variable names that the returned callable expects as keyword
        arguments. Names referenced in ``expr`` but not listed here
        are rejected at parse time via a ``free_symbols`` check.
        Names listed here but unused in ``expr`` are harmless.

    Raises
    ------
    ExpressionError
        On parse failure, lambdify failure, missing kwarg at call
        time, or non-finite numeric literal.
    """
    if isinstance(expr, bool):
        raise ExpressionError(
            f"boolean is not a valid scalar expression: {expr!r}",
        )

    if isinstance(expr, (int, float)):
        constant = float(expr)
        if not math.isfinite(constant):
            raise ExpressionError(
                f"non-finite numeric literal: {expr!r}",
            )

        def _const_call(**_: Any) -> float:
            return constant

        return _const_call

    if not isinstance(expr, str):
        raise ExpressionError(
            f"expected string or numeric literal, got "
            f"{type(expr).__name__}: {expr!r}",
        )

    syms = tuple(sympy.Symbol(name) for name in names)
    local_dict = dict(zip(names, syms, strict=True))

    try:
        parsed = parse_expr(
            expr,
            local_dict=local_dict,
            global_dict=_SAFE_GLOBAL_DICT,
            transformations=standard_transformations,
        )
    except Exception as e:
        raise ExpressionError(
            f"failed to parse expression {expr!r}: {e}",
        ) from e

    free = parsed.free_symbols - set(syms)
    if free:
        unknown = sorted(str(s) for s in free)
        raise ExpressionError(
            f"expression {expr!r} references unknown name(s): {unknown}",
        )

    undef_funcs = parsed.atoms(AppliedUndef)
    if undef_funcs:
        unknown_fns = sorted({type(f).__name__ for f in undef_funcs})
        raise ExpressionError(
            f"expression {expr!r} calls unknown function(s): "
            f"{unknown_fns}",
        )

    try:
        fn = sympy.lambdify(syms, parsed, modules="jax")
    except Exception as e:
        raise ExpressionError(
            f"failed to compile expression {expr!r} for jax: {e}",
        ) from e

    def _call(**kwargs: Any) -> Any:
        for name in names:
            if name not in kwargs:
                raise ExpressionError(
                    f"missing required variable '{name}' for "
                    f"expression {expr!r}",
                )
        args = tuple(kwargs[name] for name in names)
        return fn(*args)

    return _call
