"""Tests for the sympy-backed scalar expression parser."""

import math
import unittest

import jax.numpy as jnp
from jax import jit

from cmad.io.expressions import ExpressionError, parse_scalar_expression


class TestNumericLiterals(unittest.TestCase):
    def test_int_literal(self) -> None:
        f = parse_scalar_expression(3, ())
        self.assertEqual(f(), 3.0)
        self.assertIsInstance(f(), float)

    def test_float_literal(self) -> None:
        f = parse_scalar_expression(0.25, ())
        self.assertEqual(f(), 0.25)

    def test_scientific_notation_literal(self) -> None:
        f = parse_scalar_expression(1.0e-3, ())
        self.assertEqual(f(), 1.0e-3)

    def test_negative_literal(self) -> None:
        f = parse_scalar_expression(-9.81, ())
        self.assertEqual(f(), -9.81)

    def test_constant_callable_ignores_kwargs(self) -> None:
        # Constant short-circuit accepts (and ignores) kwargs so the
        # caller can use a uniform call-site shape across constant /
        # string forms.
        f = parse_scalar_expression(1.5, ())
        self.assertEqual(f(x=42.0, t=99.0), 1.5)

    def test_non_finite_literal_rejected(self) -> None:
        for bad in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ExpressionError):
                parse_scalar_expression(bad, ())

    def test_bool_rejected(self) -> None:
        with self.assertRaises(ExpressionError):
            parse_scalar_expression(True, ())  # type: ignore[arg-type]
        with self.assertRaises(ExpressionError):
            parse_scalar_expression(False, ())  # type: ignore[arg-type]

    def test_unsupported_type_rejected(self) -> None:
        with self.assertRaises(ExpressionError):
            parse_scalar_expression(None, ())  # type: ignore[arg-type]
        with self.assertRaises(ExpressionError):
            parse_scalar_expression([1.0], ())  # type: ignore[arg-type]


class TestStringExpressions(unittest.TestCase):
    def test_simple_arithmetic(self) -> None:
        f = parse_scalar_expression("1 + 2 * 3", ())
        self.assertEqual(float(f()), 7.0)

    def test_time_ramp(self) -> None:
        f = parse_scalar_expression("0.01 * t", ("t",))
        self.assertAlmostEqual(float(f(t=0.0)), 0.0)
        self.assertAlmostEqual(float(f(t=2.0)), 0.02)

    def test_spatial_only(self) -> None:
        f = parse_scalar_expression("sin(pi * x)", ("x",))
        self.assertAlmostEqual(float(f(x=0.5)), 1.0)
        self.assertAlmostEqual(float(f(x=0.0)), 0.0)

    def test_spatial_and_temporal(self) -> None:
        f = parse_scalar_expression(
            "sin(pi*x) * cos(pi*y) * t",
            ("x", "y", "z", "t"),
        )
        self.assertAlmostEqual(float(f(x=0.5, y=0.0, z=0.0, t=1.0)), 1.0)
        self.assertAlmostEqual(float(f(x=0.0, y=0.0, z=0.0, t=1.0)), 0.0)
        self.assertAlmostEqual(float(f(x=0.5, y=0.0, z=0.0, t=0.0)), 0.0)

    def test_unused_name_is_harmless(self) -> None:
        # Names listed but absent from expr are ignored.
        f = parse_scalar_expression("t", ("x", "y", "z", "t"))
        self.assertEqual(float(f(x=99.0, y=99.0, z=99.0, t=0.5)), 0.5)

    def test_numpy_math_exp(self) -> None:
        f = parse_scalar_expression("exp(-t)", ("t",))
        self.assertAlmostEqual(float(f(t=0.0)), 1.0)
        self.assertAlmostEqual(float(f(t=1.0)), math.exp(-1.0))

    def test_numpy_math_sqrt(self) -> None:
        f = parse_scalar_expression("sqrt(x**2 + y**2)", ("x", "y"))
        self.assertAlmostEqual(float(f(x=3.0, y=4.0)), 5.0)

    def test_pi_constant_available(self) -> None:
        f = parse_scalar_expression("pi", ())
        self.assertAlmostEqual(float(f()), math.pi)

    def test_E_constant_available(self) -> None:
        # sympy uses uppercase E for Euler's number.
        f = parse_scalar_expression("E", ())
        self.assertAlmostEqual(float(f()), math.e)


class TestRejection(unittest.TestCase):
    def test_unknown_name_rejected(self) -> None:
        # Sympy creates an auto-Symbol for unknown names; the
        # free_symbols check rejects at parse time.
        with self.assertRaises(ExpressionError) as ctx:
            parse_scalar_expression("foo + 1", ("x",))
        self.assertIn("foo", str(ctx.exception))

    def test_syntax_error_rejected(self) -> None:
        with self.assertRaises(ExpressionError) as ctx:
            parse_scalar_expression("1 + 2 +", ())
        self.assertIn("parse", str(ctx.exception))

    def test_unknown_function_rejected(self) -> None:
        # Calls to unknown function names are auto-created as
        # AppliedUndef instances and rejected at parse time.
        with self.assertRaises(ExpressionError) as ctx:
            parse_scalar_expression("unknown_fn(x)", ("x",))
        self.assertIn("unknown_fn", str(ctx.exception))

    def test_dunder_import_rejected(self) -> None:
        # __builtins__ are masked in the parse_expr global_dict, so
        # __import__ does not resolve to the Python builtin and gets
        # auto-created as an unknown function. The "os" arg also
        # becomes an unknown free Symbol.
        with self.assertRaises(ExpressionError):
            parse_scalar_expression('__import__("os")', ())

    def test_getattr_rejected(self) -> None:
        with self.assertRaises(ExpressionError):
            parse_scalar_expression('getattr(x, "__class__")', ("x",))

    def test_open_rejected(self) -> None:
        with self.assertRaises(ExpressionError):
            parse_scalar_expression('open("/etc/passwd")', ())

    def test_missing_kwarg_rejected(self) -> None:
        f = parse_scalar_expression("x + t", ("x", "t"))
        with self.assertRaises(ExpressionError) as ctx:
            f(x=1.0)
        self.assertIn("t", str(ctx.exception))


class TestJitTraceability(unittest.TestCase):
    def test_jit_compiles_string_expression(self) -> None:
        # The whole point of using sympy.lambdify(modules="jax") is
        # so that parsed callables work inside jit (NBC + body
        # force sites are vmap-traced).
        f = parse_scalar_expression("sin(pi*x) * t", ("x", "t"))
        jitted = jit(lambda x, t: f(x=x, t=t))
        self.assertAlmostEqual(float(jitted(0.5, 1.0)), 1.0)
        self.assertAlmostEqual(float(jitted(0.0, 1.0)), 0.0)

    def test_jit_compiles_constant(self) -> None:
        # Numeric short-circuit: the const callable must also be
        # jit-traceable (the BC builders use a uniform calling
        # convention across const / string forms).
        f = parse_scalar_expression(0.5, ("x", "t"))
        jitted = jit(lambda x, t: f(x=x, t=t))
        self.assertEqual(float(jitted(jnp.array(1.0), jnp.array(2.0))), 0.5)


class TestCompileOnce(unittest.TestCase):
    def test_repeated_calls_consistent(self) -> None:
        # Compile-once-call-many: repeated calls on the same parser
        # closure produce results consistent with re-parsing.
        f = parse_scalar_expression("a * b + 1", ("a", "b"))
        for a in (-1.0, 0.0, 0.5, 7.0):
            for b in (0.0, 0.25, 3.0):
                self.assertAlmostEqual(
                    float(f(a=a, b=b)), a * b + 1.0,
                )

    def test_independent_parsers_dont_share_state(self) -> None:
        # Two parsers built from the same expression should be
        # independent (no cross-call symtable leak).
        f1 = parse_scalar_expression("k * x", ("k", "x"))
        f2 = parse_scalar_expression("k * x", ("k", "x"))
        self.assertEqual(float(f1(k=2.0, x=3.0)), 6.0)
        self.assertEqual(float(f2(k=10.0, x=10.0)), 100.0)
        # f1 should still work after f2 was used
        self.assertEqual(float(f1(k=2.0, x=4.0)), 8.0)


if __name__ == "__main__":
    unittest.main()
