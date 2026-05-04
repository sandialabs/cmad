"""Registry round-trip and discovery tests for the GR registry."""

import unittest

from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.io.registry import (
    register_global_residual,
    registered_global_residuals,
    resolve_global_residual,
)


class TestGRRegistry(unittest.TestCase):
    def test_register_and_resolve_round_trip(self) -> None:
        cls = resolve_global_residual("small_disp_equilibrium")
        self.assertIs(cls, SmallDispEquilibrium)

    def test_resolve_unknown_raises_with_listing(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_global_residual("not_a_real_gr")
        msg = str(ctx.exception)
        self.assertIn("not_a_real_gr", msg)
        self.assertIn("not registered", msg)
        self.assertIn("small_disp_equilibrium", msg)

    def test_registered_lists_via_fragment_dir(self) -> None:
        names = registered_global_residuals()
        self.assertIn("small_disp_equilibrium", names)
        self.assertEqual(names, sorted(names))

    def test_double_register_raises(self) -> None:
        # SmallDispEquilibrium is already registered at module import;
        # re-registering under the same name should fail.
        with self.assertRaises(ValueError) as ctx:
            register_global_residual("small_disp_equilibrium")(
                SmallDispEquilibrium,
            )
        msg = str(ctx.exception)
        self.assertIn("small_disp_equilibrium", msg)
        self.assertIn("already registered", msg)

    def test_register_empty_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            register_global_residual("")
        with self.assertRaises(ValueError):
            register_global_residual("   ")


if __name__ == "__main__":
    unittest.main()
