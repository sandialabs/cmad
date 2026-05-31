"""Output-API override tests for :class:`SmallDispEquilibrium`.

Locks the primary (nodal) output catalog and the nodal dispatch-by-name
evaluator in isolation. The model-derived ``cauchy`` element field is
resolved off the GR now (see :mod:`cmad.fem.postprocess` and
:func:`cmad.io.writers.resolve_fe_output_plan`); its evaluation has
coverage in ``tests/fem/test_postprocess.py`` and the FE primal
round-trip.
"""
import unittest

import numpy as np

from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.var_types import VarType


class _StubState:
    """Minimal :class:`FEState` shim — implements only ``U_at``."""
    def __init__(self, U: np.ndarray) -> None:
        self._U = U

    def U_at(self, step: int) -> np.ndarray:
        return self._U


class TestSmallDispEquilibriumOutputAPI(unittest.TestCase):
    def test_primary_output_fields_declares_u(self):
        gr = SmallDispEquilibrium(ndims=3)
        self.assertEqual(
            gr.primary_output_fields(), [("u", VarType.VECTOR)],
        )

    def test_evaluate_nodal_field_u_reshapes_U(self):
        gr = SmallDispEquilibrium(ndims=3)
        U_flat = np.arange(12.0)
        stub_state = _StubState(U_flat)
        result = gr.evaluate_nodal_field(
            "u", None, stub_state, 0,  # type: ignore[arg-type]
        )
        self.assertEqual(result.shape, (4, 3))
        self.assertTrue(np.array_equal(result, U_flat.reshape(4, 3)))

    def test_unknown_nodal_name_raises_via_super_with_subclass_name(self):
        gr = SmallDispEquilibrium(ndims=3)
        with self.assertRaises(ValueError) as ctx:
            gr.evaluate_nodal_field(
                "garbage", None, None, 0,  # type: ignore[arg-type]
            )
        msg = str(ctx.exception)
        self.assertIn("SmallDispEquilibrium", msg)
        self.assertIn("garbage", msg)


if __name__ == "__main__":
    unittest.main()
