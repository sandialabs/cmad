"""Output-API override tests for :class:`SmallDispEquilibrium`.

Locks the default-output catalog and the dispatch-by-name evaluators
in isolation. End-to-end cauchy evaluation through
``evaluate_element_field`` needs a full :class:`FEProblem` fixture
and is covered by the FE primal round-trip test; the underlying
``evaluate_cauchy_at_ips`` helper itself has its own coverage in
``tests/fem/test_postprocess.py``.
"""
import unittest

import numpy as np

from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType


class _StubState:
    """Minimal :class:`FEState` shim — implements only ``U_at``."""
    def __init__(self, U: np.ndarray) -> None:
        self._U = U

    def U_at(self, step: int) -> np.ndarray:
        return self._U


class TestSmallDispEquilibriumOutputAPI(unittest.TestCase):
    def test_default_output_fields_declares_displacement_and_cauchy(self):
        gr = SmallDispEquilibrium(ndims=3)
        self.assertEqual(
            gr.default_output_fields(),
            {
                "nodal": [FieldSpec("displacement", VarType.VECTOR)],
                "element": [FieldSpec("cauchy", VarType.SYM_TENSOR)],
            },
        )

    def test_evaluate_nodal_field_displacement_reshapes_U(self):
        gr = SmallDispEquilibrium(ndims=3)
        U_flat = np.arange(12.0)
        stub_state = _StubState(U_flat)
        result = gr.evaluate_nodal_field(
            "displacement", None, stub_state, 0,  # type: ignore[arg-type]
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

    def test_unknown_element_name_raises_via_super_with_subclass_name(self):
        gr = SmallDispEquilibrium(ndims=3)
        with self.assertRaises(ValueError) as ctx:
            gr.evaluate_element_field(
                "garbage", None, None, 0, "blk",  # type: ignore[arg-type]
            )
        msg = str(ctx.exception)
        self.assertIn("SmallDispEquilibrium", msg)
        self.assertIn("garbage", msg)


if __name__ == "__main__":
    unittest.main()
