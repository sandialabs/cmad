"""Contract tests for the :class:`GlobalResidual` base output API.

A test-local stub subclass that does not override
:meth:`default_output_fields` / :meth:`evaluate_nodal_field` /
:meth:`evaluate_element_field` exercises the base defaults — empty
catalog and ``ValueError`` on any field-name dispatch — so the
contract is regression-protected independently of any concrete GR.
"""
import unittest

import numpy as np

from cmad.global_residuals import GlobalResidual
from cmad.models.var_types import VarType


class _StubGR(GlobalResidual):
    """Minimal GR with no output overrides; residual_fn is never called."""
    def __init__(self) -> None:
        self._is_complex = False
        self.dtype = np.float64
        self._ndims = 3
        self._init_residuals(1)
        self._var_types[0] = VarType.VECTOR
        self._num_eqs[0] = 3
        self.resid_names[0] = "displacement"
        self.var_names[0] = "u"

        def residual_fn(xi, xi_prev, params, U, U_prev,
                        model, mode, shapes_ip, w, dv, ip_set):
            raise AssertionError(  # pragma: no cover - never reached
                "_StubGR.residual_fn must not be called in base-API tests"
            )

        super().__init__(residual_fn)


class TestGlobalResidualBaseOutputAPI(unittest.TestCase):
    def test_default_output_fields_returns_empty_buckets(self):
        gr = _StubGR()
        self.assertEqual(
            gr.default_output_fields(), {"nodal": [], "element": []},
        )

    def test_evaluate_nodal_field_raises_value_error_with_name_and_class(self):
        gr = _StubGR()
        with self.assertRaises(ValueError) as ctx:
            gr.evaluate_nodal_field("disp", None, None, 0)  # type: ignore[arg-type]
        msg = str(ctx.exception)
        self.assertIn("disp", msg)
        self.assertIn("_StubGR", msg)
        self.assertIn("nodal", msg)

    def test_evaluate_element_field_raises_value_error_with_name_and_class(self):
        gr = _StubGR()
        with self.assertRaises(ValueError) as ctx:
            gr.evaluate_element_field(
                "flux", None, None, 0, "blk",  # type: ignore[arg-type]
            )
        msg = str(ctx.exception)
        self.assertIn("flux", msg)
        self.assertIn("_StubGR", msg)
        self.assertIn("element", msg)


if __name__ == "__main__":
    unittest.main()
