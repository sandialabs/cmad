"""Tests for the ``print_global_convergence`` /
``print_local_convergence`` toggles on the FE driver and per-IP
local Newton.

Captures stdout via :func:`contextlib.redirect_stdout` and asserts
the expected format substrings appear when each flag is set. Off-
path silence is implicitly covered by every other FE test (all run
with the flags defaulted to False).
"""
import unittest
from contextlib import redirect_stdout
from io import StringIO

from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.driver import fe_quasistatic_drive
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.model import Model
from tests.fem.test_fe_quasistatic_drive import (
    _make_elastic_model,
    _make_J2_model,
    _uniaxial_dbcs,
)


def _build_problem(
        model: Model,
        mode: GlobalResidualMode,
        print_local_convergence: bool = False,
) -> FEProblem:
    mesh = StructuredHexMesh(
        lengths=(1.0, 1.0, 1.0), divisions=(1, 1, 1),
    )
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    dof_map = build_dof_map(
        mesh, [layout], _uniaxial_dbcs(slope=1e-3),
        components_by_field={"u": 3},
    )
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=SmallDispEquilibrium(ndims=3),
        models_by_block={"all": model},
        modes_by_block={"all": mode},
        print_local_convergence=print_local_convergence,
    )


class TestPrintGlobalConvergence(unittest.TestCase):
    """``print_global_convergence=True`` on
    :func:`fe_quasistatic_drive` emits the per-step header and
    per-Newton-iter norm lines via :func:`jax.debug.print`."""

    def test_emits_step_header_and_iter_norms(self) -> None:
        fe_problem = _build_problem(
            _make_elastic_model(), GlobalResidualMode.CLOSED_FORM,
        )
        buf = StringIO()
        with redirect_stdout(buf):
            fe_quasistatic_drive(
                fe_problem, [0.0, 1.0],
                print_global_convergence=True,
            )
        output = buf.getvalue()
        self.assertIn("ON PRIMAL STEP (1) at t=", output)
        self.assertIn("Newton iteration", output)
        self.assertIn("absolute ||R||", output)
        self.assertIn("relative ||R||", output)


class TestPrintLocalConvergence(unittest.TestCase):
    """``build_fe_problem(..., print_local_convergence=True)`` bakes
    a per-(elem, ip) per-iter convergence print into the COUPLED
    block's local-Newton evaluator; lines surface during any
    subsequent forward solve."""

    def test_emits_local_iter_norms(self) -> None:
        fe_problem = _build_problem(
            _make_J2_model(), GlobalResidualMode.COUPLED,
            print_local_convergence=True,
        )
        buf = StringIO()
        with redirect_stdout(buf):
            fe_quasistatic_drive(fe_problem, [0.0, 1.0])
        output = buf.getvalue()
        self.assertIn("[LOCAL elem=", output)
        self.assertIn("ip=", output)
        self.assertIn("abs ||C||", output)
        self.assertIn("rel ||C||", output)


if __name__ == "__main__":
    unittest.main()
