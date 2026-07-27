"""Finite mixed (u-p) uniaxial check for BeBarElasticPlastic.

Drives the finite-deformation mixed Mechanics residual in uniaxial
tension and compares the axial Cauchy stress to the continuous finite J2
reference, exercising the finite mixed path (cof(F) momentum + finite
stabilization) with a path-dependent model.
"""
import unittest
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.fe_problem import FEState
from cmad.fem.nonlinear_solver import fe_newton_solve
from cmad.fem.postprocess import evaluate_cauchy_at_ips
from cmad.models.be_bar_elastic_plastic import BeBarElasticPlastic
from cmad.models.deformation_types import DefType
from cmad.typing import JaxArray
from tests.fem.test_mixed_up_plastic import _build_mixed_fe
from tests.support.test_problems import (
    J2AnalyticalProblem,
    finite_uniaxial_j2_voce,
)

_E, _NU, _Y, _S, _D = 200e3, 0.3, 200.0, 200.0, 20.0
_ALPHA = 0.15
_NUM_STEPS = 25


class TestMixedUpFinitePlastic(unittest.TestCase):

    def test_uniaxial_axial_cauchy(self) -> None:
        lambda_axial, _, cauchy_ref = finite_uniaxial_j2_voce(
            _E, _NU, _Y, _S, _D, _ALPHA,
        )

        problem = J2AnalyticalProblem()
        model = BeBarElasticPlastic(
            problem.J2_parameters, def_type=DefType.FULL_3D,
        )
        fe_problem = _build_mixed_fe(model)
        params = params_by_block_from_models(fe_problem)

        state = FEState.from_problem(fe_problem)
        U_solved: NDArray[np.floating] | JaxArray = state.U_at(0)
        xi_prev: Mapping[str, NDArray[np.floating] | JaxArray] = {
            "all": state.xi_at(0, "all"),
        }
        for step in range(1, _NUM_STEPS + 1):
            t = (lambda_axial - 1.0) * step / _NUM_STEPS
            U_solved, xi_solved = fe_newton_solve(
                fe_problem, params, U_prev=U_solved, t=t,
                xi_prev_by_block=xi_prev,
            )
            state.append(U_solved, xi_solved, t)
            xi_prev = xi_solved

        cauchy = evaluate_cauchy_at_ips(
            fe_problem, state, _NUM_STEPS, "all",
        )
        np.testing.assert_allclose(cauchy[..., 0], cauchy_ref, rtol=2e-3)
        self.assertLess(
            float(np.max(np.abs(cauchy[..., 3]))), 5e-3 * cauchy_ref,
        )
        self.assertLess(
            float(np.max(np.abs(cauchy[..., 5]))), 5e-3 * cauchy_ref,
        )
        p = np.asarray(U_solved)[fe_problem.dof_map.block_offsets[1]:]
        np.testing.assert_allclose(p, -cauchy_ref / 3.0, rtol=5e-3)


if __name__ == "__main__":
    unittest.main()
