"""``assemble_global_residual`` returns the same ``R`` as ``assemble_global``
for a COUPLED block, whose per-IP local Newton runs through the ``"R"``
evaluator rather than the tangent evaluator the solve uses.
"""
import unittest
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from numpy.typing import NDArray

from cmad.fem.assembly import (
    assemble_global,
    assemble_global_residual,
    params_by_block_from_models,
)
from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import Mesh, StructuredHexMesh
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.model import Model
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray, PyTreeDict, Scalar

_KAPPA = 100.0
_MU = 50.0
_BODY_FORCE = (3.0, -2.0, 5.0)


def _make_elastic_model() -> Elastic:
    values = cast(PyTreeDict, {"elastic": {"kappa": _KAPPA, "mu": _MU}})
    active = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Elastic(
        Parameters(values, active, transforms),
        def_type=DefType.FULL_3D,
    )


def _body_force(
        coords_ip: NDArray[np.floating] | JaxArray, t: Scalar,
) -> JaxArray:
    del coords_ip, t  # constant body force
    return jnp.asarray(_BODY_FORCE, dtype=jnp.float64)


def _build_fe_problem(
        mesh: Mesh,
        modes_by_block: dict[str, GlobalResidualMode],
        forcing: dict[int, Any],
) -> FEProblem:
    """SmallDispEquilibrium Elastic FULL_3D problem; homogeneous DBC on all
    six faces, with ``forcing`` keyed by residual-block index."""
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    bc = DirichletBC(
        sideset_names=[
            "xmin_sides", "xmax_sides",
            "ymin_sides", "ymax_sides",
            "zmin_sides", "zmax_sides",
        ],
        field_name="u",
        dofs=(0, 1, 2),
        values=None,
    )
    dof_map = build_dof_map(
        mesh, [layout], [bc], components_by_field={"u": 3},
    )
    gr = SmallDispEquilibrium(ndims=3)
    models_by_block: dict[str, Model] = {
        b: _make_elastic_model() for b in mesh.element_blocks
    }
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=gr,
        models_by_block=models_by_block,
        modes_by_block=modes_by_block,
        forcing_fns_by_block_idx=forcing,
    )


class TestCoupledResidualMatchesAssembleGlobal(unittest.TestCase):
    def test_matches(self) -> None:
        mesh = StructuredHexMesh(
            lengths=(1.0, 1.0, 1.0), divisions=(2, 2, 2),
        )
        # The body force makes the residual's f_ext term nonzero.
        fe = _build_fe_problem(
            mesh, {"all": GlobalResidualMode.COUPLED}, {0: _body_force},
        )
        n_dofs = fe.dof_map.num_total_dofs
        n_elems = mesh.connectivity.shape[0]
        rng = np.random.default_rng(seed=11)
        U = 1e-3 * rng.standard_normal(n_dofs)
        xi_prev_by_block: dict[str, NDArray[np.floating]] = {
            "all": np.zeros((n_elems, 8, 6)),
        }

        params = params_by_block_from_models(fe)
        _, R, _ = assemble_global(
            fe, fe.kernel_arrays, params, U, U, t=0.3,
            xi_prev_by_block=xi_prev_by_block,
        )
        R_only = assemble_global_residual(
            fe, fe.kernel_arrays, params, U, U, t=0.3,
            xi_prev_by_block=xi_prev_by_block,
        )
        np.testing.assert_allclose(
            np.asarray(R_only), np.asarray(R), atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
