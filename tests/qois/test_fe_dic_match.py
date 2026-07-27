"""Tests for ``FEDicMatch``: GMLS remap of a DIC cloud onto a sideset.

The DIC field is linear, which GMLS reproduces exactly, so the remapped
nodal target equals the field sampled directly at the sideset nodes.
That lets the squared-mismatch objective be checked against the already
verified surface mode of ``FEDisplacementMatch`` (the independent
reference) rather than against the same code path.
"""
import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.io.point_cloud import PointCloud, write_point_cloud
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.parameters.parameters import Parameters
from cmad.qois.fe_dic_match import FEDicMatch
from cmad.qois.fe_displacement_match import FEDisplacementMatch


def _elastic_parameters(kappa: float = 100.0, mu: float = 50.0) -> Parameters:
    values = {"elastic": {"kappa": kappa, "mu": mu}}
    active = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Parameters(values, active, transforms)


def _unit_cube_problem(n: int = 2):
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (n, n, n))
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    dof_map = build_dof_map(mesh, [layout], [], components_by_field={"u": 3})
    gr = SmallDispEquilibrium(ndims=3)
    model = Elastic(_elastic_parameters(), def_type=DefType.FULL_3D)
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.CLOSED_FORM},
    )


def _linear_field(coords: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """``u_i = c0 + cx x + cy y + cz z`` with ``coeffs`` shaped ``(3, 4)``."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    basis = np.stack([np.ones_like(x), x, y, z], axis=1)
    return basis @ coeffs.T


def _dic_cloud(coeffs: np.ndarray, n_points: int = 60, n_steps: int = 2):
    """Scattered points on the x=1 face carrying the linear field ``coeffs``."""
    rng = np.random.default_rng(0)
    yz = rng.uniform(0.0, 1.0, (n_points, 2))
    coords = np.column_stack([np.ones(n_points), yz])
    disp = _linear_field(coords, coeffs)
    frames = np.stack([disp] * n_steps, axis=0)
    return PointCloud(
        coords=coords,
        times=np.arange(n_steps, dtype=np.float64),
        fields={"displacement": frames},
    )


_F = np.array([
    [0.0, 0.01, 0.02, -0.01],
    [0.0, 0.0, 0.03, 0.01],
    [0.0, -0.02, 0.0, 0.015],
])
_G = np.array([
    [0.0, 0.005, 0.0, 0.02],
    [0.0, 0.01, -0.01, 0.0],
    [0.0, 0.0, 0.02, -0.005],
])


class TestFEDicMatch(unittest.TestCase):
    def setUp(self) -> None:
        self.fe = _unit_cube_problem(2)
        self.params = params_by_block_from_models(self.fe)
        self.n_dofs = self.fe.dof_map.num_total_dofs
        self.t = [0.0, 1.0]

    def _step(self, qoi, U) -> float:
        closure = qoi.step_contribution(self.params, self.fe.kernel_arrays)
        return float(closure(
            jnp.asarray(U), jnp.zeros(self.n_dofs), {}, {},
            jnp.asarray(1.0), jnp.asarray(0.0),
        ))

    def _dic_J(self, cloud, U) -> float:
        return self._step(
            FEDicMatch(self.fe, self.t, cloud, "xmax_sides"), U,
        )

    def _reference_J(self, node_field, U) -> float:
        data = np.broadcast_to(node_field.ravel(), (2, self.n_dofs))
        qoi = FEDisplacementMatch(
            self.fe, self.t, jnp.asarray(data), sideset="xmax_sides",
        )
        return self._step(qoi, U)

    def test_exact_field_gives_zero(self) -> None:
        cloud = _dic_cloud(_F)
        U = _linear_field(self.fe.mesh.nodes, _F).ravel()
        self.assertAlmostEqual(self._dic_J(cloud, U), 0.0, places=12)

    def test_remap_matches_direct_nodal_target(self) -> None:
        cloud = _dic_cloud(_G)
        U = _linear_field(self.fe.mesh.nodes, _F).ravel()
        g_nodes = _linear_field(self.fe.mesh.nodes, _G)
        j_dic = self._dic_J(cloud, U)
        j_ref = self._reference_J(g_nodes, U)
        self.assertGreater(j_ref, 0.0)
        self.assertAlmostEqual(j_dic, j_ref, places=10)

    def test_remap_converges_with_cloud_density(self) -> None:
        # A non-polynomial DIC field is reconstructed only approximately, so
        # the surface mismatch against the true nodal field shrinks as the
        # cloud densifies -- unlike the exact reproduction of a linear field.
        # U_FE is the field at the nodes, so J is a surface norm of the nodal
        # remap error and goes to zero as the cloud refines.
        def field(coords: np.ndarray) -> np.ndarray:
            y, z = coords[:, 1], coords[:, 2]
            return np.stack(
                [np.sin(2.0 * y) * np.sin(2.0 * z), np.cos(2.0 * y),
                 np.sin(2.0 * z)],
                axis=1,
            )

        U = field(self.fe.mesh.nodes).ravel()
        errors = []
        for n_points in (50, 200, 800):
            rng = np.random.default_rng(1)
            yz = rng.uniform(0.0, 1.0, (n_points, 2))
            coords = np.column_stack([np.ones(n_points), yz])
            disp = field(coords)
            cloud = PointCloud(
                coords=coords, times=np.array([0.0, 1.0]),
                fields={"displacement": np.stack([disp, disp], axis=0)},
            )
            errors.append(self._dic_J(cloud, U))

        # each 4x densification cuts J by well over an order of magnitude
        # (observed ~80x then ~160x), the signature of a converging remap
        self.assertGreater(errors[0], 10.0 * errors[1])
        self.assertGreater(errors[1], 10.0 * errors[2])

    def test_from_deck_matches_direct(self) -> None:
        cloud = _dic_cloud(_G)
        U = _linear_field(self.fe.mesh.nodes, _F).ravel()
        j_direct = self._dic_J(cloud, U)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "dic.h5"
            write_point_cloud(path, cloud, write_xdmf=False)
            section = {
                "name": "fe_dic_match",
                "dic_file": str(path),
                "sideset": "xmax_sides",
            }
            qoi = FEDicMatch.from_deck(section, self.fe, self.t)
            j_deck = self._step(qoi, U)
        self.assertAlmostEqual(j_deck, j_direct, places=12)

    def test_frame_count_mismatch_raises(self) -> None:
        cloud = _dic_cloud(_F, n_steps=3)
        with self.assertRaises(ValueError):
            FEDicMatch(self.fe, self.t, cloud, "xmax_sides")


if __name__ == "__main__":
    unittest.main()
