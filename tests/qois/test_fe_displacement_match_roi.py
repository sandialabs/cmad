"""Region of interest option of ``FEDisplacementMatch``.

The mismatch is integrated and normalized over the elements a
full-field measurement covers, not over the whole domain. Two checks on
a unit square of four quads, against hand-built displacements rather
than FE solutions:

- a mismatch of 0.25 everywhere scores ``0.25**2`` whether or not the
  region is restricted to half the elements, because the integral and
  the area it is divided by shrink together. Normalizing by the whole
  area while integrating half would halve it.
- a mismatch on a node that only one element owns scores zero once the
  region excludes that element.

``load_roi`` is covered separately: the mesh dimension chooses whether
``elements`` or ``sides`` is read, so a file built for the other kind of
mesh raises rather than integrating over the wrong entities.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from jax.tree_util import tree_map

from cmad.fem.assembly import params_by_block_from_models
from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import build_fe_problem
from cmad.fem.finite_element import Q1_QUAD
from cmad.fem.mesh import StructuredQuadMesh
from cmad.global_residuals.mechanics import Mechanics
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.io.qoi_data import load_roi
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.global_fields import StepTime
from cmad.parameters.parameters import Parameters
from cmad.qois.fe_displacement_match import FEDisplacementMatch

_T_SCHEDULE = [0.0, 1.0]


def _problem():
    mesh = StructuredQuadMesh(lengths=(1.0, 1.0), divisions=(2, 2))
    dof_map = build_dof_map(
        mesh, [GlobalFieldLayout(name="u", finite_element=Q1_QUAD)], [],
        components_by_field={"u": 2},
    )
    values = {"elastic": {"kappa": 100.0, "mu": 50.0}}
    model = Elastic(
        Parameters(values, tree_map(lambda _: True, values),
                   tree_map(lambda _: None, values)),
        def_type=DefType.PLANE_STRAIN,
    )
    return mesh, build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=Mechanics(ndims=2),
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.CLOSED_FORM},
    )


def _evaluate(fe_problem, data, U, roi):
    qoi = FEDisplacementMatch(fe_problem, _T_SCHEDULE, data, roi=roi)
    closure = qoi.step_contribution(
        params_by_block_from_models(fe_problem), fe_problem.kernel_arrays)
    return float(closure(U, U, {}, {}, StepTime(1.0, 0.0)))


class TestRegionOfInterest(unittest.TestCase):

    def setUp(self) -> None:
        self.mesh, self.fe_problem = _problem()
        self.n_nodes = self.mesh.nodes.shape[0]
        self.U = np.zeros(self.n_nodes * 2)

    def _uniform_data(self, shift: float) -> np.ndarray:
        data = np.zeros((len(_T_SCHEDULE), self.n_nodes, 2))
        data[..., 0] = shift
        return data

    def test_normalization_uses_the_region(self) -> None:
        data = self._uniform_data(0.25)
        whole = _evaluate(self.fe_problem, data, self.U, None)
        half = _evaluate(
            self.fe_problem, data, self.U, np.array([0, 1], dtype=np.intp))
        self.assertAlmostEqual(whole, 0.25 ** 2, places=12)
        self.assertAlmostEqual(half, whole, places=12)

    def test_mismatch_outside_the_region_is_not_scored(self) -> None:
        # the far corner node belongs only to the far corner element, and
        # the region keeps the element diagonally opposite it
        far_node = int(np.argmin(
            np.linalg.norm(self.mesh.nodes - np.array([1.0, 1.0]), axis=1)))
        centroids = self.mesh.nodes[self.mesh.connectivity].mean(axis=1)
        near_elem = int(np.argmin(
            np.linalg.norm(centroids - np.array([0.25, 0.25]), axis=1)))
        self.assertNotIn(far_node, self.mesh.connectivity[near_elem])

        data = np.zeros((len(_T_SCHEDULE), self.n_nodes, 2))
        data[:, far_node, 0] = 1.0
        self.assertGreater(_evaluate(self.fe_problem, data, self.U, None), 0.0)
        self.assertAlmostEqual(
            _evaluate(self.fe_problem, data, self.U,
                      np.array([near_elem], dtype=np.intp)),
            0.0, places=12,
        )

    def test_empty_region_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "selects no elements"):
            FEDisplacementMatch(
                self.fe_problem, _T_SCHEDULE, self._uniform_data(0.25),
                roi=np.array([], dtype=np.intp))

    def test_sideset_and_region_together_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "not both"):
            FEDisplacementMatch(
                self.fe_problem, _T_SCHEDULE, self._uniform_data(0.25),
                sideset="xmax_sides", roi=np.array([0], dtype=np.intp))


class TestLoadRoi(unittest.TestCase):

    def test_dimension_chooses_the_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roi.npz"
            np.savez(path, elements=np.array([1, 2, 3]),
                     sides=np.array([[0, 1], [2, 3]]))
            section = {"roi_file": str(path)}
            np.testing.assert_array_equal(
                load_roi(section, 2), np.array([1, 2, 3]))
            np.testing.assert_array_equal(
                load_roi(section, 3), np.array([[0, 1], [2, 3]]))

    def test_wrong_dimension_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roi.npz"
            np.savez(path, elements=np.array([1, 2, 3]))
            with self.assertRaisesRegex(ValueError, "no 'sides' entry"):
                load_roi({"roi_file": str(path)}, 3)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_roi({"roi_file": "does/not/exist.npz"}, 2)


if __name__ == "__main__":
    unittest.main()
