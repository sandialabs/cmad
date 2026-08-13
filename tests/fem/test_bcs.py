"""Unit tests for `cmad.fem.bcs`.

The dataclass surface is small (no resolution, no mesh handle); these
tests cover construction round-trip and the `__post_init__` validation
paths for both BC types, plus the nodal-field value callable. Resolution
behavior (sideset walk, intra-BC dedup, broadcasting None / Sequence /
Callable values to flat arrays) is exercised in `test_dof.py` (DBC) and
`test_neumann.py` (NBC).
"""
import unittest

import jax.numpy as jnp
import numpy as np
from jax import grad, jit

from cmad.fem.bcs import DirichletBC, NeumannBC, make_nodal_field_values


class TestDirichletBC(unittest.TestCase):

    def test_dataclass_round_trip(self):
        bc = DirichletBC(
            sideset_names=["xmin_sides"],
            field_name="u",
            dofs=[0, 1, 2],
            values=[0.0, 0.5, 1.0],
        )
        self.assertEqual(list(bc.sideset_names), ["xmin_sides"])
        self.assertEqual(bc.field_name, "u")
        self.assertEqual(list(bc.dofs), [0, 1, 2])
        self.assertEqual(list(bc.values), [0.0, 0.5, 1.0])

    def test_multi_sideset_round_trip(self):
        bc = DirichletBC(
            sideset_names=["xmin_sides", "xmax_sides", "ymin_sides"],
            field_name="u",
            dofs=[0],
        )
        self.assertEqual(
            list(bc.sideset_names),
            ["xmin_sides", "xmax_sides", "ymin_sides"],
        )

    def test_default_homogeneous(self):
        bc = DirichletBC(
            sideset_names=["zmin_sides"], field_name="u", dofs=[2],
        )
        self.assertIsNone(bc.values)

    def test_callable_values_accepted(self):
        def vals(coords, t):
            return np.zeros((coords.shape[0], 1))
        bc = DirichletBC(
            sideset_names=["xmin_sides"],
            field_name="u",
            dofs=[0],
            values=vals,
        )
        self.assertTrue(callable(bc.values))


class TestPostInitValidation(unittest.TestCase):

    def test_empty_sideset_names_raises(self):
        with self.assertRaisesRegex(ValueError, "sideset_names .* non-empty"):
            DirichletBC(sideset_names=[], field_name="u", dofs=[0])

    def test_empty_dofs_raises(self):
        with self.assertRaisesRegex(ValueError, "dofs .* non-empty"):
            DirichletBC(
                sideset_names=["xmin_sides"], field_name="u", dofs=[],
            )

    def test_sequence_length_mismatch_raises(self):
        with self.assertRaisesRegex(
            ValueError, "values length .* does not match dofs length"
        ):
            DirichletBC(
                sideset_names=["xmin_sides"],
                field_name="u",
                dofs=[0, 1],
                values=[0.0],
            )


class TestNeumannBC(unittest.TestCase):

    def test_dataclass_round_trip(self):
        bc = NeumannBC(
            sideset_names=["xmax_sides"],
            field_name="u",
            values=[0.0, 0.0, -1.0],
        )
        self.assertEqual(list(bc.sideset_names), ["xmax_sides"])
        self.assertEqual(bc.field_name, "u")
        self.assertEqual(list(bc.values), [0.0, 0.0, -1.0])

    def test_multi_sideset_round_trip(self):
        bc = NeumannBC(
            sideset_names=["xmax_sides", "ymax_sides"],
            field_name="u",
            values=[1.0, 2.0, 3.0],
        )
        self.assertEqual(
            list(bc.sideset_names), ["xmax_sides", "ymax_sides"],
        )

    def test_callable_values_accepted(self):
        def vals(coords, t):
            return np.zeros((coords.shape[0], 3))
        bc = NeumannBC(
            sideset_names=["xmax_sides"], field_name="u", values=vals,
        )
        self.assertTrue(callable(bc.values))


class TestNeumannBCPostInitValidation(unittest.TestCase):

    def test_empty_sideset_names_raises(self):
        with self.assertRaisesRegex(ValueError, "sideset_names .* non-empty"):
            NeumannBC(sideset_names=[], field_name="u", values=[1.0])

    def test_empty_values_raises(self):
        with self.assertRaisesRegex(ValueError, "values .* non-empty"):
            NeumannBC(
                sideset_names=["xmax_sides"], field_name="u", values=[],
            )


def _frames() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three frames over four vertices and two dofs, at uneven times.

    Component values differ per vertex and per dof so a transposed or
    scrambled axis cannot pass, and the times are unevenly spaced so an
    interpolation weight of one half is not accidentally right.
    """
    data = np.array(
        [
            [[k * (v + 1), 10.0 * k + v] for v in range(4)]
            for k in range(3)
        ],
        dtype=float,
    )
    times = np.array([0.0, 1.0, 3.0])
    coords = np.zeros((4, 3))
    return data, times, coords


class TestNodalFieldValues(unittest.TestCase):

    def test_reproduces_each_frame_exactly(self):
        data, times, coords = _frames()
        values = make_nodal_field_values(data, times)
        for k in range(data.shape[0]):
            np.testing.assert_allclose(
                np.asarray(values(coords, times[k])), data[k],
            )

    def test_interpolates_between_frames(self):
        data, times, coords = _frames()
        values = make_nodal_field_values(data, times)
        # t = 2 sits halfway between the frames at t = 1 and t = 3.
        np.testing.assert_allclose(
            np.asarray(values(coords, 2.0)), 0.5 * (data[1] + data[2]),
        )
        # t = 0.25 sits a quarter of the way from the frame at t = 0.
        np.testing.assert_allclose(
            np.asarray(values(coords, 0.25)),
            0.75 * data[0] + 0.25 * data[1],
        )

    def test_holds_outside_the_measured_range(self):
        data, times, coords = _frames()
        values = make_nodal_field_values(data, times)
        np.testing.assert_allclose(np.asarray(values(coords, -5.0)), data[0])
        np.testing.assert_allclose(np.asarray(values(coords, 99.0)), data[-1])

    def test_single_frame_holds_for_any_time(self):
        data, times, coords = _frames()
        values = make_nodal_field_values(data[:1], times[:1])
        for t in (-1.0, 0.0, 99.0):
            np.testing.assert_allclose(np.asarray(values(coords, t)), data[0])

    def test_ignores_the_coordinates_it_is_passed(self):
        data, times, coords = _frames()
        values = make_nodal_field_values(data, times)
        shuffled = coords[::-1] + 1.0
        np.testing.assert_allclose(
            np.asarray(values(coords, 0.4)),
            np.asarray(values(shuffled, 0.4)),
        )

    def test_traceable_under_jit(self):
        data, times, coords = _frames()
        values = make_nodal_field_values(data, times)
        compiled = jit(lambda t: values(coords, t))
        for t in (0.0, 0.4, 2.0, 3.0):
            np.testing.assert_allclose(
                np.asarray(compiled(t)), np.asarray(values(coords, t)),
            )

    def test_time_derivative_matches_the_frame_slope(self):
        data, times, coords = _frames()
        values = make_nodal_field_values(data, times)
        slope = grad(lambda t: jnp.sum(values(coords, t)[:, 0]))(2.0)
        expected = np.sum(data[2][:, 0] - data[1][:, 0]) / (times[2] - times[1])
        self.assertAlmostEqual(float(slope), float(expected))

    def test_non_increasing_times_raises(self):
        data, _times, _coords = _frames()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            make_nodal_field_values(data, np.array([0.0, 0.0, 1.0]))

    def test_wrong_rank_raises(self):
        data, times, _coords = _frames()
        with self.assertRaisesRegex(ValueError, r"num_frames, N_set, num_dofs"):
            make_nodal_field_values(data[:, :, 0], times)

    def test_frame_count_mismatch_raises(self):
        data, times, _coords = _frames()
        with self.assertRaisesRegex(ValueError, "3 frames but data_times has 2"):
            make_nodal_field_values(data, times[:2])


if __name__ == "__main__":
    unittest.main()
