"""Material parameters written as functions of temperature.

The two forms are checked against numbers by hand, the walk leaves every
other subtree alone, the resolver reads the point's temperature or the
reference temperature, and the builder rejects an active flag inside a
form.
"""
import unittest

import numpy as np

from cmad.io.params_builder import build_parameters
from cmad.models.deformation_types import DefType
from cmad.models.elastic_constants import compute_mu
from cmad.models.global_fields import (
    GlobalFieldsAtPoint,
    mp_U_from_F,
    temperature_at_point,
)
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.temperature_dependent_parameters import (
    check_parameter_forms,
    evaluate_parameters,
    make_parameter_resolver,
)

_C0, _C1 = 2.0, 0.5
_KNOTS = np.array([300.0, 500.0])
_ENTRIES = np.array([10.0, 30.0])


def _forms():
    return {
        "poly": {"polynomial": {"coefficients": np.array([_C0, _C1])}},
        "tab": {"table": {"T": _KNOTS, "values": _ENTRIES}},
    }


def _plain_tree():
    return {
        "elastic": {"E": 200e3, "nu": 0.3},
        "plastic": {
            "effective stress": {"J2": {}},
            "flow stress": {
                "initial yield": {"Y": 200.0},
                "hardening": {"voce": {"S": 200.0, "D": 20.0}},
            },
        },
        "rotation matrix": np.eye(3),
    }


class TestForms(unittest.TestCase):

    def test_polynomial_and_table_by_hand(self):
        values = evaluate_parameters(_forms(), 300.0)
        self.assertAlmostEqual(float(values["poly"]), _C0 + _C1 * 300.0)
        values = evaluate_parameters(_forms(), 400.0)
        self.assertAlmostEqual(float(values["tab"]), 20.0)

    def test_table_is_held_outside_the_knots(self):
        below = evaluate_parameters(_forms(), 100.0)["tab"]
        above = evaluate_parameters(_forms(), 900.0)["tab"]
        self.assertEqual(float(below), _ENTRIES[0])
        self.assertEqual(float(above), _ENTRIES[-1])


class TestWalk(unittest.TestCase):

    def test_plain_tree_comes_back_equal(self):
        tree = _plain_tree()
        values = evaluate_parameters(tree, 350.0)
        self.assertEqual(values["elastic"], tree["elastic"])
        self.assertEqual(values["plastic"], tree["plastic"])
        np.testing.assert_array_equal(
            values["rotation matrix"], tree["rotation matrix"])

    def test_forms_inside_subtrees_are_replaced(self):
        tree = _plain_tree()
        tree["elastic"]["E"] = _forms()["poly"]
        tree["plastic"]["flow stress"]["initial yield"]["Y"] = _forms()["tab"]
        values = evaluate_parameters(tree, 300.0)
        self.assertAlmostEqual(float(values["elastic"]["E"]), _C0 + _C1 * 300.0)
        self.assertEqual(
            float(values["plastic"]["flow stress"]["initial yield"]["Y"]),
            _ENTRIES[0])
        self.assertEqual(values["plastic"]["effective stress"], {"J2": {}})


class TestResolver(unittest.TestCase):

    def test_identity_without_forms(self):
        tree = _plain_tree()
        resolve = make_parameter_resolver(tree, 293.15)
        U = mp_U_from_F(np.eye(3))
        self.assertIs(resolve(tree, U), tree)

    def test_reference_temperature_without_a_temperature_field(self):
        tree = _plain_tree()
        tree["elastic"]["E"] = _forms()["poly"]
        resolve = make_parameter_resolver(tree, 400.0)
        values = resolve(tree, mp_U_from_F(np.eye(3)))
        self.assertAlmostEqual(float(values["elastic"]["E"]), _C0 + _C1 * 400.0)

    def test_the_point_temperature_when_the_field_exists(self):
        tree = _plain_tree()
        tree["elastic"]["E"] = _forms()["poly"]
        resolve = make_parameter_resolver(tree, 400.0)
        U = mp_U_from_F(np.eye(3), T=600.0)
        self.assertEqual(float(temperature_at_point(U)), 600.0)
        values = resolve(tree, U)
        self.assertAlmostEqual(float(values["elastic"]["E"]), _C0 + _C1 * 600.0)


class TestChecks(unittest.TestCase):

    def test_polynomial_with_an_extra_key(self):
        tree = {"E": {"polynomial": {"coefficients": [1.0], "T0": 300.0}}}
        with self.assertRaisesRegex(ValueError, "E.polynomial"):
            check_parameter_forms(tree)

    def test_table_with_decreasing_knots(self):
        tree = {"elastic": {"E": {"table": {"T": [500.0, 300.0],
                                           "values": [1.0, 2.0]}}}}
        with self.assertRaisesRegex(ValueError, "elastic.E.table.T"):
            check_parameter_forms(tree)

    def test_builder_rejects_an_active_flag_inside_a_form(self):
        section = {"elastic": {
            "E": {"polynomial": {"coefficients": {
                "value": [1.0, 2.0], "active": True}}},
            "nu": 0.3,
        }}
        with self.assertRaisesRegex(ValueError, "not supported yet"):
            build_parameters(section)


class TestModel(unittest.TestCase):

    def test_scale_factor_at_the_reference_temperature(self):
        tree = _plain_tree()
        del tree["rotation matrix"]
        tree["elastic"]["E"] = {"polynomial": {
            "coefficients": np.array([250e3, -100.0])}}
        reference_temperature = 500.0
        model = SmallElasticPlastic(
            build_parameters(tree), DefType.FULL_3D,
            reference_temperature=reference_temperature)
        E_ref = 250e3 - 100.0 * reference_temperature
        self.assertAlmostEqual(
            model.shear_scale_factor, 2.0 * compute_mu(E_ref, 0.3))

    def test_mp_U_from_F_carries_a_temperature(self):
        U = mp_U_from_F(np.eye(2), T=350.0)
        self.assertIsInstance(U, GlobalFieldsAtPoint)
        self.assertEqual(U.fields["T"].shape, (1,))
        self.assertEqual(U.grad_fields["T"].shape, (1, 2))
        self.assertIsNone(temperature_at_point(mp_U_from_F(np.eye(2))))


if __name__ == "__main__":
    unittest.main()
