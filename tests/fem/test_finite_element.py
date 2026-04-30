"""Unit tests for `cmad.fem.finite_element`.

Covers FiniteElement.num_dofs_per_element across P1 (shipped) and a
synthetic P2 / Q2 / DG0; constructor validation rejects negative DOF
counts and non-EntityType keys; the shipped P1_TET and Q1_HEX module
constants pair with the expected element families and interpolants.
"""
import unittest
from typing import cast

from cmad.fem.element_family import ElementFamily
from cmad.fem.finite_element import (
    P1_TET,
    Q1_HEX,
    EntityType,
    FiniteElement,
)
from cmad.fem.interpolants import hex_linear, tet_linear


class TestFiniteElementDofCount(unittest.TestCase):

    def test_p1_tet_has_4_dofs(self):
        self.assertEqual(P1_TET.num_dofs_per_element, 4)

    def test_q1_hex_has_8_dofs(self):
        self.assertEqual(Q1_HEX.num_dofs_per_element, 8)

    def test_synthetic_p2_tet_has_10_dofs(self):
        # P2 tet: 1 DOF per vertex (4) + 1 DOF per edge (6) = 10.
        p2_tet = FiniteElement(
            name="P2_TET",
            element_family=ElementFamily.TET_LINEAR,
            dofs_per_entity={EntityType.VERTEX: 1, EntityType.EDGE: 1},
            interpolant_fn=tet_linear,
        )
        self.assertEqual(p2_tet.num_dofs_per_element, 10)

    def test_synthetic_q2_full_hex_has_27_dofs(self):
        # Q2 full hex: 1/V (8) + 1/E (12) + 1/F (6) + 1/C (1) = 27.
        q2_hex = FiniteElement(
            name="Q2_HEX_FULL",
            element_family=ElementFamily.HEX_LINEAR,
            dofs_per_entity={
                EntityType.VERTEX: 1,
                EntityType.EDGE: 1,
                EntityType.FACE: 1,
                EntityType.CELL: 1,
            },
            interpolant_fn=hex_linear,
        )
        self.assertEqual(q2_hex.num_dofs_per_element, 27)

    def test_synthetic_dg0_tet_has_1_dof(self):
        # DG0 (element-constant) on a tet: 1 cell-local DOF, no
        # inter-element sharing.
        dg0_tet = FiniteElement(
            name="DG0_TET",
            element_family=ElementFamily.TET_LINEAR,
            dofs_per_entity={EntityType.CELL: 1},
            interpolant_fn=tet_linear,
        )
        self.assertEqual(dg0_tet.num_dofs_per_element, 1)

    def test_omitted_entity_types_imply_zero(self):
        # Only VERTEX provided; EDGE/FACE/CELL implicitly 0.
        fe = FiniteElement(
            name="vertex_only_tet",
            element_family=ElementFamily.TET_LINEAR,
            dofs_per_entity={EntityType.VERTEX: 1},
            interpolant_fn=tet_linear,
        )
        self.assertEqual(fe.num_dofs_per_element, 4)


class TestFiniteElementValidation(unittest.TestCase):

    def test_rejects_negative_dof_count(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            FiniteElement(
                name="bad",
                element_family=ElementFamily.TET_LINEAR,
                dofs_per_entity={EntityType.VERTEX: -1},
                interpolant_fn=tet_linear,
            )

    def test_rejects_non_entity_type_key(self):
        # Cast a string-keyed dict through dict[EntityType, int] so
        # the call site type-checks; the runtime validation should
        # fire.
        bad_dofs = cast("dict[EntityType, int]", {"vertex": 1})
        with self.assertRaisesRegex(ValueError, "EntityType members"):
            FiniteElement(
                name="bad",
                element_family=ElementFamily.TET_LINEAR,
                dofs_per_entity=bad_dofs,
                interpolant_fn=tet_linear,
            )


class TestShippedConstants(unittest.TestCase):

    def test_p1_tet_pairs_with_tet_linear(self):
        self.assertIs(P1_TET.interpolant_fn, tet_linear)
        self.assertEqual(P1_TET.element_family, ElementFamily.TET_LINEAR)

    def test_q1_hex_pairs_with_hex_linear(self):
        self.assertIs(Q1_HEX.interpolant_fn, hex_linear)
        self.assertEqual(Q1_HEX.element_family, ElementFamily.HEX_LINEAR)


if __name__ == "__main__":
    unittest.main()
