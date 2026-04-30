"""Unit tests for `cmad.fem.topology`.

Pins the per-(family, local_side_id) ref-side lift tables: outward
orientation (right-hand-rule ``cross(t_s, t_t)`` matches the documented
face-normal direction) and vertex-consistency (parameter-space face
vertices lift to the volume-frame positions of the corresponding
``_LOCAL_FACES_PER_ELEMENT`` entries).
"""
import unittest

import numpy as np

from cmad.fem.element_family import ElementFamily
from cmad.fem.topology import (
    _HEX_REFERENCE_NODES,
    _LOCAL_FACES_PER_ELEMENT,
    _TET_REFERENCE_NODES,
    ref_side_lift,
)

# Documented outward unit normals per (family, local_side_id), keyed
# off the face-numbering tables in ``cmad/fem/topology.py``.
_HEX_OUTWARD_NORMALS = np.array(
    [
        [0.0, 0.0, -1.0],   # face 0: -z
        [0.0, 0.0, +1.0],   # face 1: +z
        [0.0, -1.0, 0.0],   # face 2: -y
        [+1.0, 0.0, 0.0],   # face 3: +x
        [0.0, +1.0, 0.0],   # face 4: +y
        [-1.0, 0.0, 0.0],   # face 5: -x
    ],
)

_TET_OUTWARD_NORMALS = np.array(
    [
        [0.0, -1.0, 0.0],                       # face 0: -y
        [1.0, 1.0, 1.0] / np.sqrt(3.0),         # face 1: slant (+x+y+z)
        [-1.0, 0.0, 0.0],                       # face 2: -x
        [0.0, 0.0, -1.0],                       # face 3: -z
    ],
)

# Parameter-space vertex coords matching ``_LOCAL_FACES_PER_ELEMENT``
# row order. For a hex face with vertices (v0, v1, v2, v3) in CCW-
# from-outside order, the bilinear lift sends ``(s, t)`` over
# ``[-1, 1]^2`` in the order ``(-1, -1), (+1, -1), (+1, +1), (-1, +1)``
# to v0..v3. For a tet face with vertices (v0, v1, v2), the linear
# lift sends ``(s, t)`` over the unit triangle in the order
# ``(0, 0), (1, 0), (0, 1)`` to v0..v2.
_HEX_FACE_PARAM_VERTICES = np.array(
    [[-1.0, -1.0], [+1.0, -1.0], [+1.0, +1.0], [-1.0, +1.0]],
)
_TET_FACE_PARAM_VERTICES = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
)


_FAMILY_FIXTURES = [
    (
        ElementFamily.HEX_LINEAR,
        _HEX_REFERENCE_NODES,
        _HEX_OUTWARD_NORMALS,
        _HEX_FACE_PARAM_VERTICES,
    ),
    (
        ElementFamily.TET_LINEAR,
        _TET_REFERENCE_NODES,
        _TET_OUTWARD_NORMALS,
        _TET_FACE_PARAM_VERTICES,
    ),
]


class TestRefSideLift(unittest.TestCase):

    def test_lift_tangents_cross_to_outward_unit_normal(self):
        for family, _, expected_normals, _ in _FAMILY_FIXTURES:
            face_table = _LOCAL_FACES_PER_ELEMENT[family]
            for local_side_id in range(face_table.shape[0]):
                _, tangents = ref_side_lift(family, local_side_id)
                cross = np.cross(tangents[:, 0], tangents[:, 1])
                mag = np.linalg.norm(cross)
                self.assertGreater(mag, 0.0)
                np.testing.assert_allclose(
                    cross / mag,
                    expected_normals[local_side_id],
                    atol=1e-12,
                    err_msg=(
                        f"family={family.name}, "
                        f"local_side_id={local_side_id}"
                    ),
                )

    def test_lift_param_vertices_match_face_volume_vertices(self):
        for family, ref_nodes, _, param_vertices in _FAMILY_FIXTURES:
            face_table = _LOCAL_FACES_PER_ELEMENT[family]
            for local_side_id in range(face_table.shape[0]):
                origin, tangents = ref_side_lift(family, local_side_id)
                face_vertex_ids = face_table[local_side_id]
                for k, st in enumerate(param_vertices):
                    lifted = origin + tangents @ st
                    expected = ref_nodes[face_vertex_ids[k]]
                    np.testing.assert_allclose(
                        lifted, expected, atol=1e-12,
                        err_msg=(
                            f"family={family.name}, "
                            f"local_side_id={local_side_id}, vertex k={k}"
                        ),
                    )

    def test_tet_slant_face_jacobian_magnitude_matches_sqrt3(self):
        # Slant face area in the volume frame = sqrt(3) / 2; with
        # unit-triangle quadrature weights summing to 1/2, the per-IP
        # tangent-cross magnitude must equal sqrt(3) so the surface
        # element integrates to the face area.
        _, tangents = ref_side_lift(ElementFamily.TET_LINEAR, 1)
        cross = np.cross(tangents[:, 0], tangents[:, 1])
        np.testing.assert_allclose(
            np.linalg.norm(cross), np.sqrt(3.0), atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
