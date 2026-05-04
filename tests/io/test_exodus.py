"""Tests for :mod:`cmad.io.exodus`.

External-fixture coverage uses ``tests/io/fixtures/small_hex.exo``
(a 2x2x2 hex mesh on the unit cube, one element block, two nodesets,
no sidesets — meshio has no first-class sideset concept). Internal
round-trip tests exercise the writer + reader together on
``StructuredHexMesh`` instances that include all six built-in sidesets.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import netCDF4
import numpy as np

from cmad.fem.element_family import ElementFamily
from cmad.fem.mesh import Mesh, StructuredHexMesh, hex_to_tet_split
from cmad.io.exodus import (
    ExodusFormatError,
    ExodusWriter,
    read_mesh,
    read_results,
)
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType

_FIXTURE = Path(__file__).parent / "fixtures" / "small_hex.exo"


class TestReadMeshSmallHex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mesh = read_mesh(_FIXTURE)

    def test_node_count_and_coords(self):
        self.assertEqual(self.mesh.nodes.shape, (27, 3))
        # corner nodes of the unit cube
        np.testing.assert_allclose(self.mesh.nodes[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(self.mesh.nodes[-1], [1.0, 1.0, 1.0])

    def test_element_family_hex_linear(self):
        self.assertEqual(self.mesh.element_family, ElementFamily.HEX_LINEAR)

    def test_connectivity_shape(self):
        self.assertEqual(self.mesh.connectivity.shape, (8, 8))

    def test_first_element_node_ordering_matches_hex_linear(self):
        # cmad hex_linear: bottom CCW from (-,-,-), top CCW from (-,-,+).
        # Element 0 spans the (-,-,-) corner cell of the 2x2x2 mesh.
        expected = np.array([
            [0.0, 0.0, 0.0],   # 0: (-,-,-)
            [0.5, 0.0, 0.0],   # 1: (+,-,-)
            [0.5, 0.5, 0.0],   # 2: (+,+,-)
            [0.0, 0.5, 0.0],   # 3: (-,+,-)
            [0.0, 0.0, 0.5],   # 4: (-,-,+)
            [0.5, 0.0, 0.5],   # 5: (+,-,+)
            [0.5, 0.5, 0.5],   # 6: (+,+,+)
            [0.0, 0.5, 0.5],   # 7: (-,+,+)
        ])
        actual = self.mesh.nodes[self.mesh.connectivity[0]]
        np.testing.assert_allclose(actual, expected)

    def test_single_element_block_default_named(self):
        # No eb_names in meshio output; reader defaults to
        # f"block_{eb_prop1[i]}" — the fixture has eb_prop1 = [1].
        self.assertEqual(list(self.mesh.element_blocks), ["block_1"])
        np.testing.assert_array_equal(
            self.mesh.element_blocks["block_1"], np.arange(8)
        )
        self.assertEqual(self.mesh.element_block_ids["block_1"], 1)

    def test_node_sets_decoded_and_zero_based(self):
        self.assertEqual(
            sorted(self.mesh.node_sets), ["xmax_nodes", "xmin_nodes"]
        )
        # x = 0 face: 9 nodes at indices 0..8 in cmad's (i, j, k) ordering
        np.testing.assert_array_equal(
            np.sort(self.mesh.node_sets["xmin_nodes"]), np.arange(9)
        )
        # x = 1 face: 9 nodes at indices 18..26
        np.testing.assert_array_equal(
            np.sort(self.mesh.node_sets["xmax_nodes"]), np.arange(18, 27)
        )

    def test_node_set_coordinates_lie_on_expected_face(self):
        xmin = self.mesh.nodes[self.mesh.node_sets["xmin_nodes"]]
        xmax = self.mesh.nodes[self.mesh.node_sets["xmax_nodes"]]
        np.testing.assert_allclose(xmin[:, 0], 0.0)
        np.testing.assert_allclose(xmax[:, 0], 1.0)

    def test_no_side_sets(self):
        self.assertEqual(self.mesh.side_sets, {})


class TestReadMeshErrors(unittest.TestCase):

    def test_missing_file_raises(self):
        with self.assertRaises((FileNotFoundError, OSError)):
            read_mesh(_FIXTURE.parent / "does_not_exist.exo")


def _assert_meshes_equal(test: unittest.TestCase, expected, actual):
    np.testing.assert_allclose(actual.nodes, expected.nodes)
    np.testing.assert_array_equal(actual.connectivity, expected.connectivity)
    test.assertEqual(actual.element_family, expected.element_family)

    test.assertEqual(set(actual.element_blocks), set(expected.element_blocks))
    for name in expected.element_blocks:
        np.testing.assert_array_equal(
            actual.element_blocks[name], expected.element_blocks[name]
        )

    test.assertEqual(set(actual.node_sets), set(expected.node_sets))
    for name in expected.node_sets:
        np.testing.assert_array_equal(
            np.sort(actual.node_sets[name]),
            np.sort(expected.node_sets[name]),
        )

    test.assertEqual(set(actual.side_sets), set(expected.side_sets))
    for name in expected.side_sets:
        np.testing.assert_array_equal(
            actual.side_sets[name], expected.side_sets[name]
        )


class TestWriterReaderRoundTrip(unittest.TestCase):

    def test_structured_hex_mesh_with_sidesets(self):
        mesh = StructuredHexMesh(
            lengths=(1.0, 2.0, 3.0),
            divisions=(2, 3, 4),
            origin=(0.5, 1.0, 1.5),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rt_hex.exo"
            with ExodusWriter(path, mesh, title="hex round-trip"):
                pass
            rt = read_mesh(path)
        _assert_meshes_equal(self, mesh, rt)
        # Sanity: 6 named sidesets survived the round trip.
        self.assertEqual(
            sorted(rt.side_sets),
            sorted([
                "xmin_sides", "xmax_sides",
                "ymin_sides", "ymax_sides",
                "zmin_sides", "zmax_sides",
            ]),
        )

    def test_tet_mesh_round_trip(self):
        hex_mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        tet_mesh = hex_to_tet_split(hex_mesh)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rt_tet.exo"
            with ExodusWriter(path, tet_mesh):
                pass
            rt = read_mesh(path)
        _assert_meshes_equal(self, tet_mesh, rt)
        self.assertEqual(rt.element_family, ElementFamily.TET_LINEAR)


class TestSetIdPreservation(unittest.TestCase):

    def test_round_trip_preserves_non_sequential_side_set_ids(self):
        # Build a mesh with cmad's writer (sequential IDs by default),
        # then mutate the file: set ss_prop1 to non-sequential values and
        # blank ss_names so the reader exercises the f"sideset_{id}"
        # default-name path. Read with read_mesh, write again, read again,
        # and confirm IDs survived the second round-trip.
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
        non_seq_ids = [3, 1, 7, 4, 5, 6]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "non_seq_in.exo"
            with ExodusWriter(path, mesh):
                pass
            with netCDF4.Dataset(str(path), "r+") as ds:
                ds["ss_prop1"][:] = np.array(non_seq_ids, dtype=np.int32)
                ds["ss_names"][:] = np.zeros(
                    ds["ss_names"].shape, dtype="S1"
                )
            in_mesh = read_mesh(path)
            expected = {f"sideset_{i}": i for i in non_seq_ids}
            self.assertEqual(in_mesh.side_set_ids, expected)

            out_path = Path(tmpdir) / "non_seq_rt.exo"
            with ExodusWriter(out_path, in_mesh):
                pass
            rt = read_mesh(out_path)
        self.assertEqual(rt.side_set_ids, expected)

    def test_in_house_mesh_writer_assigns_sequential_ids_when_ids_empty(self):
        mesh = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        self.assertEqual(mesh.element_block_ids, {})
        self.assertEqual(mesh.node_set_ids, {})
        self.assertEqual(mesh.side_set_ids, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "in_house.exo"
            with ExodusWriter(path, mesh):
                pass
            rt = read_mesh(path)
        self.assertEqual(
            rt.side_set_ids,
            {
                "xmin_sides": 1,
                "xmax_sides": 2,
                "ymin_sides": 3,
                "ymax_sides": 4,
                "zmin_sides": 5,
                "zmax_sides": 6,
            },
        )


class TestWriteStepSchema(unittest.TestCase):
    """Verify ExodusWriter.write_step lays down the expected
    netCDF4 structure (dimensions, name vars, truth table, value
    arrays). Inspects the file via netCDF4.Dataset directly; the
    full round-trip via read_results lands in a follow-on commit."""

    def _mesh(self):
        return StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))

    def test_writer_with_no_specs_emits_no_var_dims(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no_vars.exo"
            with ExodusWriter(path, self._mesh()):
                pass
            with netCDF4.Dataset(str(path)) as ds:
                self.assertNotIn("num_nod_var", ds.dimensions)
                self.assertNotIn("num_elem_var", ds.dimensions)

    def test_nodal_schema_dims_and_names(self):
        mesh = self._mesh()
        specs = [
            FieldSpec("displacement", VarType.VECTOR),
            FieldSpec("temperature", VarType.SCALAR),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nodal.exo"
            with ExodusWriter(path, mesh, nodal_field_specs=specs):
                pass
            with netCDF4.Dataset(str(path)) as ds:
                self.assertEqual(int(ds.dimensions["num_nod_var"].size), 4)
                names = [
                    ds["name_nod_var"][i].tobytes().rstrip(b"\x00").decode()
                    for i in range(4)
                ]
                self.assertEqual(
                    names,
                    [
                        "displacement_x", "displacement_y",
                        "displacement_z", "temperature",
                    ],
                )
                # SEACAS-canonical layout: single 3D nodal-vars array
                # indexed (time_step, num_nod_var, num_nodes).
                self.assertIn("vals_nod_var", ds.variables)
                self.assertEqual(
                    ds["vals_nod_var"].dimensions,
                    ("time_step", "num_nod_var", "num_nodes"),
                )

    def test_element_schema_truth_table_sparsity(self):
        # Two-block mesh: block_a declares cauchy (SYM_TENSOR);
        # block_b declares damage (SCALAR). Truth table reflects this.
        base = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        mesh = Mesh(
            nodes=base.nodes,
            connectivity=base.connectivity,
            element_family=base.element_family,
            element_blocks={
                "block_a": np.arange(0, 4, dtype=np.intp),
                "block_b": np.arange(4, 8, dtype=np.intp),
            },
            node_sets={},
            side_sets={},
        )
        specs = {
            "block_a": [FieldSpec("cauchy", VarType.SYM_TENSOR)],
            "block_b": [FieldSpec("damage", VarType.SCALAR)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "elem.exo"
            with ExodusWriter(path, mesh, element_field_specs=specs):
                pass
            with netCDF4.Dataset(str(path)) as ds:
                self.assertEqual(
                    int(ds.dimensions["num_elem_var"].size), 7,
                )
                names = [
                    ds["name_elem_var"][i].tobytes().rstrip(b"\x00").decode()
                    for i in range(7)
                ]
                self.assertEqual(
                    names,
                    [
                        "cauchy_xx", "cauchy_yy", "cauchy_zz",
                        "cauchy_xy", "cauchy_xz", "cauchy_yz",
                        "damage",
                    ],
                )
                truth = np.asarray(ds["elem_var_tab"][:])
                self.assertEqual(truth.shape, (2, 7))
                np.testing.assert_array_equal(
                    truth[0], [1, 1, 1, 1, 1, 1, 0],
                )
                np.testing.assert_array_equal(
                    truth[1], [0, 0, 0, 0, 0, 0, 1],
                )
                for n in range(1, 7):
                    self.assertIn(f"vals_elem_var{n}eb1", ds.variables)
                self.assertNotIn("vals_elem_var7eb1", ds.variables)
                self.assertIn("vals_elem_var7eb2", ds.variables)
                for n in range(1, 7):
                    self.assertNotIn(f"vals_elem_var{n}eb2", ds.variables)

    def test_write_step_appends_time_and_values(self):
        mesh = self._mesh()
        n_nodes = mesh.nodes.shape[0]
        nodal_specs = [
            FieldSpec("u", VarType.VECTOR),
            FieldSpec("T", VarType.SCALAR),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.exo"
            with ExodusWriter(
                path, mesh, nodal_field_specs=nodal_specs,
            ) as w:
                u_step0 = np.zeros((n_nodes, 3))
                T_step0 = np.zeros(n_nodes)
                w.write_step(0.0, {"u": u_step0, "T": T_step0})
                u_step1 = np.tile(np.arange(3.0), (n_nodes, 1))
                T_step1 = 0.5 * np.arange(n_nodes, dtype=np.float64)
                w.write_step(0.25, {"u": u_step1, "T": T_step1})
            with netCDF4.Dataset(str(path)) as ds:
                np.testing.assert_allclose(
                    np.asarray(ds["time_whole"][:]), [0.0, 0.25],
                )
                vals = np.asarray(ds["vals_nod_var"][:])
                self.assertEqual(vals.shape, (2, 4, n_nodes))
                self.assertEqual(vals[1, 0, :].tolist(), [0.0] * n_nodes)
                np.testing.assert_allclose(
                    vals[1, 1, :], np.full(n_nodes, 1.0),
                )
                np.testing.assert_allclose(
                    vals[1, 2, :], np.full(n_nodes, 2.0),
                )
                np.testing.assert_allclose(
                    vals[1, 3, :],
                    0.5 * np.arange(n_nodes, dtype=np.float64),
                )

    def test_write_step_sym_tensor_disk_permutation(self):
        # Internal sym-tensor input becomes Exodus-order on disk.
        mesh = self._mesh()
        n_nodes = mesh.nodes.shape[0]
        specs = [FieldSpec("cauchy", VarType.SYM_TENSOR)]
        # internal: [xx, xy, xz, yy, yz, zz] = [10, 12, 13, 22, 23, 33]
        # exodus:   [xx, yy, zz, xy, xz, yz] = [10, 22, 33, 12, 13, 23]
        internal = np.broadcast_to(
            np.array([10.0, 12.0, 13.0, 22.0, 23.0, 33.0]),
            (n_nodes, 6),
        ).copy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sym.exo"
            with ExodusWriter(
                path, mesh, nodal_field_specs=specs,
            ) as w:
                w.write_step(1.0, {"cauchy": internal})
            with netCDF4.Dataset(str(path)) as ds:
                # vals_nod_var[step, comp, node]; comps in disk order
                # [xx, yy, zz, xy, xz, yz] for SYM_TENSOR.
                vals = np.asarray(ds["vals_nod_var"][:])
                self.assertEqual(vals.shape, (1, 6, n_nodes))
                self.assertEqual(float(vals[0, 0, 0]), 10.0)  # xx
                self.assertEqual(float(vals[0, 1, 0]), 22.0)  # yy
                self.assertEqual(float(vals[0, 2, 0]), 33.0)  # zz
                self.assertEqual(float(vals[0, 3, 0]), 12.0)  # xy
                self.assertEqual(float(vals[0, 4, 0]), 13.0)  # xz
                self.assertEqual(float(vals[0, 5, 0]), 23.0)  # yz

    def test_write_step_rejects_extra_data_keys(self):
        mesh = self._mesh()
        specs = [FieldSpec("u", VarType.VECTOR)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extra.exo"
            with ExodusWriter(
                path, mesh, nodal_field_specs=specs,
            ) as w, self.assertRaises(ValueError):
                w.write_step(
                    0.0,
                    {
                        "u": np.zeros((mesh.nodes.shape[0], 3)),
                        "extra": np.zeros((mesh.nodes.shape[0], 3)),
                    },
                )

    def test_write_step_rejects_shape_mismatch(self):
        mesh = self._mesh()
        specs = [FieldSpec("u", VarType.VECTOR)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shape.exo"
            with ExodusWriter(
                path, mesh, nodal_field_specs=specs,
            ) as w, self.assertRaises(ValueError):
                w.write_step(
                    0.0,
                    {"u": np.zeros((mesh.nodes.shape[0], 4))},
                )

    def test_write_step_rejects_when_no_specs(self):
        mesh = self._mesh()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nospecs.exo"
            with ExodusWriter(path, mesh) as w, self.assertRaises(ValueError):
                w.write_step(0.0, {})

    def test_zero_steps_writer_close_does_not_corrupt_file(self):
        mesh = self._mesh()
        specs = [FieldSpec("u", VarType.VECTOR)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zerosteps.exo"
            with ExodusWriter(
                path, mesh, nodal_field_specs=specs,
            ):
                pass
            with netCDF4.Dataset(str(path)) as ds:
                self.assertEqual(
                    int(ds.dimensions["num_nod_var"].size), 3,
                )
                self.assertEqual(
                    np.asarray(ds["time_whole"][:]).shape, (0,),
                )
                self.assertEqual(
                    np.asarray(ds["vals_nod_var"][:]).shape,
                    (0, 3, mesh.nodes.shape[0]),
                )

    def test_constructor_rejects_unknown_block(self):
        mesh = self._mesh()
        specs = {"nope": [FieldSpec("foo", VarType.SCALAR)]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "badblock.exo"
            with self.assertRaises(ExodusFormatError):
                ExodusWriter(path, mesh, element_field_specs=specs)

    def test_constructor_rejects_var_type_mismatch_across_blocks(self):
        base = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        mesh = Mesh(
            nodes=base.nodes,
            connectivity=base.connectivity,
            element_family=base.element_family,
            element_blocks={
                "block_a": np.arange(0, 4, dtype=np.intp),
                "block_b": np.arange(4, 8, dtype=np.intp),
            },
            node_sets={},
            side_sets={},
        )
        specs = {
            "block_a": [FieldSpec("stress", VarType.SYM_TENSOR)],
            "block_b": [FieldSpec("stress", VarType.SCALAR)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mismatch.exo"
            with self.assertRaises(ExodusFormatError):
                ExodusWriter(path, mesh, element_field_specs=specs)


class TestResultRoundTrip(unittest.TestCase):
    """End-to-end ExodusWriter.write_step + read_results round-trip."""

    def _mesh(self):
        return StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))

    def _two_block_mesh(self):
        base = StructuredHexMesh((1.0, 1.0, 1.0), (2, 2, 2))
        return Mesh(
            nodes=base.nodes,
            connectivity=base.connectivity,
            element_family=base.element_family,
            element_blocks={
                "block_a": np.arange(0, 4, dtype=np.intp),
                "block_b": np.arange(4, 8, dtype=np.intp),
            },
            node_sets={},
            side_sets={},
        )

    def test_nodal_only_round_trip(self):
        mesh = self._mesh()
        n_nodes = mesh.nodes.shape[0]
        specs = [
            FieldSpec("displacement", VarType.VECTOR),
            FieldSpec("temperature", VarType.SCALAR),
        ]
        rng = np.random.default_rng(0)
        u = rng.standard_normal((3, n_nodes, 3))
        T = rng.standard_normal((3, n_nodes))
        times = [0.0, 0.1, 0.25]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nodal_rt.exo"
            with ExodusWriter(
                path, mesh, nodal_field_specs=specs,
            ) as w:
                for s, t in enumerate(times):
                    w.write_step(
                        t,
                        {"displacement": u[s], "temperature": T[s]},
                    )
            results = read_results(
                path,
                nodal_field_specs=specs,
            )
        np.testing.assert_allclose(results.time, times)
        self.assertEqual(set(results.nodal), {"displacement", "temperature"})
        self.assertEqual(results.element, {})
        self.assertEqual(results.nodal["displacement"].shape, (3, n_nodes, 3))
        self.assertEqual(results.nodal["temperature"].shape, (3, n_nodes))
        np.testing.assert_allclose(results.nodal["displacement"], u)
        np.testing.assert_allclose(results.nodal["temperature"], T)

    def test_element_only_round_trip_sym_tensor_permutation(self):
        # Two blocks; SYM_TENSOR + SCALAR per block. Verifies that
        # internal-order data written is read back as internal-order
        # (permutation goes to disk and back).
        mesh = self._two_block_mesh()
        specs = {
            "block_a": [
                FieldSpec("cauchy", VarType.SYM_TENSOR),
                FieldSpec("alpha", VarType.SCALAR),
            ],
            "block_b": [
                FieldSpec("cauchy", VarType.SYM_TENSOR),
                FieldSpec("alpha", VarType.SCALAR),
            ],
        }
        rng = np.random.default_rng(1)
        cauchy_a = rng.standard_normal((2, 4, 6))
        cauchy_b = rng.standard_normal((2, 4, 6))
        alpha_a = rng.standard_normal((2, 4))
        alpha_b = rng.standard_normal((2, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "elem_rt.exo"
            with ExodusWriter(
                path, mesh, element_field_specs=specs,
            ) as w:
                for s in range(2):
                    w.write_step(
                        float(s),
                        element_data={
                            "block_a": {
                                "cauchy": cauchy_a[s],
                                "alpha": alpha_a[s],
                            },
                            "block_b": {
                                "cauchy": cauchy_b[s],
                                "alpha": alpha_b[s],
                            },
                        },
                    )
            results = read_results(
                path, element_field_specs=specs,
            )
        np.testing.assert_allclose(results.time, [0.0, 1.0])
        self.assertEqual(results.nodal, {})
        self.assertEqual(set(results.element), {"block_a", "block_b"})
        np.testing.assert_allclose(
            results.element["block_a"]["cauchy"], cauchy_a,
        )
        np.testing.assert_allclose(
            results.element["block_a"]["alpha"], alpha_a,
        )
        np.testing.assert_allclose(
            results.element["block_b"]["cauchy"], cauchy_b,
        )
        np.testing.assert_allclose(
            results.element["block_b"]["alpha"], alpha_b,
        )

    def test_combined_all_var_types(self):
        mesh = self._mesh()
        n_nodes = mesh.nodes.shape[0]
        n_elem = int(mesh.element_blocks["all"].shape[0])
        nodal = [
            FieldSpec("u", VarType.VECTOR),
            FieldSpec("T", VarType.SCALAR),
        ]
        elem = {
            "all": [
                FieldSpec("cauchy", VarType.SYM_TENSOR),
                FieldSpec("F", VarType.TENSOR),
            ],
        }
        rng = np.random.default_rng(2)
        u = rng.standard_normal((n_nodes, 3))
        T = rng.standard_normal(n_nodes)
        cauchy = rng.standard_normal((n_elem, 6))
        F = rng.standard_normal((n_elem, 9))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "combined.exo"
            with ExodusWriter(
                path, mesh,
                nodal_field_specs=nodal,
                element_field_specs=elem,
            ) as w:
                w.write_step(
                    0.5,
                    {"u": u, "T": T},
                    {"all": {"cauchy": cauchy, "F": F}},
                )
            results = read_results(
                path,
                nodal_field_specs=nodal,
                element_field_specs=elem,
            )
        np.testing.assert_allclose(results.time, [0.5])
        np.testing.assert_allclose(results.nodal["u"], u[None, ...])
        np.testing.assert_allclose(results.nodal["T"], T[None, ...])
        np.testing.assert_allclose(
            results.element["all"]["cauchy"], cauchy[None, ...],
        )
        np.testing.assert_allclose(
            results.element["all"]["F"], F[None, ...],
        )

    def test_alias_renames_on_read(self):
        # Writer uses on-disk root "displacement_measured"; reader
        # asks for "displacement" with an alias and gets that key back.
        mesh = self._mesh()
        n_nodes = mesh.nodes.shape[0]
        on_disk = [FieldSpec("displacement_measured", VarType.VECTOR)]
        cmad_side = [FieldSpec("displacement", VarType.VECTOR)]
        rng = np.random.default_rng(3)
        u = rng.standard_normal((n_nodes, 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alias.exo"
            with ExodusWriter(
                path, mesh, nodal_field_specs=on_disk,
            ) as w:
                w.write_step(0.0, {"displacement_measured": u})
            results = read_results(
                path,
                nodal_field_specs=cmad_side,
                field_name_aliases={
                    "displacement": "displacement_measured",
                },
            )
        self.assertEqual(set(results.nodal), {"displacement"})
        np.testing.assert_allclose(
            results.nodal["displacement"], u[None, ...],
        )

    def test_per_block_truth_table_sparsity(self):
        # block_a declares cauchy; block_b does not. Reading block_a
        # works; asking the reader for cauchy on block_b raises.
        mesh = self._two_block_mesh()
        write_specs = {
            "block_a": [FieldSpec("cauchy", VarType.SYM_TENSOR)],
        }
        rng = np.random.default_rng(4)
        cauchy_a = rng.standard_normal((4, 6))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sparse.exo"
            with ExodusWriter(
                path, mesh, element_field_specs=write_specs,
            ) as w:
                w.write_step(
                    0.0,
                    element_data={
                        "block_a": {"cauchy": cauchy_a},
                    },
                )
            results = read_results(
                path,
                element_field_specs={
                    "block_a": [FieldSpec("cauchy", VarType.SYM_TENSOR)],
                },
            )
            np.testing.assert_allclose(
                results.element["block_a"]["cauchy"], cauchy_a[None, ...],
            )
            with self.assertRaises(ExodusFormatError):
                read_results(
                    path,
                    element_field_specs={
                        "block_b": [
                            FieldSpec("cauchy", VarType.SYM_TENSOR),
                        ],
                    },
                )

    def test_zero_steps_round_trip(self):
        mesh = self._mesh()
        specs = [FieldSpec("u", VarType.VECTOR)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zero.exo"
            with ExodusWriter(
                path, mesh, nodal_field_specs=specs,
            ):
                pass
            results = read_results(path, nodal_field_specs=specs)
        self.assertEqual(results.time.shape, (0,))
        self.assertEqual(
            results.nodal["u"].shape, (0, mesh.nodes.shape[0], 3),
        )

    def test_alias_rejects_unknown_spec_key(self):
        mesh = self._mesh()
        specs = [FieldSpec("u", VarType.VECTOR)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_alias.exo"
            with ExodusWriter(
                path, mesh, nodal_field_specs=specs,
            ):
                pass
            with self.assertRaises(ExodusFormatError):
                read_results(
                    path,
                    nodal_field_specs=specs,
                    field_name_aliases={"not_a_spec": "u"},
                )

    def test_read_rejects_missing_nodal_component(self):
        mesh = self._mesh()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.exo"
            with ExodusWriter(path, mesh):
                pass
            with self.assertRaises(ExodusFormatError):
                read_results(
                    path,
                    nodal_field_specs=[FieldSpec("u", VarType.VECTOR)],
                )

    def test_read_rejects_unknown_block(self):
        mesh = self._two_block_mesh()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_block.exo"
            with ExodusWriter(path, mesh):
                pass
            with self.assertRaises(ExodusFormatError):
                read_results(
                    path,
                    element_field_specs={
                        "nope": [FieldSpec("foo", VarType.SCALAR)],
                    },
                )


if __name__ == "__main__":
    unittest.main()
