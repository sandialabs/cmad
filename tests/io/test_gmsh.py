"""Tests for the gmsh mesh reader (:mod:`cmad.io.gmsh`) and the suffix
dispatch (:mod:`cmad.io.mesh_io`).

Meshes are generated in-process with the gmsh API into a temporary ``.msh``
and read back.
"""
import tempfile
import unittest
from pathlib import Path

import gmsh
import numpy as np
import yaml

from cmad.cli.main import main as cmad_main
from cmad.fem.element_family import ElementFamily
from cmad.fem.mesh import coordinate_side_sets
from cmad.io.gmsh import GmshFormatError, read_gmsh_mesh
from cmad.io.mesh_io import read_mesh_file


def _write_tet_box(
        path: Path, *, size: float = 0.5, physical: bool = True,
) -> None:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        if physical:
            vols = [v[1] for v in gmsh.model.getEntities(3)]
            gmsh.model.addPhysicalGroup(3, vols, name="solid")
        else:
            gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def _write_hex_box(path: Path, *, n: int = 3) -> None:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("hexbox")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        for curve in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(curve[1], n + 1)
        for surf in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(surf[1])
            gmsh.model.mesh.setRecombine(2, surf[1])
        for vol in gmsh.model.getEntities(3):
            gmsh.model.mesh.setTransfiniteVolume(vol[1])
        vols = [v[1] for v in gmsh.model.getEntities(3)]
        gmsh.model.addPhysicalGroup(3, vols, name="solid")
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


class TestReadGmshMesh(unittest.TestCase):
    def test_tet_box(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "box.msh"
            _write_tet_box(path, size=0.5)
            mesh = read_gmsh_mesh(path)
        self.assertEqual(mesh.element_family, ElementFamily.TET_LINEAR)
        self.assertEqual(mesh.nodes.shape[1], 3)
        self.assertEqual(mesh.connectivity.shape[1], 4)
        self.assertEqual(list(mesh.element_blocks), ["solid"])
        self.assertEqual(
            mesh.element_blocks["solid"].shape[0], mesh.connectivity.shape[0],
        )
        self.assertEqual(mesh.element_block_ids["solid"], 1)
        self.assertEqual(mesh.node_sets, {})
        self.assertEqual(mesh.side_sets, {})
        np.testing.assert_allclose(mesh.nodes.min(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(mesh.nodes.max(axis=0), 1.0, atol=1e-12)

    def test_hex_box(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "hexbox.msh"
            _write_hex_box(path, n=3)
            mesh = read_gmsh_mesh(path)
        self.assertEqual(mesh.element_family, ElementFamily.HEX_LINEAR)
        self.assertEqual(mesh.connectivity.shape[1], 8)
        self.assertEqual(mesh.connectivity.shape[0], 27)

    def test_no_physical_group_yields_all_block(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nogroup.msh"
            _write_tet_box(path, size=0.5, physical=False)
            mesh = read_gmsh_mesh(path)
        self.assertEqual(list(mesh.element_blocks), ["all"])
        self.assertEqual(mesh.element_block_ids, {})

    def test_missing_file_raises(self) -> None:
        with self.assertRaisesRegex(GmshFormatError, "not found"):
            read_gmsh_mesh("does_not_exist.msh")


class TestReadMeshFileDispatch(unittest.TestCase):
    def test_msh_routes_to_gmsh(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "box.msh"
            _write_tet_box(path)
            mesh = read_mesh_file(path)
        self.assertEqual(mesh.element_family, ElementFamily.TET_LINEAR)

    def test_unknown_suffix_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unrecognized mesh suffix"):
            read_mesh_file("mesh.txt")


class TestGmshBoundingBoxSidesets(unittest.TestCase):
    """A gmsh mesh through the bounding box side-set builder (the BC path)."""

    def test_six_bounding_box_sides(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "box.msh"
            _write_tet_box(path, size=0.5)
            mesh = read_gmsh_mesh(path)
        sides = coordinate_side_sets(mesh)
        expected = {
            f"{axis}{end}_sides"
            for axis in ("x", "y", "z") for end in ("min", "max")
        }
        self.assertEqual(set(sides), expected)
        for pairs in sides.values():
            self.assertEqual(pairs.shape[1], 2)
            self.assertGreater(pairs.shape[0], 0)


class TestGmshPrimalEndToEnd(unittest.TestCase):
    """``cmad primal`` on a gmsh ``.msh`` through the suffix dispatch.

    Drives the whole deck path: ``read_mesh_file`` routes the ``.msh`` to the
    gmsh reader, ``build coordinate sidesets`` supplies the BC sets, and the
    elastic FE solve runs to completion.
    """

    def test_elastic_primal_on_gmsh_box(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _write_tet_box(tmp / "box.msh", size=0.6, physical=False)
            deck = {
                "problem": {"type": "fe"},
                "discretization": {
                    "mesh file": str(tmp / "box.msh"),
                    "build coordinate sidesets": True,
                    "num steps": 2,
                    "step size": 0.5,
                },
                "residuals": {
                    "global residual": {
                        "type": "small_disp_equilibrium", "def_type": "full_3d",
                    },
                    "local residual": {
                        "type": "elastic",
                        "materials": {
                            "all": {"elastic": {"kappa": 100.0, "mu": 50.0}},
                        },
                    },
                },
                "dirichlet bcs": {
                    "expression": {
                        "sym_x": ["equilibrium", 0, "xmin_sides", "0.0"],
                        "sym_y": ["equilibrium", 1, "ymin_sides", "0.0"],
                        "sym_z": ["equilibrium", 2, "zmin_sides", "0.0"],
                        "load_x": ["equilibrium", 0, "xmax_sides", "0.05 * t"],
                    },
                },
                "output": {
                    "path": str(tmp / "out"),
                    "exodus filename": "primal.exo",
                    "global residual": ["u"],
                    "local residual": {"all": ["cauchy"]},
                },
            }
            deck_path = tmp / "deck.yaml"
            deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))
            self.assertEqual(cmad_main(["primal", str(deck_path)]), 0)
            self.assertTrue((tmp / "out" / "primal.exo").exists())


if __name__ == "__main__":
    unittest.main()
