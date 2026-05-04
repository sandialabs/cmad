"""Tests for :func:`cmad.io.writers.write_fe_exodus`.

Spec-resolution unit tests use a stub :class:`FEProblem`; the round-
trip smoke test builds a real 1-block hex-cube FE problem with zero
displacement and confirms the Exodus file reads back with the GR's
declared default surface populated.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from jax.tree_util import tree_map

from cmad.fem.dof import GlobalFieldLayout, build_dof_map
from cmad.fem.fe_problem import FEState, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.global_residuals.modes import GlobalResidualMode
from cmad.global_residuals.small_disp_equilibrium import SmallDispEquilibrium
from cmad.io.exodus import read_results
from cmad.io.results import FieldSpec
from cmad.io.writers import _resolve_fe_field_specs, write_fe_exodus
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.var_types import VarType
from cmad.parameters.parameters import Parameters

_DISPLACEMENT_FS = FieldSpec("displacement", VarType.VECTOR)
_CAUCHY_FS = FieldSpec("cauchy", VarType.SYM_TENSOR)


class _StubMesh:
    def __init__(self, blocks: tuple[str, ...]) -> None:
        self.element_blocks = {b: np.array([0]) for b in blocks}


class _StubGR:
    def __init__(self, defaults: dict[str, list[FieldSpec]]) -> None:
        self._defaults = defaults

    def default_output_fields(self) -> dict[str, list[FieldSpec]]:
        return self._defaults


class _StubFEProblem:
    def __init__(
            self,
            blocks: tuple[str, ...],
            defaults: dict[str, list[FieldSpec]],
    ) -> None:
        self.mesh = _StubMesh(blocks)
        self.gr = _StubGR(defaults)


def _stub_problem(
        blocks: tuple[str, ...] = ("blk_a",),
) -> _StubFEProblem:
    return _StubFEProblem(
        blocks,
        defaults={"nodal": [_DISPLACEMENT_FS], "element": [_CAUCHY_FS]},
    )


class TestResolveFieldSpecs(unittest.TestCase):
    def test_default_when_deck_omits_both(self):
        fe_problem = _stub_problem(blocks=("blk_a", "blk_b"))
        nodal, elem_by_block = _resolve_fe_field_specs(
            output_section={}, fe_problem=fe_problem,  # type: ignore[arg-type]
        )
        self.assertEqual(nodal, [_DISPLACEMENT_FS])
        self.assertEqual(
            elem_by_block,
            {"blk_a": [_CAUCHY_FS], "blk_b": [_CAUCHY_FS]},
        )

    def test_deck_overrides_nodal_only(self):
        fe_problem = _stub_problem()
        deck = {
            "nodal fields": [
                {"name": "displacement", "var_type": "vector"},
            ],
        }
        nodal, elem_by_block = _resolve_fe_field_specs(
            output_section=deck, fe_problem=fe_problem,  # type: ignore[arg-type]
        )
        self.assertEqual(nodal, [_DISPLACEMENT_FS])
        self.assertEqual(elem_by_block, {"blk_a": [_CAUCHY_FS]})

    def test_deck_overrides_element_only(self):
        fe_problem = _stub_problem()
        deck = {
            "element fields by block": {
                "blk_a": [
                    {"name": "cauchy", "var_type": "sym_tensor"},
                ],
            },
        }
        nodal, elem_by_block = _resolve_fe_field_specs(
            output_section=deck, fe_problem=fe_problem,  # type: ignore[arg-type]
        )
        self.assertEqual(nodal, [_DISPLACEMENT_FS])
        self.assertEqual(elem_by_block, {"blk_a": [_CAUCHY_FS]})

    def test_deck_overrides_both(self):
        fe_problem = _stub_problem()
        deck = {
            "nodal fields": [
                {"name": "u_alt", "var_type": "vector"},
            ],
            "element fields by block": {
                "blk_a": [
                    {"name": "stress_alt", "var_type": "sym_tensor"},
                ],
            },
        }
        nodal, elem_by_block = _resolve_fe_field_specs(
            output_section=deck, fe_problem=fe_problem,  # type: ignore[arg-type]
        )
        self.assertEqual(
            nodal, [FieldSpec("u_alt", VarType.VECTOR)],
        )
        self.assertEqual(
            elem_by_block,
            {"blk_a": [FieldSpec("stress_alt", VarType.SYM_TENSOR)]},
        )

    def test_var_type_string_coerces_to_enum(self):
        fe_problem = _stub_problem()
        deck = {
            "nodal fields": [
                {"name": "scalar_field", "var_type": "scalar"},
                {"name": "vec_field", "var_type": "vector"},
                {"name": "sym_field", "var_type": "sym_tensor"},
                {"name": "ten_field", "var_type": "tensor"},
            ],
        }
        nodal, _ = _resolve_fe_field_specs(
            output_section=deck, fe_problem=fe_problem,  # type: ignore[arg-type]
        )
        self.assertEqual(nodal[0].var_type, VarType.SCALAR)
        self.assertEqual(nodal[1].var_type, VarType.VECTOR)
        self.assertEqual(nodal[2].var_type, VarType.SYM_TENSOR)
        self.assertEqual(nodal[3].var_type, VarType.TENSOR)


def _elastic_parameters(
        kappa: float = 100.0, mu: float = 50.0,
) -> Parameters:
    values = {"elastic": {"kappa": kappa, "mu": mu}}
    active = tree_map(lambda _: True, values)
    transforms = tree_map(lambda _: None, values)
    return Parameters(values, active, transforms)


def _build_elastic_problem():
    mesh = StructuredHexMesh((1.0, 1.0, 1.0), (1, 1, 1))
    layout = GlobalFieldLayout(name="u", finite_element=Q1_HEX)
    dof_map = build_dof_map(
        mesh, [layout], [], components_by_field={"u": 3},
    )
    gr = SmallDispEquilibrium(ndims=3)
    model = Elastic(_elastic_parameters(), def_type=DefType.FULL_3D)
    return build_fe_problem(
        mesh=mesh, dof_map=dof_map, gr=gr,
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.CLOSED_FORM},
    )


class TestWriteFeExodusRoundTrip(unittest.TestCase):
    def test_default_output_zero_state_round_trip(self):
        fe_problem = _build_elastic_problem()
        fe_state = FEState.from_problem(fe_problem, t_init=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_fe_exodus(
                out_dir=out_dir, prefix="", fe_problem=fe_problem,
                fe_state=fe_state, output_section={},
            )
            exo_path = out_dir / "primal.exo"
            self.assertTrue(exo_path.exists())

            results = read_results(
                exo_path,
                nodal_field_specs=[_DISPLACEMENT_FS],
                element_field_specs={"all": [_CAUCHY_FS]},
            )
            n_nodes = fe_problem.mesh.nodes.shape[0]
            n_elems = len(fe_problem.mesh.element_blocks["all"])
            self.assertEqual(results.time.shape, (1,))
            self.assertEqual(
                results.nodal["displacement"].shape,
                (1, n_nodes, 3),
            )
            self.assertTrue(
                np.allclose(results.nodal["displacement"], 0.0),
            )
            self.assertEqual(
                results.element["all"]["cauchy"].shape,
                (1, n_elems, 6),
            )
            self.assertTrue(
                np.allclose(results.element["all"]["cauchy"], 0.0),
            )


if __name__ == "__main__":
    unittest.main()
