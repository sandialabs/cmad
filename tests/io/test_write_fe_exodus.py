"""Tests for :func:`cmad.io.writers.resolve_fe_output_plan` and
:func:`cmad.io.writers.write_fe_exodus`.

Plan-resolution unit tests use stub GR / Model / FEProblem objects (only
the catalog methods are exercised; the resolved evaluators are stored,
not called). The round-trip smoke test builds a real 1-block hex-cube FE
problem with zero displacement and confirms the Exodus file reads back
with the default output surface (nodal ``u`` + element ``cauchy``)
populated.
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
from cmad.io.writers import resolve_fe_output_plan, write_fe_exodus
from cmad.models.deformation_types import DefType
from cmad.models.elastic import Elastic
from cmad.models.var_types import VarType
from cmad.parameters.parameters import Parameters

_U_FS = FieldSpec("u", VarType.VECTOR)
_CAUCHY_FS = FieldSpec("cauchy", VarType.SYM_TENSOR)


class _StubGR:
    def __init__(self, primary: list[tuple[str, VarType]]) -> None:
        self._primary = primary

    def primary_output_fields(self) -> list[tuple[str, VarType]]:
        return self._primary


class _StubModel:
    def __init__(
            self,
            state_fields: list[tuple[str, VarType]],
            derived_names: list[str],
    ) -> None:
        self._state = state_fields
        self._derived = derived_names

    def state_output_fields(self) -> list[tuple[str, VarType]]:
        return self._state

    def derived_output_field_names(self) -> list[str]:
        return self._derived


class _StubMesh:
    def __init__(self, blocks: tuple[str, ...]) -> None:
        self.element_blocks = {b: np.array([0]) for b in blocks}


class _StubFEProblem:
    def __init__(
            self,
            gr: _StubGR,
            blocks: tuple[str, ...],
            models_by_block: dict,
            modes_by_block: dict,
    ) -> None:
        self.gr = gr
        self.mesh = _StubMesh(blocks)
        self.models_by_block = models_by_block
        self.modes_by_block = modes_by_block


def _coupled_plasticity_problem() -> _StubFEProblem:
    """COUPLED block: state [plastic strain, alpha] + derived [cauchy]."""
    model = _StubModel(
        state_fields=[
            ("plastic strain", VarType.SYM_TENSOR),
            ("alpha", VarType.SCALAR),
        ],
        derived_names=["cauchy"],
    )
    return _StubFEProblem(
        gr=_StubGR([("u", VarType.VECTOR)]),
        blocks=("all",),
        models_by_block={"all": model},
        modes_by_block={"all": GlobalResidualMode.COUPLED},
    )


class TestResolveFEOutputPlan(unittest.TestCase):
    @staticmethod
    def _names(fields):
        return [f.name for f in fields]

    def test_blank_groups_default_to_full_advertised_set(self):
        fe_problem = _coupled_plasticity_problem()
        plan = resolve_fe_output_plan({}, fe_problem)  # type: ignore[arg-type]
        self.assertEqual(self._names(plan.nodal), ["u"])
        self.assertEqual(plan.nodal[0].var_type, VarType.VECTOR)
        self.assertEqual(
            self._names(plan.element_by_block["all"]),
            ["plastic strain", "alpha", "cauchy"],
        )

    def test_closed_form_block_skips_state_defaults_to_derived(self):
        # CLOSED_FORM never solves xi, so state vars are not selectable;
        # the blank-group default is the derived set only.
        model = _StubModel(
            state_fields=[("cauchy", VarType.SYM_TENSOR)],
            derived_names=["cauchy"],
        )
        fe_problem = _StubFEProblem(
            gr=_StubGR([("u", VarType.VECTOR)]),
            blocks=("all",),
            models_by_block={"all": model},
            modes_by_block={"all": GlobalResidualMode.CLOSED_FORM},
        )
        plan = resolve_fe_output_plan({}, fe_problem)  # type: ignore[arg-type]
        self.assertEqual(
            self._names(plan.element_by_block["all"]), ["cauchy"],
        )

    def test_explicit_selection_subsets_the_catalog(self):
        fe_problem = _coupled_plasticity_problem()
        plan = resolve_fe_output_plan(
            {
                "global residual": ["u"],
                "local residual": {"all": ["cauchy"]},
            },
            fe_problem,  # type: ignore[arg-type]
        )
        self.assertEqual(self._names(plan.nodal), ["u"])
        self.assertEqual(
            self._names(plan.element_by_block["all"]), ["cauchy"],
        )

    def test_unknown_nodal_field_raises(self):
        fe_problem = _coupled_plasticity_problem()
        with self.assertRaises(ValueError) as ctx:
            resolve_fe_output_plan(
                {"global residual": ["nope"]},
                fe_problem,  # type: ignore[arg-type]
            )
        self.assertIn("nope", str(ctx.exception))

    def test_unknown_element_field_raises(self):
        fe_problem = _coupled_plasticity_problem()
        with self.assertRaises(ValueError) as ctx:
            resolve_fe_output_plan(
                {"local residual": {"all": ["nope"]}},
                fe_problem,  # type: ignore[arg-type]
            )
        self.assertIn("nope", str(ctx.exception))

    def test_unknown_block_raises(self):
        fe_problem = _coupled_plasticity_problem()
        with self.assertRaises(ValueError) as ctx:
            resolve_fe_output_plan(
                {"local residual": {"ghost": ["cauchy"]}},
                fe_problem,  # type: ignore[arg-type]
            )
        self.assertIn("ghost", str(ctx.exception))

    def test_state_derived_name_collision_raises(self):
        model = _StubModel(
            state_fields=[("cauchy", VarType.SYM_TENSOR)],
            derived_names=["cauchy"],
        )
        fe_problem = _StubFEProblem(
            gr=_StubGR([("u", VarType.VECTOR)]),
            blocks=("all",),
            models_by_block={"all": model},
            modes_by_block={"all": GlobalResidualMode.COUPLED},
        )
        with self.assertRaises(ValueError) as ctx:
            resolve_fe_output_plan({}, fe_problem)  # type: ignore[arg-type]
        self.assertIn("cauchy", str(ctx.exception))


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
        output_plan = resolve_fe_output_plan({}, fe_problem)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_fe_exodus(
                out_dir=out_dir, prefix="", fe_problem=fe_problem,
                fe_state=fe_state, output_plan=output_plan,
                exodus_filename="primal.exo",
            )
            exo_path = out_dir / "primal.exo"
            self.assertTrue(exo_path.exists())

            results = read_results(
                exo_path,
                nodal_field_specs=[_U_FS],
                element_field_specs={"all": [_CAUCHY_FS]},
            )
            n_nodes = fe_problem.mesh.nodes.shape[0]
            n_elems = len(fe_problem.mesh.element_blocks["all"])
            self.assertEqual(results.time.shape, (1,))
            self.assertEqual(
                results.nodal["u"].shape,
                (1, n_nodes, 3),
            )
            self.assertTrue(
                np.allclose(results.nodal["u"], 0.0),
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
