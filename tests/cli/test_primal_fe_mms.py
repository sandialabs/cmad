"""End-to-end MMS convergence regression for ``cmad primal --fe``.

Deck-driven port of ``tests/fem/test_mms_cube_3d.py``: the body-force
component expressions are derived symbolically from the manufactured
solution ``u = sin(pi*x) * sin(pi*y) * sin(pi*z)`` (all 3 components),
stringified, and dropped into the deck's ``body forces.expression``
slot. The CLI's sympy parser re-parses each component string into a
JAX callable and feeds them through the assembly. After the primal
completes and writes Exodus, displacement is read back and reduced
to L2 / H1 errors against the analytic solution; rates are checked
on consecutive-N ratios.

Hex sweep ``N in {4, 8, 16}`` (two ratios) and tet sweep
``N in {4, 8}`` via ``hex_to_tet_split`` (one ratio) match the
direct-form regression's coverage and budget.
"""
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sympy import Matrix, eye, pi, simplify, sin, symbols

from cmad.cli.common import build_fe_problem_from_deck
from cmad.cli.main import main as cmad_main
from cmad.fem.mesh import Mesh, StructuredHexMesh, hex_to_tet_split
from cmad.io.exodus import ExodusWriter, read_results
from cmad.io.results import FieldSpec
from cmad.models.var_types import VarType
from tests.fem._mms_helpers import build_mms_callables, l2_h1_errors

_KAPPA = 100.0
_MU = 50.0


def _derive_body_force_strs(
        u_sym: Matrix,
        coord_syms: tuple[Any, ...],
        kappa: float,
        mu: float,
) -> tuple[str, ...]:
    """Re-derive ``b = -div(sigma(u_sym))`` and stringify each component.

    Mirrors the symbolic derivation in
    :func:`tests.fem._mms_helpers.build_mms_callables` but exposes the
    per-component strings instead of a lambdified callable, so the deck
    builder can drop them into ``body forces.expression``.
    """
    n = len(coord_syms)
    grad_u_sym = u_sym.jacobian(list(coord_syms))
    eps_sym = (grad_u_sym + grad_u_sym.T) / 2
    tr_eps = eps_sym.trace()
    dev_eps_sym = eps_sym - (tr_eps / n) * eye(n)
    sigma_sym = kappa * tr_eps * eye(n) + 2 * mu * dev_eps_sym
    b_sym = simplify(Matrix([
        -sum(sigma_sym[i, j].diff(coord_syms[j]) for j in range(n))
        for i in range(n)
    ]))
    return tuple(str(b_sym[i, 0]) for i in range(n))


def _make_mms_deck(
        mesh_filename: str,
        body_force_strs: tuple[str, ...],
) -> dict[str, Any]:
    """Build the deck dict: 6-face homogeneous DBC + MMS body force."""
    sidesets = (
        "xmin_sides", "xmax_sides",
        "ymin_sides", "ymax_sides",
        "zmin_sides", "zmax_sides",
    )
    dbc_entries: dict[str, list[Any]] = {
        f"{s}_d{d}": ["displacement", d, s, 0.0]
        for s in sidesets
        for d in (0, 1, 2)
    }
    return {
        "problem": {"type": "fe"},
        "discretization": {
            "mesh file": mesh_filename,
            "num steps": 1,
            "step size": 1.0,
        },
        "residuals": {
            "global residual": {"type": "small_disp_equilibrium"},
            "local residual": {
                "type": "elastic",
                "def_type": "full_3d",
                "materials": {
                    "all": {"elastic": {"kappa": _KAPPA, "mu": _MU}},
                },
            },
        },
        "dirichlet bcs": {"expression": dbc_entries},
        "body forces": {
            "expression": {
                "mms_body_force": ["displacement", *body_force_strs],
            },
        },
        "output": {"path": "out", "format": "exodus"},
    }


class TestPrimalFeMmsCube3D(unittest.TestCase):
    """Manufactured-solution rate check on the deck-driven FE primal path."""

    body_force_strs: tuple[str, ...]
    u_exact: Any
    grad_u_exact: Any

    @classmethod
    def setUpClass(cls) -> None:
        x, y, z = symbols("x y z", real=True)
        profile = sin(pi * x) * sin(pi * y) * sin(pi * z)
        u_sym = Matrix([profile, profile, profile])
        cls.body_force_strs = _derive_body_force_strs(
            u_sym, (x, y, z), _KAPPA, _MU,
        )
        _, cls.u_exact, cls.grad_u_exact, _ = build_mms_callables(
            u_sym, (x, y, z), _KAPPA, _MU,
        )

    def _solve_one_N_via_cli(
            self, mesh: Mesh, work_dir: Path,
    ) -> tuple[float, float]:
        """Drive a single CLI primal solve and return ``(L2, H1)``."""
        mesh_path = work_dir / "mesh.exo"
        with ExodusWriter(str(mesh_path), mesh):
            pass

        deck = _make_mms_deck(
            mesh_filename="mesh.exo",
            body_force_strs=type(self).body_force_strs,
        )
        deck_path = work_dir / "deck.yaml"
        deck_path.write_text(yaml.safe_dump(deck, sort_keys=False))

        self.assertEqual(cmad_main(["primal", str(deck_path)]), 0)

        results = read_results(
            work_dir / "out" / "primal.exo",
            nodal_field_specs=[
                FieldSpec("displacement", VarType.VECTOR),
            ],
        )
        # SmallDispEquilibrium: single field block, ndofs=3, block_offset=0.
        # Node-major dof-fastest reshape matches the dof_map's eq layout
        # since the CLI builder reads the mesh in Exodus node order.
        U_solved = np.asarray(
            results.nodal["displacement"][-1], dtype=np.float64,
        ).reshape(-1)

        bundle = build_fe_problem_from_deck(deck_path, "primal")
        return l2_h1_errors(
            bundle.fe_problem, U_solved,
            type(self).u_exact, type(self).grad_u_exact,
        )

    def _sweep_and_check_rates(
            self, build_mesh: Any, Ns: tuple[int, ...],
            l2_floor: float, h1_floor: float,
    ) -> None:
        L2_errs: list[float] = []
        H1_errs: list[float] = []
        with tempfile.TemporaryDirectory() as outer:
            outer_path = Path(outer)
            for N in Ns:
                sub = outer_path / f"N_{N}"
                sub.mkdir()
                L2, H1 = self._solve_one_N_via_cli(build_mesh(N), sub)
                L2_errs.append(L2)
                H1_errs.append(H1)

        L2_rates = [
            float(np.log2(L2_errs[i] / L2_errs[i + 1]))
            for i in range(len(Ns) - 1)
        ]
        H1_rates = [
            float(np.log2(H1_errs[i] / H1_errs[i + 1]))
            for i in range(len(Ns) - 1)
        ]
        for r in L2_rates:
            self.assertGreaterEqual(r, l2_floor, f"L2 rates {L2_rates}")
        for r in H1_rates:
            self.assertGreaterEqual(r, h1_floor, f"H1 rates {H1_rates}")

    def test_hex_convergence_rates(self) -> None:
        self._sweep_and_check_rates(
            build_mesh=lambda N: StructuredHexMesh(
                lengths=(1.0, 1.0, 1.0), divisions=(N, N, N),
            ),
            Ns=(4, 8, 16),
            l2_floor=1.9,
            h1_floor=0.9,
        )

    def test_tet_convergence_rates(self) -> None:
        self._sweep_and_check_rates(
            build_mesh=lambda N: hex_to_tet_split(StructuredHexMesh(
                lengths=(1.0, 1.0, 1.0), divisions=(N, N, N),
            )),
            Ns=(4, 8),
            l2_floor=1.9,
            h1_floor=0.9,
        )


if __name__ == "__main__":
    unittest.main()
