"""Transient heat transfer through the time loop: first order in time on
a manufactured solution with no spatial error, and the slab against its
Fourier series."""
import unittest
from collections.abc import Callable, Sequence

import numpy as np
from jax import numpy as jnp
from numpy.typing import NDArray

from cmad.fem.bcs import DirichletBC
from cmad.fem.dof import GlobalFieldLayout, build_dof_map, dof_physical_coords
from cmad.fem.driver import fe_quasistatic_drive
from cmad.fem.fe_problem import FEProblem, build_fe_problem
from cmad.fem.finite_element import Q1_HEX
from cmad.fem.mesh import StructuredHexMesh
from cmad.global_residuals.heat_transfer import HeatTransfer
from cmad.models.conduction import Conduction
from cmad.typing import JaxArray
from tests.fem._mms_helpers import l2_h1_errors, make_conduction_parameters

_ALL_FACES = [
    "xmin_sides", "xmax_sides",
    "ymin_sides", "ymax_sides",
    "zmin_sides", "zmax_sides",
]


def _bar_problem(
        divisions: tuple[int, int, int],
        bcs: Sequence[DirichletBC],
        source_fn: Callable[
            [NDArray[np.floating] | JaxArray, float | JaxArray],
            NDArray[np.floating] | JaxArray,
        ] | None = None,
) -> FEProblem:
    """A unit cube of hexes with ``k = rho = c = 1`` and the T field."""
    mesh = StructuredHexMesh(lengths=(1.0, 1.0, 1.0), divisions=divisions)
    layout = GlobalFieldLayout(name="T", finite_element=Q1_HEX)
    dof_map = build_dof_map(
        mesh, [layout], list(bcs), components_by_field={"T": 1},
    )
    return build_fe_problem(
        mesh=mesh,
        dof_map=dof_map,
        gr=HeatTransfer(ndims=3),
        models_by_block={
            "all": Conduction(make_conduction_parameters(1.0, 1.0, 1.0)),
        },
        forcing_fns_by_block_idx=(
            {0: source_fn} if source_fn is not None else None
        ),
    )


def _initial_condition(
        fe_problem: FEProblem,
        T_of_x: Callable[[NDArray[np.floating]], NDArray[np.floating]],
) -> NDArray[np.floating]:
    coords, eq = dof_physical_coords(
        fe_problem.mesh, fe_problem.dof_map, "T",
    )
    U_init = np.zeros(fe_problem.dof_map.num_total_dofs)
    U_init[eq[:, 0]] = T_of_x(coords)
    return U_init


def time_convergence_errors(num_steps: Sequence[int]) -> list[float]:
    """L2 errors at ``t = 1`` of ``T = x exp(-t)`` for each step count.

    Linear in space, so P1 carries no spatial error and the source
    ``-x exp(-t)`` and the boundary values leave only the backward Euler
    error, first order in the step.
    """
    def boundary(coords, t):
        return (coords[:, 0] * jnp.exp(-t))[:, None]

    def source(coords, t):
        return jnp.asarray([-coords[0] * jnp.exp(-t)])

    fe_problem = _bar_problem(
        (4, 2, 2),
        [DirichletBC(_ALL_FACES, "T", dofs=(0,), values=boundary)],
        source,
    )
    U_init = _initial_condition(fe_problem, lambda coords: coords[:, 0])
    errors: list[float] = []
    for n in num_steps:
        state, _, status = fe_quasistatic_drive(
            fe_problem, np.linspace(0.0, 1.0, n + 1).tolist(), U_init=U_init,
        )
        assert status.converged
        L2, _ = l2_h1_errors(
            fe_problem, state.U_at(-1),
            lambda coords: np.array([coords[0] * np.exp(-1.0)]),
            lambda coords: np.array([[np.exp(-1.0), 0.0, 0.0]]),
        )
        errors.append(L2)
    return errors


def slab_series(x: NDArray[np.floating], t: float) -> NDArray[np.floating]:
    """``T`` of the unit slab at zero on both ends from ``T = 1``, with
    unit diffusivity, summed until the next term is below ``1e-8``."""
    T = np.zeros_like(x)
    n = 1
    while True:
        term = 4.0 / (n * np.pi) * np.exp(-(n * np.pi) ** 2 * t)
        if term < 1e-8:
            break
        T = T + term * np.sin(n * np.pi * x)
        n += 2
    return T


def slab_max_error(
        divisions_x: int, num_steps: int, t_final: float,
) -> float:
    """Largest nodal error against :func:`slab_series` at ``t_final``."""
    fe_problem = _bar_problem(
        (divisions_x, 1, 1),
        [DirichletBC(["xmin_sides", "xmax_sides"], "T", dofs=(0,))],
    )
    U_init = _initial_condition(fe_problem, lambda coords: np.ones(len(coords)))
    state, _, status = fe_quasistatic_drive(
        fe_problem, np.linspace(0.0, t_final, num_steps + 1).tolist(),
        U_init=U_init,
    )
    assert status.converged
    coords, eq = dof_physical_coords(
        fe_problem.mesh, fe_problem.dof_map, "T",
    )
    T_h = np.asarray(state.U_at(-1))[eq[:, 0]]
    return float(np.max(np.abs(T_h - slab_series(coords[:, 0], t_final))))


class TestHeatTransient(unittest.TestCase):
    def test_backward_euler_is_first_order(self) -> None:
        errors = time_convergence_errors((5, 10, 20))
        rates = [
            float(np.log2(errors[i] / errors[i + 1])) for i in range(2)
        ]
        for r in rates:
            self.assertGreaterEqual(r, 0.9, f"rates {rates}, errors {errors}")

    def test_slab_converges_to_the_series(self) -> None:
        # Refining the mesh two times and the step four times cuts the
        # discretization error four times (second order in the mesh
        # size, first in the step), so the ratio says the computed
        # temperature converges to the series (measured 2.29e-3 and
        # 5.74e-4). The fine error is under a tenth of a percent of the
        # temperature scale.
        coarse = slab_max_error(32, 100, 0.1)
        fine = slab_max_error(64, 400, 0.1)
        self.assertGreaterEqual(
            coarse / fine, 3.0, f"errors {coarse}, {fine}",
        )
        self.assertLess(fine, 1.0e-3, f"fine error {fine}")


if __name__ == "__main__":
    unittest.main()
