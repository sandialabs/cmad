"""Composite FE QoI: a sum of sub-QoIs, each carrying its own weight."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp

from cmad.io.registry import register_qoi, resolve_qoi
from cmad.qois.fe_qoi import FEQoI, StepContribution
from cmad.typing import JaxArray, Params

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem
    from cmad.fem.kernel_arrays import FEKernelArrays


@register_qoi("fe_weighted_sum")
class FEWeightedSum(FEQoI):
    """Sum of FE sub-QoIs, each carrying its own deck ``weight``.

    Each entry in the deck ``terms`` list builds a sub-QoI via its own
    ``from_deck``; this QoI's per-step contribution is the sum of theirs.
    Relative weighting between heterogeneous terms (e.g. a displacement
    match and a load match) is each term's own ``weight``, so the composite
    holds no separate weight.
    """

    problem_type: ClassVar[str] = "fe"

    def __init__(self, terms: Sequence[FEQoI]) -> None:
        self._terms = list(terms)

    @classmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
    ) -> FEWeightedSum:
        terms: list[FEQoI] = []
        for term in qoi_section["terms"]:
            sub_cls = resolve_qoi(term["name"])
            if sub_cls.problem_type != "fe":
                raise ValueError(
                    f"fe_weighted_sum term '{term['name']}' is registered "
                    f"for problem_type='{sub_cls.problem_type}', not 'fe'"
                )
            assert issubclass(sub_cls, FEQoI)
            terms.append(sub_cls.from_deck(term, fe_problem, t_schedule))
        return cls(terms)

    def step_contribution(
            self,
            params_by_block: Mapping[str, Params],
            fe_arrays: FEKernelArrays,
    ) -> StepContribution:
        closures = [
            term.step_contribution(params_by_block, fe_arrays)
            for term in self._terms
        ]

        def _closure(
                U: JaxArray,
                U_prev: JaxArray,
                xi: Mapping[str, JaxArray],
                xi_prev: Mapping[str, JaxArray],
                t: JaxArray,
                t_prev: JaxArray,
        ) -> JaxArray:
            total = jnp.zeros(())
            for closure in closures:
                total = total + closure(
                    U, U_prev, xi, xi_prev, t, t_prev,
                )
            return total

        return _closure
