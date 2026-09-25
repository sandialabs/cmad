"""Base of the FE QoIs that combine the sub-QoIs of an input file
``terms`` list, each term carrying its own ``weight``."""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp

from cmad.io.registry import resolve_qoi
from cmad.qois.fe_qoi import FEQoI, StepContribution
from cmad.typing import JaxArray, Params

if TYPE_CHECKING:
    from cmad.fem.fe_problem import FEProblem
    from cmad.fem.kernel_arrays import FEKernelArrays
    from cmad.models.global_fields import StepTime


class FETermSum(FEQoI):
    """A list of sub-QoIs, each accumulated over the time loop as its own
    value. The weights are applied in :meth:`combine`."""

    problem_type: ClassVar[str] = "fe"

    def __init__(self, terms: Sequence[FEQoI]) -> None:
        super().__init__()
        self._terms = list(terms)
        self._weights = jnp.asarray(
            [term.weight for term in terms], dtype=jnp.float64,
        )

    @classmethod
    def from_deck(
            cls,
            qoi_section: dict[str, Any],
            fe_problem: FEProblem,
            t_schedule: Sequence[float],
    ) -> FETermSum:
        terms: list[FEQoI] = []
        for term in qoi_section["terms"]:
            sub_cls = resolve_qoi(term["name"])
            if sub_cls.problem_type != "fe":
                raise ValueError(
                    f"{qoi_section['name']} term '{term['name']}' is "
                    f"registered for problem_type='{sub_cls.problem_type}', "
                    "not 'fe'"
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
                step_time: StepTime,
        ) -> JaxArray:
            return jnp.stack([
                closure(U, U_prev, xi, xi_prev, step_time)
                for closure in closures
            ])

        return _closure

    def accumulated_qoi_names(self) -> list[str]:
        """Each term's input file name, numbered from the second
        occurrence when a name repeats."""
        names: list[str] = []
        for term in self._terms:
            (name,) = term.accumulated_qoi_names()
            count = sum(1 for n in names if n == name or n.startswith(f"{name}_"))
            names.append(name if count == 0 else f"{name}_{count + 1}")
        return names

    def data_mean_squares(self) -> dict[str, float]:
        return {
            name: mean_square
            for name, term in zip(
                self.accumulated_qoi_names(), self._terms, strict=True,
            )
            for mean_square in term.data_mean_squares().values()
        }

    @abstractmethod
    def combine(self, accumulated_qois: JaxArray) -> JaxArray:
        """The QoI value from the accumulated terms."""
        ...
