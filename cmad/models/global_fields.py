"""Structured global-state context for Model and QoI evaluation."""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
from numpy.typing import NDArray

from cmad.typing import JaxArray, Scalar


@register_pytree_node_class
@dataclass(frozen=True)
class GlobalFieldsAtPoint:
    """Interpolated global fields + their gradients at an evaluation point."""
    fields: dict[str, JaxArray]
    grad_fields: dict[str, JaxArray]

    def tree_flatten(
            self,
    ) -> tuple[tuple[dict[str, JaxArray], dict[str, JaxArray]], None]:
        return (self.fields, self.grad_fields), None

    @classmethod
    def tree_unflatten(
            cls,
            aux_data: None,
            children: tuple[dict[str, JaxArray], dict[str, JaxArray]],
    ) -> "GlobalFieldsAtPoint":
        fields, grad_fields = children
        return cls(fields=fields, grad_fields=grad_fields)


@register_pytree_node_class
@dataclass(frozen=True)
class StepTime:
    """The current and previous times for one solve step.

    ``dt`` is their difference. Passed as a single argument to the Model
    residual, the global residual, and the forcing / Neumann evaluators.
    ``t`` and ``t_prev`` are pytree children (traced), so a varying step
    size does not retrigger compilation; time is never a differentiation
    target, so the Model's fixed argnum derivatives and the global
    tangent never differentiate it.
    """
    t: Scalar
    t_prev: Scalar

    @property
    def dt(self) -> Scalar:
        return self.t - self.t_prev

    def tree_flatten(self) -> tuple[tuple[Scalar, Scalar], None]:
        return (self.t, self.t_prev), None

    @classmethod
    def tree_unflatten(
            cls, aux_data: None, children: tuple[Scalar, Scalar],
    ) -> "StepTime":
        t, t_prev = children
        return cls(t=t, t_prev=t_prev)


def mp_U_from_F(F: NDArray[np.floating] | JaxArray) -> GlobalFieldsAtPoint:
    """Build the MP-level U from a prescribed F: grad_fields['u'] = F - I."""
    F_jax = jnp.asarray(F)
    ndims = F_jax.shape[0]
    return GlobalFieldsAtPoint(
        fields={"u": jnp.zeros(ndims)},
        grad_fields={"u": F_jax - jnp.eye(ndims)},
    )
