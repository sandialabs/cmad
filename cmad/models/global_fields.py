"""Structured global-state context for Model and QoI evaluation."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.typing import JaxArray


@jax.tree_util.register_pytree_node_class
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


def mp_U_from_F(F: NDArray[np.floating] | JaxArray) -> GlobalFieldsAtPoint:
    """Build the MP-level U from a prescribed F: grad_fields['u'] = F - I."""
    F_jax = jnp.asarray(F)
    ndims = F_jax.shape[0]
    return GlobalFieldsAtPoint(
        fields={"u": jnp.zeros(ndims)},
        grad_fields={"u": F_jax - jnp.eye(ndims)},
    )
