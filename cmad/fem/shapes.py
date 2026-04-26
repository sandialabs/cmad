"""Shape-function values and gradients at a finite-element integration point."""
from dataclasses import dataclass

from jax.tree_util import register_pytree_node_class

from cmad.typing import JaxArray


@register_pytree_node_class
@dataclass(frozen=True)
class ShapeFunctionsAtIP:
    """One basis's shape-function values and gradients at an IP.

    Aggregated per-residual-block at GR composed-helper boundaries as
    ``list[ShapeFunctionsAtIP]`` (length 1 for single-field problems,
    longer for Taylor-Hood and other mixed-basis formulations).
    """
    N: JaxArray         # shape (num_basis_fns,)
    grad_N: JaxArray    # shape (num_basis_fns, ndims)

    def tree_flatten(
            self,
    ) -> tuple[tuple[JaxArray, JaxArray], None]:
        return (self.N, self.grad_N), None

    @classmethod
    def tree_unflatten(
            cls,
            aux_data: None,
            children: tuple[JaxArray, JaxArray],
    ) -> "ShapeFunctionsAtIP":
        N, grad_N = children
        return cls(N=N, grad_N=grad_N)
