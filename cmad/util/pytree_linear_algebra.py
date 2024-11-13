import jax.numpy as jnp

from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves


def make_linop(jnp_op, tree_in, tree_out):
    def linop(A, b):
        br = ravel_pytree(b)[0]
        len_in = tree_in.args[0].num_leaves
        lenA = len(tree_leaves(A))
        len_out = lenA // len_in
        rows = ravel_pytree(A)[0].shape[0] // br.shape[0]
        leavesA = [jnp.reshape(s, (s.shape[0], -1)) for s in tree_leaves(A)]
        Ar = jnp.concatenate(
            [jnp.concatenate(leavesA[j:j + len_in], axis=1)
            for j in range(0, lenA, len_in)], axis=0
        )

        return tree_out(jnp_op(Ar, br))

    return linop


def make_op(jnp_op, tree):
    def op(*args):
        return tree(jnp_op(*[ravel_pytree(arg)[0] for arg in args]))

    return op
