"""
Adapted from the jax tutorial:
https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
"""

import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

from functools import partial

from jax import jit, grad, tree_map
from jax.nn import relu, sigmoid
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_unflatten, tree_flatten


def forward_with_offset(x, params, input_scale, output_scale):
    xs = input_scale * x
    output = forward(xs, params) - forward(jnp.zeros_like(xs), params)
    return output_scale * output


def forward(x, params):
    *hidden, last = params
    for layer in hidden:
        x = sigmoid(x @ layer["weights"] + layer["biases"])
    return x @ last["weights"] + last["biases"]


class SimpleNeuralNetwork():

    def __init__(self, layer_widths: list,
                 input_scale: float = 1., output_scale: float = 1.):
        self._init_params(layer_widths)
        self.evaluate = partial(forward_with_offset, input_scale=input_scale,
                                output_scale=output_scale)

    def _init_params(self, layer_widths: list, seed: int = 22):
        np.random.seed(seed)
        num_trainable_layers = len(layer_widths) - 1
        layers_idx = np.arange(num_trainable_layers, dtype=int)
        params = [None] * num_trainable_layers
        layer_props = zip(layers_idx, layer_widths[:-1], layer_widths[1:])

        # abs initialization for monotonic networks
        for idx, num_in, num_out in layer_props:
            params[idx] = dict(weights=np.abs(np.random.normal(size=(num_in, num_out))
                * np.sqrt(2. / num_in)), biases=np.ones(num_out))

        self.params = params
