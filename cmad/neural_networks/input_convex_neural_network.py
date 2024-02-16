"""
Adapted from the jax tutorial:
https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
"""

import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

from functools import partial

from jax import jit, grad, tree_map
from jax.nn import relu, sigmoid, softplus
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_unflatten, tree_flatten


def forward_with_offset(x, params, input_scale, output_scale):
    xs = input_scale * x
    output = forward(xs, params) - forward(jnp.zeros_like(xs), params)
    return output_scale * output


def forward(x, params):
    activation = softplus
    *x_hidden, x_last = params["x params"]
    *z_hidden, z_last = params["z params"]

    z = activation(x @ x_hidden[0]["weights"] + x_hidden[0]["biases"])
    for x_layer, z_layer in zip(x_hidden[1:], z_hidden):
        z = activation(z @ z_layer["weights"] + x @ x_layer["weights"] \
          + x_layer["biases"])

    return z @ z_last["weights"] + x @ x_last["weights"] \
           + x_last["biases"]


class InputConvexNeuralNetwork():

    def __init__(self, layer_widths: list,
                 input_scale: float = 1., output_scale: float = 1.):
        self._init_params(layer_widths)
        self.evaluate = partial(forward_with_offset, input_scale=input_scale,
                                output_scale=output_scale)

    def _init_params(self, layer_widths: list, seed: int = 22):
        np.random.seed(seed)
        num_x_trainable_layers = len(layer_widths) - 1
        num_z_trainable_layers = len(layer_widths) - 2
        x_layers_idx = np.arange(num_x_trainable_layers, dtype=int)
        z_layers_idx = np.arange(num_z_trainable_layers, dtype=int)
        x_params = [None] * num_x_trainable_layers
        x_input_widths = [layer_widths[0]] * num_x_trainable_layers
        z_params = [None] * num_z_trainable_layers
        x_layer_props = zip(x_layers_idx, x_input_widths, layer_widths[1:])
        z_layer_props = zip(z_layers_idx, layer_widths[1:-1], layer_widths[2:])

        for idx, num_in, num_out in x_layer_props:
            x_params[idx] = dict(weights=np.random.normal(
                size=(num_in, num_out))
                * np.sqrt(2. / num_in), biases=np.ones(num_out))

        # abs initialization for monotonic networks
        for idx, num_in, num_out in z_layer_props:
            z_params[idx] = dict(weights=np.abs(np.random.normal(
                size=(num_in, num_out))
                * np.sqrt(2. / num_in)))

        self.x_params = x_params
        self.z_params = z_params
