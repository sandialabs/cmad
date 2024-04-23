import numpy as np
from numpy import ndarray

from functools import partial

import jax.numpy as jnp
from jax import tree_map, jit
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten, tree_flatten_with_path, tree_reduce


def bounds_transform(value, bounds, transform_from_canonical=True):
    span = 0.5 * (bounds[1] - bounds[0])
    mean = 0.5 * (bounds[0] + bounds[1])
    if transform_from_canonical:
        transformed_value = span * value + mean
    else:
        transformed_value = (value - mean) / span
        if transformed_value < -1.:
            transformed_value = -1.
        if transformed_value > 1.:
            transformed_value = 1.

    return transformed_value


def log_transform(value, ref_value, transform_from_canonical=True):
    if transform_from_canonical:
        transformed_value = ref_value[0] * jnp.exp(value)
    else:
        transformed_value = np.log(value / ref_value[0])

    return transformed_value


def get_size(x):
    if isinstance(x, np.float64) or isinstance(x, float):
        return 1
    elif isinstance(x, ndarray):
        return np.size(x)
    else:
        raise TypeError


def expand_leaf_by_value_size(value, leaf):
    if isinstance(value, np.float64) or isinstance(value, float):
        return [leaf]
    elif isinstance(value, ndarray):
        num_params = np.size(value)
        return [leaf] * num_params


def flatten_by_value_size(values, pytree):
    leaf_values, _ = tree_flatten(values)
    leaves, _ = tree_flatten(pytree, is_leaf=lambda x: x is None)

    expanded_leaves = [expand_leaf_by_value_size(value, leaf)
                       for value, leaf in zip(leaf_values, leaves)]

    flat_pytree = sum(expanded_leaves, [])
    return flat_pytree


def grad_transform(grad, value, transform):
    if transform is None:
        return grad
    if len(transform) == 2:
        return 0.5 * (transform[1] - transform[0]) * grad
    if len(transform) == 1:
        return value * grad


def get_opt_bounds(transform):
    if transform is None or len(transform) == 1:
        return [None, None]
    else:
        return [-1., 1.]


def transform_from_canonical(value, active_flag, transform):
    if active_flag and transform is not None:
        if len(transform) == 2:
            transform_fun = bounds_transform
        elif len(transform) == 1:
            transform_fun = log_transform
        return transform_fun(value, transform)
    else:
        return value


def transform_to_canonical(value, active_flag, transform):
    if active_flag and transform is not None:
        if len(transform) == 2:
            transform_fun = \
                partial(bounds_transform, transform_from_canonical=False)
        elif len(transform) == 1:
            transform_fun = \
                partial(log_transform, transform_from_canonical=False)
        return transform_fun(value, transform)
    else:
        return value


def unpack_elastic_params(params):
    elastic_params = params["elastic"]
    E, nu = elastic_params["E"], elastic_params["nu"]

    return E, nu


class Parameters():
    """ Handle constitutive model parameters with Pytrees """

    def __init__(self, values, active_flags=None, transforms=None):
        self.values = values
        self._active_flags = active_flags
        self._transforms = transforms

        self._flat_values, self.reconstruct_from_flat \
            = ravel_pytree(values)
        self.num_params = len(self._flat_values)

        self._names = []
        flattened, _ = tree_flatten_with_path(values)
        for key_path, value in flattened:
            self._names.append(str(key_path[-1]))

        if active_flags is not None:
            param_sizes = tree_map(lambda x: get_size(x), self.values)
            flat_param_sizes, _ = tree_flatten(param_sizes)
            self.block_shapes = [(x, y)
                for x in flat_param_sizes
                for y in flat_param_sizes]

            self._flat_active_flags = \
                np.array(flatten_by_value_size(values, active_flags)).squeeze()
            self.num_active_params = np.sum(self._flat_active_flags)
            self._active_idx = \
                np.arange(self.num_params)[self._flat_active_flags]
            self.model_active_params_jacobian = \
                partial(self._active_params_jacobian,
                        active_idx=self._active_idx)
            self.qoi_active_params_jacobian = \
                jit(partial(self._active_params_jacobian,
                            num_eqns=1,
                            active_idx=self._active_idx))

            self._expanded_flat_transforms = \
                flatten_by_value_size(values, transforms)
            self._flat_transforms, _ = \
                tree_flatten(self._expanded_flat_transforms,
                             is_leaf=lambda x: x is None)
            self._flat_active_transforms = [self._flat_transforms[ii]
                                            for ii in self._active_idx]
            self.opt_bounds = \
                np.array([get_opt_bounds(transform)
                          for transform in self._flat_active_transforms])
        else:
            assert active_flags == transforms
            self.num_active_params = 0

    def set_active_values(self, values, are_canonical=True):
        if are_canonical:
            self.values = \
                tree_map(lambda v, a, t: transform_from_canonical(v, a, t),
                         values, self._active_flags, self._transforms)
        else:
            self.values = values

    def set_active_values_from_flat(self, flat_active_values,
                                    are_canonical=True):

        updated_flat_values = np.array(self._flat_values)
        updated_flat_values[self._active_idx] = flat_active_values
        updated_values = self.reconstruct_from_flat(updated_flat_values)
        self.set_active_values(updated_values, are_canonical)

    def flat_active_values(self, return_canonical=False):
        flat_values, _ = ravel_pytree(self.values)
        if return_canonical:
            active_flat_values = np.array([
                transform_to_canonical(v, a, t) for v, a, t in
                zip(flat_values, self._flat_active_flags,
                    self._flat_transforms)])[self._active_idx]
        else:
            active_flat_values = \
                np.array(flat_values[self._active_idx])

        return active_flat_values

    def get_active_from_flat(self, pytree):
        flat, _ = ravel_pytree(pytree)
        return flat[self._active_idx]

    def transform_grad(self, grad):
        active_flat_values = self.get_active_from_flat(self.values)
        for ii in range(self.num_active_params):
            value = active_flat_values[ii]
            transform = self._flat_active_transforms[ii]
            grad[ii] = grad_transform(grad[ii], value, transform)

        return grad

    @staticmethod
    def _active_params_jacobian(jacobian, num_eqns, active_idx):
        reshaped_jacobian = \
            tree_map(lambda x: x.reshape(num_eqns, -1), jacobian)
        flat_jac, _ = tree_flatten(reshaped_jacobian)
        array_jac = jnp.hstack(flat_jac)
        return array_jac[:, active_idx]

    def scalar_active_params_jacobian(self, jacobian):
        return self._active_params_jacobian(jacobian, 1, self._active_idx)
