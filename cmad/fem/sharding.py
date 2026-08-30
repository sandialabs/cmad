"""Element axis sharding of the FE assembly across the process's devices.

With more than one JAX device (several GPUs, or CPU devices from
``jax_num_cpu_devices``), the arrays that carry a leading element axis (the
U gather and R scatter indices, the per element geometry cache and the per
element state ``xi``) are placed with a ``NamedSharding`` over a single
axis device mesh; everything else stays on one device and is replicated as
the compiler needs it, so the assembly's per element kernels are
partitioned across the devices without any change to the kernels. With one
device every array is left as it is.

A sharded axis must divide by the device count, so each element block's
element axis is padded to a multiple of it with copies of the block's last
element whose quadrature weights are zero: they cost a few kernel
evaluations and contribute nothing. The padding stays inside the kernel
arrays and the traced state; the mesh, the stored state history and the
outputs see the true element counts.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar

import jax.numpy as jnp
from jax import device_put, devices
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.tree_util import tree_map

if TYPE_CHECKING:
    from cmad.fem.kernel_arrays import FEKernelArrays

T = TypeVar("T")

ELEMENT_AXIS = "elements"
"""Name of the device mesh axis the element axis is sharded across."""

_device_count: int | None = None


def set_device_count(num_devices: int | None) -> None:
    """The number of JAX devices a problem uses when it is built without
    an explicit ``num_devices``. ``None`` means all of them. The ``cmad``
    command sets it once from ``--devices``."""
    global _device_count
    _device_count = num_devices


def build_device_mesh(num_devices: int | None = None) -> Mesh | None:
    """The single axis device mesh the element axis is sharded across.

    The mesh uses the first ``num_devices`` of JAX's devices. If
    ``num_devices`` is ``None``, the count set by :func:`set_device_count`
    is used; if that is unset too, every device JAX sees is used. With
    one device there is nothing to shard and the result is ``None``."""
    all_devices = devices()
    if num_devices is None:
        num_devices = _device_count
    if num_devices is not None:
        if num_devices > len(all_devices):
            raise ValueError(
                f"{num_devices} devices requested, but JAX sees "
                f"{len(all_devices)}",
            )
        all_devices = all_devices[:num_devices]
    if len(all_devices) == 1:
        return None
    return Mesh(all_devices, (ELEMENT_AXIS,))


def padded_count(count: int, device_mesh: Mesh | None) -> int:
    """``count`` rounded up to a multiple of the device count; ``count``
    itself without a mesh."""
    if device_mesh is None:
        return count
    return math.ceil(count / device_mesh.size) * device_mesh.size


def pad_element_leaves(tree: T, n_padded: int, zero: bool = False) -> T:
    """``tree`` with every leaf's leading (element) axis extended to
    ``n_padded`` rows, by repeating the leaf's last row, or with zero rows
    when ``zero`` is set. A leaf that already has ``n_padded`` rows is
    returned as it is."""
    def pad(leaf):
        n = leaf.shape[0]
        if n == n_padded:
            return leaf
        tail_shape = (n_padded - n, *leaf.shape[1:])
        if zero:
            tail = jnp.zeros(tail_shape, dtype=leaf.dtype)
        else:
            tail = jnp.broadcast_to(leaf[-1:], tail_shape)
        return jnp.concatenate([jnp.asarray(leaf), tail], axis=0)

    return tree_map(pad, tree)


def strip_element_padding(
        xi_by_block: Mapping[str, T], n_elems_by_block: Mapping[str, int],
) -> dict[str, T]:
    """Each block's state array cut back to the block's true element
    count."""
    return {
        block: xi[:n_elems_by_block[block]]  # type: ignore[index]
        for block, xi in xi_by_block.items()
    }


def place_element_leaves(tree: T, device_mesh: Mesh | None) -> T:
    """``tree`` (JAX array leaves) padded along its leading (element) axis
    to a multiple of the device count, by repeating each leaf's last row,
    and sharded across ``device_mesh``; unchanged when there is no
    mesh."""
    if device_mesh is None:
        return tree
    sharding = NamedSharding(device_mesh, PartitionSpec(ELEMENT_AXIS))

    def place(leaf):
        padded = pad_element_leaves(
            leaf, padded_count(leaf.shape[0], device_mesh),
        )
        return device_put(padded, sharding)

    return tree_map(place, tree)


def shard_kernel_arrays(
        arrays: FEKernelArrays, device_mesh: Mesh,
) -> FEKernelArrays:
    """``arrays`` with its element axis leaves, the U gather and R scatter
    indices and the per element geometry of every block (already padded to
    a multiple of the device count by
    :func:`cmad.fem.kernel_arrays.build_fe_kernel_arrays`), sharded across
    ``device_mesh``; the other fields are left as they are."""
    return replace(
        arrays,
        u_gather_eq_by_block=place_element_leaves(
            arrays.u_gather_eq_by_block, device_mesh,
        ),
        r_scatter_eq_by_block=place_element_leaves(
            arrays.r_scatter_eq_by_block, device_mesh,
        ),
        geometry_cache={
            block: replace(
                cache,
                per_elem=place_element_leaves(cache.per_elem, device_mesh),
            )
            for block, cache in arrays.geometry_cache.items()
        },
    )
