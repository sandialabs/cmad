"""Element axis sharding of the FE assembly across the process's devices.

With more than one JAX device (several GPUs, or CPU devices from
``jax_num_cpu_devices``), the arrays that carry a leading element axis (the
U gather and R scatter indices, the per element geometry cache and the per
element state ``xi``) are placed with a ``NamedSharding`` over a single
axis device mesh; everything else stays on one device and is replicated as
the compiler needs it, so the assembly's per element kernels are
partitioned across the devices without any change to the kernels. With one
device every array is left as it is.

The shard count must divide every element block's element count, so it is
the largest common divisor of the block counts that fits the device count;
a mesh whose block counts have no such divisor runs on one device.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar

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


def element_shard_count(
        n_elems_by_block: Mapping[str, int], n_devices: int,
) -> int:
    """The largest common divisor of the block element counts that is at
    most ``n_devices``."""
    common = math.gcd(*n_elems_by_block.values())
    return max(d for d in range(1, n_devices + 1) if common % d == 0)


def build_device_mesh(
        n_elems_by_block: Mapping[str, int], num_devices: int | None = None,
) -> Mesh | None:
    """The single axis device mesh the element axis is sharded across, or
    ``None`` when the assembly stays on one device: there is one JAX
    device, or the block element counts have no common divisor that fits
    the device count (a warning says so). The mesh is built from the
    first ``num_devices`` of JAX's devices; when ``num_devices`` is
    ``None`` the count set by :func:`set_device_count` applies, and with
    neither every device is used."""
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
    count = element_shard_count(n_elems_by_block, len(all_devices))
    if count == 1:
        warnings.warn(
            f"{len(all_devices)} devices, but the element counts "
            f"{dict(n_elems_by_block)} have no common divisor between 2 "
            f"and {len(all_devices)}; the assembly runs on one device",
            stacklevel=2,
        )
        return None
    return Mesh(all_devices[:count], (ELEMENT_AXIS,))


def place_element_leaves(tree: T, device_mesh: Mesh | None) -> T:
    """``tree`` (JAX array leaves) sharded along its leading (element) axis
    across ``device_mesh``; unchanged when there is no mesh."""
    if device_mesh is None:
        return tree
    sharding = NamedSharding(device_mesh, PartitionSpec(ELEMENT_AXIS))
    return tree_map(lambda leaf: device_put(leaf, sharding), tree)


def shard_kernel_arrays(
        arrays: FEKernelArrays, device_mesh: Mesh,
) -> FEKernelArrays:
    """``arrays`` with its element axis leaves, the U gather and R scatter
    indices and the per element geometry of every block, sharded across
    ``device_mesh``; the other fields are the same objects."""
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
