"""
State variable types and sizes
"""
from enum import IntEnum

import jax.numpy as jnp

from cmad.models.deformation_types import DefType
from cmad.typing import JaxArray, StateBlock


class VarType(IntEnum):
    SCALAR = 0
    VECTOR = 1
    SYM_TENSOR = 2
    TENSOR = 3

# vector <-> var_type conversions act on jax arrays


def get_num_eqs(var_type: int, ndims: int) -> int:
    if var_type == VarType.SCALAR:
        return 1
    elif var_type == VarType.VECTOR:
        return ndims
    elif var_type == VarType.SYM_TENSOR:
        return (ndims + 1) * ndims // 2
    elif var_type == VarType.TENSOR:
        return ndims**2
    raise ValueError(f"Unknown var_type: {var_type}")


def get_scalar(var: StateBlock) -> StateBlock:
    assert len(var) == 1
    return var


def get_vector(var: StateBlock, ndims: int) -> StateBlock:
    assert len(var) == ndims
    return var


def get_sym_tensor_from_vector(vec: StateBlock, ndims: int) -> JaxArray:
    if ndims == 3:
        tensor = jnp.array([[vec[0], vec[1], vec[2]],
                            [vec[1], vec[3], vec[4]],
                            [vec[2], vec[4], vec[5]]])
    elif ndims == 2:
        tensor = jnp.array([[vec[0], vec[1]], [vec[1], vec[2]]])
    elif ndims == 1:
        tensor = jnp.array([[vec[0]]])
    else:
        raise ValueError("Dimension != 1, 2, or 3")

    return tensor


def get_tensor_from_vector(vec: StateBlock, ndims: int) -> JaxArray:
    if ndims == 3:
        tensor = jnp.array([[vec[0], vec[1], vec[2]],
                            [vec[3], vec[4], vec[5]],
                            [vec[6], vec[7], vec[8]]])
    elif ndims == 2:
        tensor = jnp.array([[vec[0], vec[1]], [vec[2], vec[3]]])
    elif ndims == 1:
        tensor = jnp.array([[vec[0]]])
    else:
        raise ValueError("Dimension != 1, 2, or 3")

    return tensor


def get_vector_from_sym_tensor(tensor: JaxArray, ndims: int) -> JaxArray:
    if ndims == 3:
        vec = jnp.array([tensor[0, 0], tensor[0, 1], tensor[0, 2],
                         tensor[1, 1], tensor[1, 2], tensor[2, 2]])
    elif ndims == 2:
        vec = jnp.array([tensor[0, 0], tensor[0, 1], tensor[1, 1]])
    elif ndims == 1:
        vec = jnp.array([tensor[0, 0]])
    else:
        raise ValueError("Dimension != 1, 2, or 3")

    return vec


def get_vector_from_tensor(tensor: JaxArray, ndims: int) -> JaxArray:
    if ndims == 3:
        vec = jnp.array([tensor[0, 0], tensor[0, 1], tensor[0, 2],
                         tensor[1, 0], tensor[1, 1], tensor[1, 2],
                         tensor[2, 0], tensor[2, 1], tensor[2, 2]])
    elif ndims == 2:
        vec = jnp.array([tensor[0, 0], tensor[0, 1], tensor[1, 0],
                         tensor[1, 1]])
    elif ndims == 1:
        vec = jnp.array([tensor[0, 0]])
    else:
        raise ValueError("Dimension != 1, 2, or 3")

    return vec


def put_2D_tensor_into_3D(tensor_2D: JaxArray) -> JaxArray:
    assert tensor_2D.shape == (2, 2)
    tensor_3D = jnp.array([[tensor_2D[0, 0], tensor_2D[0, 1], 0.],
                           [tensor_2D[1, 0], tensor_2D[1, 1], 0.],
                           [0., 0., 0.]])

    return tensor_3D


def get_2D_tensor_from_3D(tensor_3D: JaxArray) -> JaxArray:
    assert tensor_3D.shape == (3, 3)
    tensor_2D = tensor_3D[:2, :2]

    return tensor_2D


def put_tensor_into_3D(tensor: JaxArray, def_type: int) -> JaxArray:
    if def_type == DefType.FULL_3D:
        tensor_3D = tensor
    elif def_type == DefType.PLANE_STRAIN or def_type == DefType.PLANE_STRESS:
        tensor_3D = jnp.array([[tensor[0, 0], tensor[0, 1], 0.],
                               [tensor[1, 0], tensor[1, 1], 0.],
                               [0., 0., 0.]])
    elif def_type == DefType.UNIAXIAL_STRESS:
        tensor_3D = jnp.array([[tensor[0, 0], 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.]])
    elif def_type == DefType.PURE_SHEAR:
        tensor_3D = jnp.array([[0., tensor[0, 0], 0.],
                               [tensor[0, 0], 0., 0.],
                               [0., 0., 0.]])
    else:
        raise ValueError(f"Unknown def_type: {def_type}")

    return tensor_3D


def get_tensor_from_3D(tensor_3D: JaxArray, def_type: int) -> JaxArray:
    assert tensor_3D.shape == (3, 3)
    if def_type == DefType.FULL_3D:
        return tensor_3D
    elif def_type == DefType.PLANE_STRAIN or def_type == DefType.PLANE_STRESS:
        return tensor_3D[:2, :2]
    elif def_type == DefType.UNIAXIAL_STRESS:
        return tensor_3D[0, 0]
    elif def_type == DefType.PURE_SHEAR:
        return tensor_3D[0, 1]
    else:
        raise ValueError(f"Unknown def_type: {def_type}")
