"""Rotations between the global frame and a material's own frame.

``Q`` is the material's ``rotation matrix``, material to global:
``Q_ij = e_i (global) · e_j (material)``.
"""
from typing import Any

import numpy as np

from cmad.models.deformation_types import DefType
from cmad.parameters.parameters import Parameters
from cmad.typing import JaxArray


def rotate_into_material_frame(
        A: JaxArray, params: dict[str, Any], has_material_rotation: bool,
) -> JaxArray:
    if has_material_rotation:
        Q = params["rotation matrix"]
        return Q.T @ A @ Q

    return A


def rotate_out_of_material_frame(
        A: JaxArray, params: dict[str, Any], has_material_rotation: bool,
) -> JaxArray:
    if has_material_rotation:
        Q = params["rotation matrix"]
        return Q @ A @ Q.T

    return A


def require_in_plane_rotation(
        parameters: Parameters, def_type: int, model_name: str,
) -> None:
    """Raise unless a plane strain or plane stress material's rotation
    matrix is about the out-of-plane axis."""
    is_2D = def_type in (DefType.PLANE_STRAIN, DefType.PLANE_STRESS)
    if is_2D and "rotation matrix" in parameters.values:
        Q = np.asarray(parameters.values["rotation matrix"])
        if not (np.allclose(Q[2, :2], 0.0) and np.allclose(Q[:2, 2], 0.0)):
            raise ValueError(
                f"{model_name} in plane strain or plane stress needs a "
                "rotation matrix about the out-of-plane axis")
