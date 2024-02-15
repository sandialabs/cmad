from enum import IntEnum


class DefType(IntEnum):
    FULL_3D = 0
    PLANE_STRAIN = 1
    PLANE_STRESS = 2
    UNIAXIAL_STRESS = 3
    PURE_SHEAR = 4


def def_type_ndims(def_type):
    if def_type == DefType.FULL_3D:
        return 3
    elif def_type == DefType.PLANE_STRAIN or def_type == DefType.PLANE_STRESS:
        return 2
    elif def_type == DefType.UNIAXIAL_STRESS or def_type == DefType.PURE_SHEAR:
        return 1
    else:
        raise NotImplementedError
