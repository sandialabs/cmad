from enum import IntEnum


class GlobalDerivType(IntEnum):
    DU = 0
    DU_prev = 1
    DParams = 2
    DXI = 3
    DXI_prev = 4
    DNONE = 5

    DU_DU = 0
    DXI_DXI = 1
    DU_DXI = 2