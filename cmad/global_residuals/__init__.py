"""Global-residual abstract contract and composed-helper builders."""
from cmad.global_residuals.interpolation import interpolate_global_fields_at_ip
from cmad.global_residuals.modes import GlobalResidualMode

__all__ = [
    "GlobalResidualMode",
    "interpolate_global_fields_at_ip",
]
