"""Per-block interpolation of element-local basis coefficients to an integration point."""
from collections.abc import Sequence

from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.typing import JaxArray


def interpolate_global_fields_at_ip(
        U: Sequence[JaxArray],
        shapes_ip: Sequence[ShapeFunctionsAtIP],
        resid_names: Sequence[str | None],
) -> GlobalFieldsAtPoint:
    """Interpolate element-local basis coefficients U to an IP-level GlobalFieldsAtPoint.

    Per-residual-block generic: each block carries its own basis-function
    count, so mixed-basis formulations (Taylor-Hood, quadratic + linear,
    DG0 + CG1, ...) iterate cleanly. Same-basis multi-field problems
    (u-p, u-T, ...) pass identical shapes_ip entries.

    Each ``U[i]`` has shape ``(num_basis_fns[i], num_eqs[i])``; each
    ``shapes_ip[i].N`` has shape ``(num_basis_fns[i],)`` and
    ``shapes_ip[i].grad_N`` has shape ``(num_basis_fns[i], ndims)``.

    Output convention: ``fields[name]`` has shape ``(num_eqs[i],)``;
    ``grad_fields[name]`` has shape ``(num_eqs[i], ndims)`` with
    ``grad_fields[name][k, j] = ∂u_k/∂x_j`` (component-outer,
    spatial-inner). Matches :func:`cmad.models.global_fields.mp_U_from_F`.
    """
    if any(name is None for name in resid_names):
        raise ValueError(
            "interpolate_global_fields_at_ip requires all resid_names "
            "entries to be set; got an unfilled placeholder. Ensure the "
            "GlobalResidual subclass populates self.resid_names[i] for "
            "every residual block in its __init__."
        )

    fields: dict[str, JaxArray] = {}
    grad_fields: dict[str, JaxArray] = {}
    for name, U_i, shapes_i in zip(resid_names, U, shapes_ip, strict=True):
        assert name is not None  # narrowed by the check above
        fields[name] = shapes_i.N @ U_i
        grad_fields[name] = U_i.T @ shapes_i.grad_N
    return GlobalFieldsAtPoint(fields=fields, grad_fields=grad_fields)
