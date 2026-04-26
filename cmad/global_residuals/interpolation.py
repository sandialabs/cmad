"""Per-block interpolation of element-local basis coefficients to an integration point."""
from collections.abc import Sequence

from cmad.fem.shapes import ShapeFunctionsAtIP
from cmad.models.global_fields import GlobalFieldsAtPoint
from cmad.typing import JaxArray


def interpolate_global_fields_at_ip(
        U: Sequence[JaxArray],
        shapes_ip: Sequence[ShapeFunctionsAtIP],
        var_names: Sequence[str | None],
) -> GlobalFieldsAtPoint:
    """Interpolate element-local basis coefficients U to an IP-level GlobalFieldsAtPoint.

    Per-residual-block generic: each block carries its own basis-function
    count, so mixed-basis formulations (Taylor-Hood, quadratic + linear,
    DG0 + CG1, ...) iterate cleanly. Same-basis multi-field problems
    (u-p, u-T, ...) pass identical shapes_ip entries.

    Each ``U[i]`` has shape ``(num_basis_fns[i], num_eqs[i])``; each
    ``shapes_ip[i].N`` has shape ``(num_basis_fns[i],)`` and
    ``shapes_ip[i].grad_N`` has shape ``(num_basis_fns[i], ndims)``.

    ``var_names[i]`` is the field symbol — ``"u"`` for displacement,
    ``"p"`` for pressure, ``"T"`` for temperature — and is the dict key
    used in ``U_ip.fields[var_names[i]]`` and
    ``U_ip.grad_fields[var_names[i]]``. The corresponding governing-
    equation label (``"displacement"``, ``"pressure"``, ``"energy"``,
    ...) lives in parallel as ``GlobalResidual.resid_names``; the split
    lets each name carry its semantic load — ``resid_names`` for output,
    deck schema, and post-processing, and ``var_names`` for the
    field-symbol-keyed pytree consumed by Models.

    Output convention: ``fields[name]`` has shape ``(num_eqs[i],)``;
    ``grad_fields[name]`` has shape ``(num_eqs[i], ndims)`` with
    ``grad_fields[name][k, j] = ∂u_k/∂x_j`` (component-outer,
    spatial-inner). Matches :func:`cmad.models.global_fields.mp_U_from_F`.
    """
    if any(name is None for name in var_names):
        raise ValueError(
            "interpolate_global_fields_at_ip requires all var_names "
            "entries to be set; got an unfilled placeholder. Ensure the "
            "GlobalResidual subclass populates self.var_names[i] for "
            "every residual block in its __init__."
        )

    fields: dict[str, JaxArray] = {}
    grad_fields: dict[str, JaxArray] = {}
    for name, U_i, shapes_i in zip(var_names, U, shapes_ip, strict=True):
        assert name is not None  # narrowed by the check above
        fields[name] = shapes_i.N @ U_i
        grad_fields[name] = U_i.T @ shapes_i.grad_N
    return GlobalFieldsAtPoint(fields=fields, grad_fields=grad_fields)
