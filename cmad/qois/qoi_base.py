"""Marker base class shared by every CMAD QoI.

The MP and FE QoI hierarchies have structurally different lifecycles
and share no methods. The registry needs a single common type so
:func:`cmad.io.registry.resolve_qoi` can return one
``type[QoIBase]`` regardless of which hierarchy a deck names; the
caller dispatches on :attr:`QoIBase.problem_type` to enforce the
deck's ``problem.type`` matches the resolved class.
"""
from __future__ import annotations

from typing import ClassVar


class QoIBase:
    """Marker class with the registry-side problem-type tag.

    Concrete QoIs do not subclass this directly; they subclass either
    :class:`cmad.qois.qoi.QoI` (MP) or :class:`cmad.qois.fe_qoi.FEQoI`
    (FE), each of which sets :attr:`problem_type` to its hierarchy.
    """

    problem_type: ClassVar[str]
