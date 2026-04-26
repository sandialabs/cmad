"""Dirichlet boundary-condition declarations.

A :class:`DirichletBC` declares a constraint on a named node set, applied
to specified components of a named global field. Resolution to flat
``(prescribed_indices, prescribed_values)`` arrays happens at
:func:`cmad.fem.dof.build_dof_map` time; this module ships the dataclass
shape only.

Value sources, in order of generality:

- ``None`` (default) — homogeneous, all zeros. Convenience for clamped
  boundaries.
- ``Sequence[float]`` of length ``len(dofs)`` — spatially+temporally
  constant per-component. Broadcast across the node set.
- ``Callable[[NDArray (N_set, 3), float], NDArray (N_set, len(dofs))]`` —
  per-node values from an expression at time ``t``. The deck loader
  builds these from string expressions via an asteval-backed parser;
  tests use plain Python lambdas. Time-independent BCs accept the ``t``
  argument and ignore it.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DirichletBC:
    """Dirichlet BC declaration.

    ``nodeset_name`` references a key in
    :attr:`cmad.fem.mesh.Mesh.node_sets`.

    ``field_name`` matches one of the field-layout names in the
    :class:`cmad.fem.dof.GlobalDofMap` the BC will be resolved against
    (which by convention align with ``GlobalResidual.var_names`` — the
    field-symbol list, parallel to the residual-equation-label
    ``resid_names``).

    ``dofs`` lists the field-local component indices to constrain — for
    a 3-component vector field, ``[0]`` constrains the x-component while
    ``[0, 1, 2]`` clamps all three.

    ``values`` follows the union semantics in the module docstring:
    ``None`` for homogeneous, a ``Sequence[float]`` of length
    ``len(dofs)`` for spatially constant non-homogeneous, or a callable
    ``(coords, t) -> (N_set, len(dofs))`` for expression-driven /
    time-dependent BCs.

    A length mismatch between ``dofs`` and a sequence-form ``values`` is
    rejected at construction. Callable-form values are not introspected
    here; their return shape is verified at resolution time.
    """

    nodeset_name: str
    field_name: str
    dofs: Sequence[int]
    values: (
        Sequence[float]
        | Callable[[NDArray[np.floating], float], NDArray[np.floating]]
        | None
    ) = None

    def __post_init__(self) -> None:
        if len(self.dofs) == 0:
            raise ValueError("DirichletBC.dofs must be non-empty")
        if self.values is None or callable(self.values):
            return
        n_values = len(self.values)
        if n_values != len(self.dofs):
            raise ValueError(
                f"DirichletBC values length ({n_values}) does not match "
                f"dofs length ({len(self.dofs)})"
            )
