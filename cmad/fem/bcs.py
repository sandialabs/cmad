"""Boundary-condition declarations.

Two sideset-keyed BC dataclasses ship here: :class:`DirichletBC` for
essential constraints on named field components, and
:class:`NeumannBC` for natural surface fluxes on a full field
vector. Both share the same side-walk pattern — name one or more
keys in :attr:`cmad.fem.mesh.Mesh.side_sets`, walk every
``(elem, local_side_id)`` pair across the named sets, call
:meth:`cmad.fem.finite_element.FiniteElement.side_basis_fns` to get
per-element basis-fn indices on each side, and consume the per-side
data — but diverge on what they do with it. Multi-sideset listing
is the natural way to act on a boundary patch covering several
Exodus side sets (e.g. all six faces of a box) as a single BC,
without spurious intra-BC double-prescription on shared edges and
corners.

DirichletBC resolves to flat ``(prescribed_indices,
prescribed_values)`` arrays at :func:`cmad.fem.dof.build_dof_map`
time. Two DBCs may prescribe the same ``(field, basis_fn, dof)`` if
and only if their values agree at the queried time. The
structurally overprescribed subset is identified at
:func:`build_dof_map` time and stored on the dofmap as
``overprescribed_dbc_groups``; the value-consistency check at
:meth:`cmad.fem.dof.GlobalDofMap.evaluate_prescribed_values` time
iterates only that subset (per-step for time-dependent BCs).
Inconsistent overlaps raise ``ValueError`` with a diagnostic naming
the conflicting BCs and the global equation number decoded to
``(field, basis_fn, dof)``.

NeumannBC resolves to per-(family, local_side_id) elem-id groups in
:func:`cmad.fem.neumann.resolve_neumann_bcs` and assembles
``R -= ∫_∂Ω N · t̄ dA`` into the global residual at the field's
basis fns on the named sides. Cross-NBC overlaps are silent-additive
(surface tractions superpose linearly); no consistency check.
Explicit ``(coords, t)`` flux is U-independent, so K gets no
contribution from NeumannBC.

DirichletBC value sources, in order of generality:

- ``None`` (default) — homogeneous, all zeros. Convenience for
  clamped boundaries.
- ``Sequence[float]`` of length ``len(dofs)`` — spatially+temporally
  constant per-component. Broadcast across the BC's vertex set.
- ``Callable[[NDArray (N_set, 3), float], NDArray (N_set, len(dofs))]``
  — per-vertex values from an expression at time ``t``. The deck
  loader builds these from string expressions via an asteval-backed
  parser; tests use plain Python lambdas. Time-independent BCs
  accept the ``t`` argument and ignore it.

NeumannBC value sources:

- ``Sequence[float]`` of length equal to the resolved field's
  component count — spatially+temporally constant flux vector
  (e.g. uniform pressure on a face). Broadcast across the BC's
  per-IP side coords at assembly time.
- ``Callable[[NDArray (N_side_ips, 3), float], NDArray (N_side_ips, num_components)]``
  — flux from an expression at time ``t``. No ``None`` form: a
  zero-flux NBC is just an absent NBC.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DirichletBC:
    """Dirichlet BC declaration.

    ``sideset_names`` is a sequence of keys in
    :attr:`cmad.fem.mesh.Mesh.side_sets`. The BC's vertex set is the
    union of vertices reached by walking ``(elem, local_side_id)``
    pairs across all named side sets and resolving each pair via
    :meth:`cmad.fem.finite_element.FiniteElement.side_basis_fns`,
    deduplicated. Single-side BCs pass a one-element sequence
    (e.g. ``["xmin_sides"]``).

    ``field_name`` matches one of the field-layout names in the
    :class:`cmad.fem.dof.GlobalDofMap` the BC will be resolved
    against (which by convention align with
    ``GlobalResidual.var_names`` — the field-symbol list, parallel
    to the residual-equation-label ``resid_names``).

    ``dofs`` lists the field-local component indices to constrain —
    for a 3-component vector field, ``[0]`` constrains the
    x-component while ``[0, 1, 2]`` clamps all three.

    ``values`` follows the union semantics in the module docstring:
    ``None`` for homogeneous, a ``Sequence[float]`` of length
    ``len(dofs)`` for spatially constant non-homogeneous, or a
    callable ``(coords, t) -> (N_set, len(dofs))`` for expression-
    driven / time-dependent BCs.

    A length mismatch between ``dofs`` and a sequence-form ``values``
    is rejected at construction. An empty ``sideset_names`` or empty
    ``dofs`` is rejected. Callable-form values are not introspected
    here; their return shape is verified at resolution time.
    """

    sideset_names: Sequence[str]
    field_name: str
    dofs: Sequence[int]
    values: (
        Sequence[float]
        | Callable[[NDArray[np.floating], float], NDArray[np.floating]]
        | None
    ) = None

    def __post_init__(self) -> None:
        if len(self.sideset_names) == 0:
            raise ValueError("DirichletBC.sideset_names must be non-empty")
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


@dataclass(frozen=True)
class NeumannBC:
    """Neumann (natural) BC declaration.

    ``sideset_names`` is a sequence of keys in
    :attr:`cmad.fem.mesh.Mesh.side_sets`. The BC's surface-integral
    domain is the union of ``(elem, local_side_id)`` pairs reached
    by walking the named side sets, deduplicated. Single-side BCs
    pass a one-element sequence (e.g. ``["xmax_sides"]``).

    ``field_name`` matches one of the field-layout names in the
    :class:`cmad.fem.dof.GlobalDofMap` the BC will be resolved
    against (which by convention align with
    ``GlobalResidual.var_names``).

    ``values`` carries the prescribed flux vector, full-field-width:

    - a ``Sequence[float]`` of length equal to the resolved field's
      component count — spatially+temporally constant flux (e.g.
      uniform ``(0, 0, -p)`` for a downward pressure on a +z face);
    - a callable ``(coords, t) -> (N_side_ips, num_components)`` for
      expression-driven / time-dependent flux. Mirrors
      :class:`DirichletBC` callable shape.

    No ``dofs`` selector — flux components that should be zero are
    set to zero in the values vector. No ``None`` form — a zero-flux
    NBC is just an absent NBC.

    An empty ``sideset_names`` or empty sequence-form ``values`` is
    rejected at construction. Sequence length is validated against
    the resolved field's component count at resolution time;
    callable-form values are not introspected here, and their return
    shape is verified at the first surface-assembly call.
    """

    sideset_names: Sequence[str]
    field_name: str
    values: (
        Sequence[float]
        | Callable[[NDArray[np.floating], float], NDArray[np.floating]]
    )

    def __post_init__(self) -> None:
        if len(self.sideset_names) == 0:
            raise ValueError("NeumannBC.sideset_names must be non-empty")
        if callable(self.values):
            return
        if len(self.values) == 0:
            raise ValueError("NeumannBC.values must be non-empty")
