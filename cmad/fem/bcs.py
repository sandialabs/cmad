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
:func:`cmad.fem.surface_bcs.resolve_neumann_bcs` and assembles
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

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmad.typing import JaxArray, Scalar


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
        | Callable[
            [NDArray[np.floating] | JaxArray, Scalar],
            NDArray[np.floating] | JaxArray,
        ]
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
        | Callable[
            [NDArray[np.floating] | JaxArray, Scalar],
            NDArray[np.floating] | JaxArray,
        ]
    )

    def __post_init__(self) -> None:
        if len(self.sideset_names) == 0:
            raise ValueError("NeumannBC.sideset_names must be non-empty")
        if callable(self.values):
            return
        if len(self.values) == 0:
            raise ValueError("NeumannBC.values must be non-empty")


@dataclass(frozen=True)
class RobinBC:
    """A surface flux that depends on the field on the side.

    ``flux(field_value, coords, t)`` takes the field value at one side
    point, shape ``(num_components,)``, and returns the outward flux
    there with the same shape; convection is ``h (T - T_inf)`` and
    radiation ``eps sigma_B (T^4 - T_inf^4)``. The tangent is taken by
    AD of the side residual in :mod:`cmad.fem.surface_bcs`.
    ``sideset_names`` and ``field_name`` are as for :class:`NeumannBC`.
    """

    sideset_names: Sequence[str]
    field_name: str
    flux: Callable[[JaxArray, JaxArray, Scalar], JaxArray]

    def __post_init__(self) -> None:
        if len(self.sideset_names) == 0:
            raise ValueError("RobinBC.sideset_names must be non-empty")


def make_nodal_field_values(
        values_by_step: NDArray[np.floating] | JaxArray,
        data_times: Sequence[float] | NDArray[np.floating],
) -> Callable[
    [NDArray[np.floating] | JaxArray, Scalar],
    JaxArray,
]:
    """Build a :class:`DirichletBC` value callable over measured nodal data.

    ``values_by_step`` is ``(num_frames, N_set, num_dofs)``, already laid
    out to match the BC: its second axis follows
    :func:`cmad.fem.dof.sideset_basis_fns` for the BC's sidesets and its
    third axis follows the BC's ``dofs``. Because the layout is fixed
    when this is built, the returned callable ignores the boundary
    coordinates it is passed.

    ``data_times`` is the strictly increasing times the frames were
    measured at. Values are linearly interpolated in ``t`` between the
    bracketing frames, so the data times need not coincide with the
    solve's steps; a ``t`` landing on a frame time reproduces that frame
    exactly. Outside the measured range the nearest frame is held
    constant. Interpolating rather than selecting keeps the callable
    valid when ``t`` is a tracer inside the time loop.
    """
    data = jnp.asarray(values_by_step, dtype=jnp.float64)
    times = jnp.asarray(data_times, dtype=jnp.float64)
    if data.ndim != 3:
        raise ValueError(
            f"values_by_step must be (num_frames, N_set, num_dofs); got "
            f"shape {data.shape}"
        )
    if data.shape[0] != times.shape[0]:
        raise ValueError(
            f"values_by_step has {data.shape[0]} frames but data_times has "
            f"{times.shape[0]}"
        )
    gaps = np.diff(np.asarray(data_times, dtype=np.float64))
    if gaps.size and not np.all(gaps > 0.0):
        raise ValueError(
            "data_times must be strictly increasing; found a non-positive "
            f"step of {gaps.min():.6g}"
        )

    num_frames = data.shape[0]
    last = num_frames - 2

    def values(
            coords: NDArray[np.floating] | JaxArray, t: Scalar,
    ) -> JaxArray:
        del coords  # the layout is fixed at build time
        if num_frames == 1:
            return data[0]  # a single frame holds for the whole history
        lo = jnp.clip(jnp.searchsorted(times, t, side="right") - 1, 0, last)
        span = times[lo + 1] - times[lo]
        weight = jnp.clip((t - times[lo]) / span, 0.0, 1.0)
        before = jnp.take(data, lo, axis=0)
        after = jnp.take(data, lo + 1, axis=0)
        return before + weight * (after - before)

    return values
