"""Batched interval propagation (docs/gpu-tensorization.md §6-7).

The semi-quantitative layer (:mod:`qrlib.semiquant`) refines a *single*
behavior's numeric bounds by interval propagation to a fixpoint. Its
per-state **algebraic** narrowing — the ``ADD``/``MINUS``/``MULT``/``At``
interval arithmetic — is the embarrassingly parallel part: it applies the
same operator independently to every state, so it vectorizes over a batch.
This module carries interval bounds as ``(B, V)`` ``lo``/``hi`` float
tensors and narrows a whole batch of states at once, pruning any state whose
bounds cross.

Parity contract: the interval primitives mirror
:class:`qrlib.semiquant.Interval` expression-for-expression (same
``0 * inf = 0`` and ``inf + -inf`` guard conventions, same "divisor
straddling zero yields no refinement" rule), in float64, and interval
narrowing is a monotone confluent operator — so the batched fixpoint
reaches exactly the bounds the reference reaches, regardless of the
in-pass update order. The equivalence tests assert this against the
reference :class:`~qrlib.semiquant.Interval` on random models.

Scope: this covers the per-state algebraic constraints. Monotonic-function
**envelopes** (arbitrary Python callables) and the cross-state couplings
(continuity, mean-value time bounds, ``CONSTANT`` intersection across a
behavior) are inherently sequential along a behavior and stay in the
reference ``semiquant.refine`` path; this is the batched screen that runs
first, or over many states/behaviors at once.
"""

from __future__ import annotations

from math import inf

import torch

from ..model import CompiledModel, Model
from ..semiquant import _mag_range
from ..state import QState

__all__ = [
    "iadd",
    "ineg",
    "isub",
    "imul",
    "idiv",
    "intersect",
    "narrow_batch",
    "bounds_from_states",
    "feasible_mask",
]

# An interval batch is a pair of tensors (lo, hi) of matching shape; lo > hi
# marks an empty (infeasible) interval, exactly as in semiquant.Interval.


def _guard(t: torch.Tensor, fallback: float) -> torch.Tensor:
    return torch.where(torch.isnan(t), torch.full_like(t, fallback), t)


def iadd(a, b):
    (alo, ahi), (blo, bhi) = a, b
    return _guard(alo + blo, -inf), _guard(ahi + bhi, inf)


def ineg(a):
    alo, ahi = a
    return -ahi, -alo


def isub(a, b):
    return iadd(a, ineg(b))


def imul(a, b):
    (alo, ahi), (blo, bhi) = a, b
    p = [
        _guard(alo * blo, 0.0),
        _guard(alo * bhi, 0.0),
        _guard(ahi * blo, 0.0),
        _guard(ahi * bhi, 0.0),
    ]
    lo = torch.minimum(torch.minimum(p[0], p[1]), torch.minimum(p[2], p[3]))
    hi = torch.maximum(torch.maximum(p[0], p[1]), torch.maximum(p[2], p[3]))
    return lo, hi


def idiv(a, b):
    """Interval quotient; lanes whose divisor straddles zero yield
    ``(-inf, inf)`` (no refinement), matching the reference's ``None``."""
    (alo, ahi), (blo, bhi) = a, b
    straddle = (blo <= 0) & (0 <= bhi)
    safe_lo = torch.where(straddle, torch.ones_like(blo), blo)
    safe_hi = torch.where(straddle, torch.ones_like(bhi), bhi)
    q = [
        _guard(alo / safe_lo, -inf),
        _guard(alo / safe_hi, -inf),
        _guard(ahi / safe_lo, -inf),
        _guard(ahi / safe_hi, -inf),
    ]
    lo = torch.minimum(torch.minimum(q[0], q[1]), torch.minimum(q[2], q[3]))
    q = [
        _guard(alo / safe_lo, inf),
        _guard(alo / safe_hi, inf),
        _guard(ahi / safe_lo, inf),
        _guard(ahi / safe_hi, inf),
    ]
    hi = torch.maximum(torch.maximum(q[0], q[1]), torch.maximum(q[2], q[3]))
    lo = torch.where(straddle, torch.full_like(lo, -inf), lo)
    hi = torch.where(straddle, torch.full_like(hi, inf), hi)
    return lo, hi


def intersect(a, b):
    (alo, ahi), (blo, bhi) = a, b
    return torch.maximum(alo, blo), torch.minimum(ahi, bhi)


def narrow_batch(
    model: Model | CompiledModel,
    lo: torch.Tensor,
    hi: torch.Tensor,
    *,
    region: str | None = None,
    max_passes: int = 30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Narrow a batch of states' bounds by the region's algebraic
    constraints to a fixpoint. ``lo``/``hi`` are ``(B, V)`` in the model's
    variable order. Returns ``(lo, hi, feasible)`` with ``feasible`` a
    ``(B,)`` bool mask (False where any variable's interval emptied)."""
    frame = model.compile() if isinstance(model, Model) else model
    if region is None:
        region = frame.initial_region
    if lo.shape != hi.shape or lo.dim() != 2 or lo.shape[1] != len(frame.var_order):
        raise ValueError(
            f"lo/hi must be (B, {len(frame.var_order)}) and matching, "
            f"got {tuple(lo.shape)} and {tuple(hi.shape)}"
        )
    lo = lo.clone()
    hi = hi.clone()

    def col(i):
        return (lo[:, i], hi[:, i])

    def narrow(i, iv):
        lo[:, i] = torch.maximum(lo[:, i], iv[0])
        hi[:, i] = torch.minimum(hi[:, i], iv[1])

    # precompute At target intervals (state-independent constants)
    at_targets = {}
    for cc in frame.constraints_of(region):
        if cc.kind == "at":
            r = _mag_range(frame.spaces[cc.vars[0]], cc.cvals[0][0])
            at_targets[id(cc)] = (r.lo, r.hi)

    for _ in range(max_passes):
        prev_lo, prev_hi = lo.clone(), hi.clone()
        for cc in frame.constraints_of(region):
            vs = cc.vars
            if cc.kind == "add":  # x + y = z
                narrow(vs[2], iadd(col(vs[0]), col(vs[1])))
                narrow(vs[0], isub(col(vs[2]), col(vs[1])))
                narrow(vs[1], isub(col(vs[2]), col(vs[0])))
            elif cc.kind == "minus":  # y = -x
                narrow(vs[1], ineg(col(vs[0])))
                narrow(vs[0], ineg(col(vs[1])))
            elif cc.kind == "mult":  # z = x * y
                narrow(vs[2], imul(col(vs[0]), col(vs[1])))
                narrow(vs[0], idiv(col(vs[2]), col(vs[1])))
                narrow(vs[1], idiv(col(vs[2]), col(vs[0])))
            elif cc.kind == "at":
                t = at_targets[id(cc)]
                narrow(vs[0], (torch.full_like(lo[:, 0], t[0]),
                               torch.full_like(hi[:, 0], t[1])))
            # mplus/mminus (envelopes), constant, deriv: not per-state
            # algebraic — handled in the reference behavior-level path
        if torch.equal(lo, prev_lo) and torch.equal(hi, prev_hi):
            break

    feasible = (lo <= hi).all(dim=1)
    return lo, hi, feasible


def bounds_from_states(
    model: Model | CompiledModel, states, *, dtype=torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the ``(B, V)`` ``lo``/``hi`` bound tensors from states'
    magnitude ranges (each variable's declared numeric range for its
    qualitative magnitude), in the model's variable order."""
    frame = model.compile() if isinstance(model, Model) else model
    states = list(states)
    lo = torch.empty((len(states), len(frame.var_order)), dtype=dtype)
    hi = torch.empty_like(lo)
    for b, st in enumerate(states):
        for vi, name in enumerate(frame.var_order):
            r = _mag_range(frame.spaces[vi], st[name].mag)
            lo[b, vi] = r.lo
            hi[b, vi] = r.hi
    return lo, hi


def feasible_mask(
    model: Model | CompiledModel,
    states,
    *,
    region: str | None = None,
    max_passes: int = 30,
) -> torch.Tensor:
    """Batched numeric feasibility screen: a ``(B,)`` bool mask, False for
    states whose magnitude ranges are algebraically inconsistent under the
    region's ``ADD``/``MINUS``/``MULT``/``At`` constraints. A sound
    pre-filter — it can only *refute* (an emptied bound is a real
    contradiction), never certify."""
    lo, hi = bounds_from_states(model, states)
    _, _, feasible = narrow_batch(
        model, lo, hi, region=region, max_passes=max_passes
    )
    return feasible
