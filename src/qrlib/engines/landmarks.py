"""QSIM new-landmark introduction (docs/qsim.md §4).

When a variable that was moving becomes steady *inside* an open interval
(an I5/I9 transition), the value it is steady at is a discovered landmark:
we mint a named landmark there, growing that variable's quantity space for
the rest of the behavior branch. Quantity spaces — and therefore compiled
constraint rank references — are per-branch: minting produces a *new*
:class:`~qrlib.model.CompiledModel` (a "frame"); ancestor states keep the
old one.

Rank arithmetic: inserting a landmark inside interval rank ``r`` (odd)
splits it into ``r`` (below), ``r+1`` (the new landmark), ``r+2`` (above);
every prior rank of that variable strictly greater than ``r`` shifts by +2,
and the minting variable's value re-encodes to ``r+1``.

Minting also records **new corresponding values**: for any M+/M-/MINUS/ADD
constraint touching a minted variable whose variables are *all* at landmarks
in the new state, the observed rank tuple is a valid corresponding value
(monotonic functions and sums pin the co-occurring values), and is added to
the branch's frame.
"""

from __future__ import annotations

from dataclasses import replace

from ..model import CompiledConstraint, CompiledModel
from ..quantity import Landmark, Qdir, QVal

__all__ = ["introduce_landmarks"]

_CVAL_KINDS = ("mplus", "mminus", "minus", "add")


def introduce_landmarks(
    compiled: CompiledModel,
    parent_vals: tuple[QVal, ...],
    vals: tuple[QVal, ...],
    max_landmarks: int,
) -> tuple[CompiledModel, tuple[QVal, ...], list[tuple[str, str]]]:
    """Mint landmarks for I5/I9 arrivals in a candidate point state.

    ``parent_vals`` are the preceding interval state's values, ``vals`` the
    candidate point values (both in ``compiled``'s encoding). Returns the
    (possibly new) frame, the re-encoded values, and ``(variable, name)``
    pairs for each minted landmark. Variables whose direction is untracked
    (``IGN``) never mint. A variable that already carries ``max_landmarks``
    discovered landmarks keeps its steady value unnamed (sound, just less
    informative).
    """
    minted: list[tuple[str, str]] = []
    minted_idx: set[int] = set()
    out = list(vals)

    for vi, (pv, nv) in enumerate(zip(parent_vals, out)):
        if nv.dir is not Qdir.STD or nv.mag % 2 == 0:
            continue
        if pv.dir not in (Qdir.INC, Qdir.DEC) or pv.mag != nv.mag:
            continue  # was already steady (or untracked): no discovery event
        var = compiled.var_order[vi]
        space = compiled.spaces[vi]
        already = sum(1 for n in space.names if n.startswith(var + "*"))
        if already >= max_landmarks:
            continue
        name = f"{var}*{already}"
        r = nv.mag
        below = space.effective_landmarks[r // 2]
        new_space = space.insert_landmark(Landmark(name), after=below)
        compiled = _shift_ranks(compiled, vi, r, new_space)
        out[vi] = QVal(r + 1, Qdir.STD)
        minted.append((var, name))
        minted_idx.add(vi)

    if minted:
        compiled = _record_corresponding_values(compiled, tuple(out), minted_idx)
    return compiled, tuple(out), minted


def _shift_ranks(
    compiled: CompiledModel, vi: int, r: int, new_space
) -> CompiledModel:
    """Grow variable ``vi``'s space and shift its rank references above ``r``."""

    def shift(rank: int) -> int:
        return rank + 2 if rank > r else rank

    spaces = list(compiled.spaces)
    spaces[vi] = new_space
    constraints = []
    for cc in compiled.constraints:
        if vi not in cc.vars:
            constraints.append(cc)
            continue
        local = tuple(k for k, v in enumerate(cc.vars) if v == vi)
        cvals = tuple(
            tuple(shift(rank) if k in local else rank for k, rank in enumerate(cv))
            for cv in cc.cvals
        )
        zeros = tuple(
            (shift(z) if k in local and z is not None else z)
            for k, z in enumerate(cc.zeros)
        )
        infs = tuple(
            (lo, shift(hi) if hi is not None else None) if k in local else (lo, hi)
            for k, (lo, hi) in enumerate(cc.infs)
        )
        constraints.append(replace(cc, cvals=cvals, zeros=zeros, infs=infs))
    return replace(compiled, spaces=tuple(spaces), constraints=tuple(constraints))


def _record_corresponding_values(
    compiled: CompiledModel, vals: tuple[QVal, ...], minted_idx: set[int]
) -> CompiledModel:
    constraints = []
    for cc in compiled.constraints:
        if (
            cc.kind in _CVAL_KINDS
            and any(v in minted_idx for v in cc.vars)
            and all(vals[i].mag % 2 == 0 for i in cc.vars)
        ):
            cv = tuple(vals[i].mag for i in cc.vars)
            if cv not in cc.cvals:
                cc = replace(cc, cvals=cc.cvals + (cv,))
        constraints.append(cc)
    return replace(compiled, constraints=tuple(constraints))
