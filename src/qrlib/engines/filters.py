"""QSIM constraint predicates and successor filtering.

Reference (readable, unvectorized) implementations of:

- per-constraint consistency predicates over qualitative values, including
  corresponding-value orderings and the algebra of infinite landmarks
  (:func:`check`);
- tuple + Waltz filtering: prune per-variable candidate domains to a
  fixpoint of per-constraint consistency (:func:`prune_domains`);
- global interpretation assembly: enumerate complete, consistent successor
  assignments (:func:`assemble`);
- the infinity admissibility rule for point states (:func:`admissible_at_infinity`).

Sign conventions: magnitudes compare through their rank encoding;
``sign`` of a magnitude is its comparison against the rank of the ``0``
landmark; a direction's sign is ``qdir - 1``. The qualitative sum of two
signs is the set of possible signs of the sum (``{-1,0,1}`` when they
oppose).

Everything here is pure and deterministic; the tensor engine (phase 5)
must reproduce these results exactly.
"""

from __future__ import annotations

from itertools import product
from typing import Iterator, Sequence

from ..model import CompiledConstraint, CompiledModel
from ..quantity import Qdir, QVal
from ..state import QState

__all__ = [
    "check",
    "check_state",
    "prune_domains",
    "assemble",
    "admissible_at_infinity",
]


def _cmp(a: int, b: int) -> int:
    return (a > b) - (a < b)


def _qsum(s1: int, s2: int) -> frozenset[int]:
    """Possible signs of a sum of two quantities with signs s1, s2."""
    if s1 == 0:
        return frozenset((s2,))
    if s2 == 0:
        return frozenset((s1,))
    if s1 == s2:
        return frozenset((s1,))
    return frozenset((-1, 0, 1))


def _inf_status(qv: QVal, inf_pair: tuple[int | None, int | None]) -> int:
    """-1 / +1 if the value sits at the -inf / +inf landmark, else 0."""
    neg, pos = inf_pair
    if neg is not None and qv.mag == neg:
        return -1
    if pos is not None and qv.mag == pos:
        return 1
    return 0


def check(cc: CompiledConstraint, vals: Sequence[QVal]) -> bool:
    """Is this tuple of values (aligned with ``cc.vars``) consistent?"""
    kind = cc.kind

    if kind == "deriv":  # d(x)/dt = y : x's direction is the sign of y
        x, y = vals
        return x.dir.sign == _cmp(y.mag, cc.zeros[1])

    if kind == "constant":
        return vals[0].dir is Qdir.STD

    if kind == "mplus":  # y = f(x), f monotonically increasing
        x, y = vals
        if x.dir != y.dir:
            return False
        return all(_cmp(x.mag, cx) == _cmp(y.mag, cy) for cx, cy in cc.cvals)

    if kind in ("mminus", "minus"):  # y = f(x) decreasing; y = -x
        x, y = vals
        if x.dir.sign != -y.dir.sign:
            return False
        return all(_cmp(x.mag, cx) == -_cmp(y.mag, cy) for cx, cy in cc.cvals)

    if kind == "add":  # x + y = z
        x, y, z = vals
        if z.dir.sign not in _qsum(x.dir.sign, y.dir.sign):
            return False
        for cx, cy, cz in cc.cvals:
            if _cmp(z.mag, cz) not in _qsum(_cmp(x.mag, cx), _cmp(y.mag, cy)):
                return False
        # Algebra of infinities: an infinite operand forces the sum, and a
        # finite sum of one infinite operand needs the opposite infinity.
        ix, iy, iz = (_inf_status(v, p) for v, p in zip(vals, cc.infs))
        if ix == 0 and iy == 0:
            allowed = frozenset((0,))
        elif ix == 0:
            allowed = frozenset((iy,))
        elif iy == 0:
            allowed = frozenset((ix,))
        elif ix == iy:
            allowed = frozenset((ix,))
        else:  # +inf + -inf: indeterminate
            allowed = frozenset((-1, 0, 1))
        return iz in allowed

    if kind == "mult":  # z = x * y
        x, y, z = vals
        sx = _cmp(x.mag, cc.zeros[0])
        sy = _cmp(y.mag, cc.zeros[1])
        sz = _cmp(z.mag, cc.zeros[2])
        if sz != sx * sy:
            return False
        # z' = x'y + xy' : direction sign of z is a qualitative sum
        return z.dir.sign in _qsum(x.dir.sign * sy, sx * y.dir.sign)

    raise ValueError(f"unknown constraint kind {kind!r}")


def check_state(
    compiled: CompiledModel,
    state: QState,
    constraints: Sequence[CompiledConstraint] | None = None,
) -> CompiledConstraint | None:
    """First constraint the state violates, or None if consistent.

    ``constraints`` restricts the check to a region's active subset
    (default: all of the model's constraints)."""
    active = compiled.constraints if constraints is None else constraints
    vals = tuple(state[v] for v in compiled.var_order)
    for cc in active:
        if not check(cc, tuple(vals[i] for i in cc.vars)):
            return cc
    return None


def prune_domains(
    compiled: CompiledModel,
    domains: list[list[QVal]],
    constraints: Sequence[CompiledConstraint] | None = None,
) -> list[list[QVal]] | None:
    """Tuple + Waltz filtering: restrict each variable's candidates to values
    that participate in some consistent tuple of every active constraint,
    iterated to a fixpoint. Returns None if any domain empties."""
    active = compiled.constraints if constraints is None else constraints
    domains = [list(d) for d in domains]
    changed = True
    while changed:
        changed = False
        for cc in active:
            local = [domains[i] for i in cc.vars]
            surviving = [t for t in product(*local) if check(cc, t)]
            if not surviving:
                return None
            for k, vi in enumerate(cc.vars):
                keep = [qv for qv in domains[vi] if any(t[k] == qv for t in surviving)]
                if len(keep) != len(domains[vi]):
                    domains[vi] = keep
                    changed = True
    return domains


def assemble(
    compiled: CompiledModel,
    domains: list[list[QVal]],
    constraints: Sequence[CompiledConstraint] | None = None,
) -> Iterator[tuple[QVal, ...]]:
    """Complete consistent assignments from (pruned) candidate domains, in
    deterministic order. Domain sizes are tiny (<= 4 per variable), so the
    reference implementation is a filtered cross-product."""
    active = compiled.constraints if constraints is None else constraints
    for combo in product(*domains):
        if all(
            check(cc, tuple(combo[i] for i in cc.vars)) for cc in active
        ):
            yield combo


def admissible_at_infinity(compiled: CompiledModel, vals: Sequence[QVal]) -> bool:
    """Infinity admissibility for point states.

    A point state containing a variable at an infinite landmark represents
    the limit t -> infinity. At that limit every variable must have reached
    its own limit: it is either at an infinite landmark itself or steady
    (a finite limit, possibly approached asymptotically). States violating
    this describe 'reaching infinity in finite time' and are excluded.
    States with no variable at infinity are unconstrained by this rule.
    A variable with untracked direction (``IGN``) might be steady, so it is
    admitted (sound: never excludes a real behavior).
    """
    statuses = [
        _inf_status(qv, compiled.inf_ranks(vi)) for vi, qv in enumerate(vals)
    ]
    if not any(statuses):
        return True
    return all(
        s != 0 or qv.dir in (Qdir.STD, Qdir.IGN) for s, qv in zip(statuses, vals)
    )
