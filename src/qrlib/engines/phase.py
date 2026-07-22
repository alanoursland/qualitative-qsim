"""Phase-space non-intersection filter (after Lee & Kuipers).

A trajectory of an autonomous planar flow cannot cross itself. Projected
onto a declared phase pair ``(x, y)`` — ``y = dx/dt`` via a ``Deriv``
constraint — every behavior traces a curve in the (x, y) plane, and the
Jordan-curve argument behind the Poincaré return map forces successive
same-direction crossings of any transversal segment to be monotone along
it: strictly closer each time, strictly farther each time, or exactly
repeating (a closed orbit). QSIM's local filters know nothing of this —
they happily extend a damped oscillation whose amplitude shrinks, then
grows, then shrinks again — so non-intersection is the classic *global*
filter: it constrains the whole path, not one transition.

Implementable form: at a point state where one coordinate of the pair
sits at a (finite) landmark while moving (dir INC or DEC), the behavior
crosses the grid line through that landmark, at a position given by the
other coordinate's magnitude. Crossings are grouped by (line, crossing
direction, declared segment of the position coordinate), and each
group's position sequence along the path must be interpretable as
all-equal, strictly increasing, or strictly decreasing. A candidate
successor whose new crossing provably violates all three patterns cannot
lie on a non-self-intersecting curve and is pruned. Only *provable*
violations prune: positions that landmark discovery never separated
compare as unknown and never eliminate anything.

Soundness rests on what declaring the pair asserts (see
``SimConfig.phase_pairs``): the pair determines the flow (second-order
autonomous system — every other variable a function of x and y), and
equilibria of the pair lie at declared landmark values, so a declared
segment between two crossings is equilibrium-free — the transversality
the Jordan argument needs. Both hold for the textbook second-order
models; declaring a pair that violates them can prune real behaviors,
exactly as a wrong constraint would.
"""

from __future__ import annotations

from ..model import CompiledModel
from ..quantity import NEG_INF, POS_INF, Qdir, QuantitySpace
from ..state import QState, TimeTag

__all__ = ["admits", "validate_pairs"]

_MOVING = (Qdir.DEC, Qdir.INC)

# A crossing position: bounding landmark names + whether it is exactly at
# one. Names outlive frames (landmark discovery only inserts), so bounds
# recorded in an ancestor's frame resolve in every descendant frame.
# Bounds = (lo_name, hi_name, at_landmark)


def validate_pairs(frame: CompiledModel, cfg) -> tuple[tuple[int, int], ...]:
    """Resolve and validate ``cfg.phase_pairs`` against the model."""
    if not cfg.phase_pairs:
        return ()
    if cfg.envisionment:
        raise ValueError(
            "phase_pairs is a path-dependent filter and is incompatible "
            "with envisionment mode (merged nodes have no single history)"
        )
    pairs = []
    for x, y in cfg.phase_pairs:
        if x == y:
            raise ValueError(f"phase pair ({x!r}, {y!r}) needs two distinct variables")
        for name in (x, y):
            if name not in frame.var_order:
                raise ValueError(f"phase pair names unknown variable {name!r}")
            if name in cfg.ignore_qdir:
                raise ValueError(
                    f"phase pair variable {name!r} is direction-ignored "
                    f"(crossings need tracked directions)"
                )
        xi, yi = frame.index(x), frame.index(y)
        if not any(
            c.kind == "deriv" and c.vars == (xi, yi) for c in frame.constraints
        ):
            raise ValueError(
                f"phase pair ({x!r}, {y!r}) requires a Deriv({x!r}, {y!r}) "
                f"constraint (y must be x's time derivative)"
            )
        pairs.append((xi, yi))
    return tuple(pairs)


def _bounds(space: QuantitySpace, rank: int) -> tuple[str, str, bool]:
    eff = space.effective_landmarks
    if rank % 2 == 0:
        return (eff[rank // 2], eff[rank // 2], True)
    return (eff[rank // 2], eff[rank // 2 + 1], False)


def _declared_rank(space: QuantitySpace, declared: QuantitySpace, rank: int) -> int:
    """Rank in the declared space (discovered landmarks dissolve into the
    declared interval containing them)."""
    eff = space.effective_landmarks
    deff = declared.effective_landmarks
    i = rank // 2
    if rank % 2 == 0:
        if eff[i] in deff:
            return 2 * deff.index(eff[i])
        upto = i
    else:
        upto = i + 1
    for k in range(upto - 1, -1, -1):
        if eff[k] in deff:
            return 2 * deff.index(eff[k]) + 1
    raise ValueError(f"rank {rank} has no declared landmark below it")


def _events(
    state: QState,
    frame: CompiledModel,
    pairs: tuple[tuple[int, int], ...],
    declared: tuple[QuantitySpace, ...],
) -> list[tuple[tuple, tuple[str, str, bool]]]:
    """The grid-line crossings a point state makes, as (key, position).

    Key = (pair, which coordinate is on the line, line landmark, crossing
    direction, declared segment of the position coordinate)."""
    if state.time is not TimeTag.POINT:
        return []
    out = []
    for p, (xi, yi) in enumerate(pairs):
        for line_i, pos_i in ((yi, xi), (xi, yi)):
            ql = state[frame.var_order[line_i]]
            if ql.mag % 2 != 0 or ql.dir not in _MOVING:
                continue
            line_name = frame.spaces[line_i].effective_landmarks[ql.mag // 2]
            if line_name in (NEG_INF, POS_INF):
                continue
            qp = state[frame.var_order[pos_i]]
            key = (
                p,
                line_i,
                line_name,
                int(ql.dir),
                _declared_rank(frame.spaces[pos_i], declared[pos_i], qp.mag),
            )
            out.append((key, _bounds(frame.spaces[pos_i], qp.mag)))
    return out


def _rel(a: tuple, b: tuple, space: QuantitySpace) -> str:
    """Provable order of two positions in a common (descendant) frame."""
    eff = space.effective_landmarks
    alo, ahi = 2 * eff.index(a[0]), 2 * eff.index(a[1])
    blo, bhi = 2 * eff.index(b[0]), 2 * eff.index(b[1])
    if a[2] and b[2]:
        return "eq" if alo == blo else ("lt" if alo < blo else "gt")
    if ahi < blo or ahi == blo:  # openness of an interval end supplies strictness
        return "lt"
    if bhi < alo or bhi == alo:
        return "gt"
    return "unknown"


def _sequence_flags(seq: list[tuple], space: QuantitySpace) -> tuple[bool, bool, bool]:
    """(monotone_ok, all_equal_ok, repeats): can this crossing sequence lie
    on a non-self-intersecting curve — interpretable as all-equal, strictly
    increasing, or strictly decreasing under the provable comparisons — and
    does it provably revisit an exact crossing point?"""
    eq_ok = inc_ok = dec_ok = True
    repeats = False
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            r = _rel(seq[i], seq[j], space)
            if r == "eq":
                inc_ok = dec_ok = False
                repeats = True
            elif r == "lt":  # later crossing farther out
                eq_ok = dec_ok = False
            elif r == "gt":  # later crossing closer in
                eq_ok = inc_ok = False
    return (eq_ok or inc_ok or dec_ok, eq_ok, repeats)


def admits(
    nodes: dict,
    parent,
    state: QState,
    frame: CompiledModel,
    pairs: tuple[tuple[int, int], ...],
    declared: tuple[QuantitySpace, ...],
) -> bool:
    """May ``state`` extend the path ending at ``parent``?

    Only sequences a new crossing extends are re-checked; crossing
    positions are compared in the candidate's frame, which (frames only
    grow along a branch) contains every ancestor's landmark names.
    """
    new = _events(state, frame, pairs, declared)
    if not new:
        return True
    # a new crossing can interact with every group of its pair (via the
    # closure rule), so collect full histories for the touched pairs
    touched = {k[0] for k, _ in new}
    history: dict[tuple, list] = {}
    chain = []
    node = parent
    while node is not None:
        chain.append(node)
        node = nodes[node.parent] if node.parent is not None else None
    for anc in reversed(chain):  # root -> parent order
        for k, pos in _events(anc.state, anc.model, pairs, declared):
            if k[0] in touched:
                history.setdefault(k, []).append(pos)
    for k, pos in new:
        history.setdefault(k, []).append(pos)

    flags: dict[tuple, tuple[bool, bool, bool]] = {}
    for k, seq in history.items():
        p, line_i = k[0], k[1]
        xi, yi = pairs[p]
        pos_i = xi if line_i == yi else yi
        flags[k] = _sequence_flags(seq, frame.spaces[pos_i])
        if not flags[k][0]:  # non-monotone returns to one transversal
            return False
    for p in touched:
        # closure: revisiting an exact crossing point closes the orbit
        # (the pair determines the flow), so every crossing group of the
        # pair must then be interpretable as all-equal
        if any(k[0] == p and f[2] for k, f in flags.items()):
            if any(k[0] == p and not f[1] for k, f in flags.items()):
                return False
    return True
