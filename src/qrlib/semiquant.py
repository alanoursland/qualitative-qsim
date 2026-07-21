"""Q2-style semi-quantitative refinement of qualitative behaviors.

Landmarks may carry numeric knowledge (exact ``value`` or ``(lo, hi)``
bounds — in the schema since phase 2) and M+/M- constraints may carry
numeric **envelopes** (``lower(x) <= f(x) <= upper(x)``). Given one
behavior from a QSIM graph, :func:`refine` propagates interval constraints
to a fixpoint and returns **guaranteed bounds** on every variable at every
qualitative state, plus bounds on the *times* of the distinguished time
points (mean-value-theorem propagation through ``DERIV`` constraints — the
classic Q2 result: "the tank reaches FULL between t=1.0 and t=1.92").

A behavior whose intervals empty out is **numerically impossible** and is
refuted with the state/variable where the contradiction surfaced —
:func:`feasible_behaviors` prunes a whole result this way. Everything is
conservative: intervals only shrink through sound deductions, so a real
trajectory of any concrete instance consistent with the annotations always
lies inside the reported bounds (and its behavior is never refuted).

Propagation channels, per fixpoint pass:

- within-state: ``ADD``/``MINUS``/``MULT`` interval arithmetic, and
  envelope forward/inverse maps for M+/M- (region-aware: only the state's
  active constraints fire);
- across states: continuity + monotonic persistence (a variable rising
  over an interval is bracketed by its endpoint values), region-entry
  point pairs share values and times, ``CONSTANT`` variables intersect
  across their active states;
- time: for an interval-state between two points, ``DERIV(x, y)`` gives
  ``delta_x = y_bar * delta_t`` for some ``y_bar`` in y's range (MVT),
  refining endpoint values, durations, and the accumulated time points.
  Asymptotic approaches correctly yield unbounded arrival times (y's
  range contains 0).

Outputs are plain data (:meth:`SemiQuantResult.to_dict`): per-state
variable bands + time bounds, ready to plot against numeric trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isnan
from typing import Callable, Mapping, Sequence

from .behavior import Behavior, BehaviorGraph, SimResult
from .constraints import Constraint
from .quantity import QuantitySpace, Qdir
from .state import TimeTag

__all__ = [
    "Interval",
    "Envelope",
    "SemiQuantResult",
    "refine",
    "refine_all",
    "feasible_behaviors",
]


@dataclass(frozen=True)
class Interval:
    """A closed numeric interval; ``lo > hi`` means empty."""

    lo: float
    hi: float

    @property
    def empty(self) -> bool:
        return self.lo > self.hi

    def intersect(self, other: "Interval") -> "Interval":
        return Interval(max(self.lo, other.lo), min(self.hi, other.hi))

    def hull(self, other: "Interval") -> "Interval":
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def add(self, other: "Interval") -> "Interval":
        return Interval(
            _guard(self.lo + other.lo, -inf), _guard(self.hi + other.hi, inf)
        )

    def neg(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def sub(self, other: "Interval") -> "Interval":
        return self.add(other.neg())

    def mul(self, other: "Interval") -> "Interval":
        # standard interval-arithmetic convention: 0 * inf = 0 (infinite
        # endpoints denote unboundedness of real values, never attained)
        ps = [
            _guard(self.lo * other.lo, 0.0),
            _guard(self.lo * other.hi, 0.0),
            _guard(self.hi * other.lo, 0.0),
            _guard(self.hi * other.hi, 0.0),
        ]
        return Interval(min(ps), max(ps))

    def div(self, other: "Interval") -> "Interval | None":
        """None when the divisor contains zero (no refinement possible)."""
        if other.lo <= 0 <= other.hi:
            return None
        qs = [
            self.lo / other.lo,
            self.lo / other.hi,
            self.hi / other.lo,
            self.hi / other.hi,
        ]
        return Interval(
            min(_guard(q, -inf) for q in qs), max(_guard(q, inf) for q in qs)
        )


def _guard(v: float, fallback: float) -> float:
    return fallback if isnan(v) else v


EVERYTHING = Interval(-inf, inf)
NON_NEGATIVE = Interval(0.0, inf)


@dataclass(frozen=True)
class Envelope:
    """Numeric envelope of an M+/M- constraint's unknown monotone function:
    ``lower(x) <= f(x) <= upper(x)`` pointwise. ``inv_lower``/``inv_upper``
    are the inverses of ``lower``/``upper`` when available (they enable
    propagating x-bounds from y-bounds). Callables are only evaluated on
    finite endpoints."""

    lower: Callable[[float], float]
    upper: Callable[[float], float]
    inv_lower: Callable[[float], float] | None = None
    inv_upper: Callable[[float], float] | None = None


@dataclass(frozen=True)
class SemiQuantResult:
    """Guaranteed numeric refinement of one qualitative behavior.

    ``bounds[i][var]`` bounds the variable over state ``i`` (a band for
    interval-states, a value range at point-states); ``times[i]`` bounds
    when state ``i`` happens (point states: the instant; interval states:
    the span's hull). ``feasible=False`` means the behavior is numerically
    impossible under the annotations — ``refuted_at``/``reason`` localize
    the contradiction."""

    feasible: bool
    bounds: tuple[dict[str, Interval], ...]
    times: tuple[Interval, ...]
    refuted_at: int | None = None
    reason: str | None = None
    passes: int = 0

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "refuted_at": self.refuted_at,
            "reason": self.reason,
            "states": [
                {
                    "time": [t.lo, t.hi],
                    "vars": {v: [iv.lo, iv.hi] for v, iv in b.items()},
                }
                for b, t in zip(self.bounds, self.times)
            ],
        }


def _landmark_ranges(space: QuantitySpace) -> list[Interval]:
    """Numeric range of each *effective* landmark, tightened by ordering
    (landmarks are strictly increasing, so bounds chain monotonically)."""
    named = []
    for lm in space.landmarks:
        if lm.value is not None:
            named.append(Interval(lm.value, lm.value))
        else:
            named.append(
                Interval(
                    lm.lo if lm.lo is not None else -inf,
                    lm.hi if lm.hi is not None else inf,
                )
            )
    for i in range(1, len(named)):
        named[i] = Interval(max(named[i].lo, named[i - 1].lo), named[i].hi)
    for i in range(len(named) - 2, -1, -1):
        named[i] = Interval(named[i].lo, min(named[i].hi, named[i + 1].hi))
    out = []
    if space.lower_unbounded:
        out.append(Interval(-inf, -inf))
    out.extend(named)
    if space.upper_unbounded:
        out.append(Interval(inf, inf))
    return out


def _mag_range(space: QuantitySpace, rank: int) -> Interval:
    lms = _landmark_ranges(space)
    if rank % 2 == 0:
        return lms[rank // 2]
    return Interval(lms[rank // 2].lo, lms[rank // 2 + 1].hi)


def refine(
    graph: BehaviorGraph,
    behavior: Behavior,
    *,
    envelopes: Mapping[Constraint, Envelope] | None = None,
    t0: float = 0.0,
    max_passes: int = 30,
) -> SemiQuantResult:
    """Propagate numeric annotations along one behavior to a fixpoint."""
    env = dict(envelopes or {})
    nodes = [graph.nodes[i] for i in behavior.node_ids]
    names = graph.var_order
    n = len(nodes)

    bounds: list[dict[str, Interval]] = []
    for node in nodes:
        frame = node.model
        bounds.append(
            {
                name: _mag_range(frame.spaces[vi], node.state[name].mag)
                for vi, name in enumerate(names)
            }
        )
    point_time: dict[int, Interval] = {
        si: (Interval(t0, t0) if si == 0 else Interval(t0, inf))
        for si, node in enumerate(nodes)
        if node.state.time is TimeTag.POINT
    }
    refuted: tuple[int, str] | None = None

    def narrow(si: int, var: str, iv: Interval, why: str) -> bool:
        nonlocal refuted
        new = bounds[si][var].intersect(iv)
        if new == bounds[si][var]:
            return False
        bounds[si][var] = new
        if new.empty and refuted is None:
            refuted = (si, f"{var} at state {si} emptied via {why}")
        return True

    def narrow_time(si: int, iv: Interval) -> bool:
        nonlocal refuted
        new = point_time[si].intersect(iv)
        if new == point_time[si]:
            return False
        point_time[si] = new
        if new.empty and refuted is None:
            refuted = (si, f"time of state {si} emptied")
        return True

    passes = 0
    for _ in range(max_passes):
        passes += 1
        changed = False

        # --- within-state constraint propagation ---------------------------
        for si, node in enumerate(nodes):
            frame = node.model
            b = bounds[si]
            for cc in frame.constraints_of(node.region):
                vs = [names[vi] for vi in cc.vars]
                if cc.kind == "add":
                    x, y, z = (b[v] for v in vs)
                    changed |= narrow(si, vs[2], x.add(y), "add")
                    changed |= narrow(si, vs[0], b[vs[2]].sub(b[vs[1]]), "add")
                    changed |= narrow(si, vs[1], b[vs[2]].sub(b[vs[0]]), "add")
                elif cc.kind == "minus":
                    changed |= narrow(si, vs[1], b[vs[0]].neg(), "minus")
                    changed |= narrow(si, vs[0], b[vs[1]].neg(), "minus")
                elif cc.kind == "mult":
                    x, y, z = (b[v] for v in vs)
                    changed |= narrow(si, vs[2], x.mul(y), "mult")
                    q = b[vs[2]].div(b[vs[1]])
                    if q is not None:
                        changed |= narrow(si, vs[0], q, "mult")
                    q = b[vs[2]].div(b[vs[0]])
                    if q is not None:
                        changed |= narrow(si, vs[1], q, "mult")
                elif cc.kind in ("mplus", "mminus") and cc.source in env:
                    e = env[cc.source]
                    x, y = b[vs[0]], b[vs[1]]
                    if cc.kind == "mplus":
                        ylo = e.lower(x.lo) if x.lo > -inf else -inf
                        yhi = e.upper(x.hi) if x.hi < inf else inf
                    else:  # decreasing: extremes swap sides
                        ylo = e.lower(x.hi) if x.hi < inf else -inf
                        yhi = e.upper(x.lo) if x.lo > -inf else inf
                    changed |= narrow(si, vs[1], Interval(ylo, yhi), "envelope")
                    if e.inv_lower is not None and e.inv_upper is not None:
                        y = b[vs[1]]
                        if cc.kind == "mplus":
                            xlo = e.inv_upper(y.lo) if y.lo > -inf else -inf
                            xhi = e.inv_lower(y.hi) if y.hi < inf else inf
                        else:
                            xlo = e.inv_lower(y.hi) if y.hi < inf else -inf
                            xhi = e.inv_upper(y.lo) if y.lo > -inf else inf
                        changed |= narrow(
                            si, vs[0], Interval(xlo, xhi), "envelope"
                        )

        # --- continuity / persistence across adjacent states ---------------
        for si in range(n - 1):
            a, c = nodes[si], nodes[si + 1]
            if (
                a.state.time is TimeTag.POINT
                and c.state.time is TimeTag.POINT
            ):
                # instantaneous region entry: values and times carry over
                for v in names:
                    changed |= narrow(si, v, bounds[si + 1][v], "entry")
                    changed |= narrow(si + 1, v, bounds[si][v], "entry")
                changed |= narrow_time(si, point_time[si + 1])
                changed |= narrow_time(si + 1, point_time[si])
                continue
            ii, pi = (si, si + 1) if a.state.time is TimeTag.INTERVAL else (si + 1, si)
            interval_node = nodes[ii]
            for v in names:
                d = interval_node.state[v].dir
                bi, bp = bounds[ii][v], bounds[pi][v]
                if d is Qdir.STD:
                    changed |= narrow(ii, v, bp, "steady")
                    changed |= narrow(pi, v, bi, "steady")
                elif d is Qdir.INC:
                    if pi > ii:  # point closes the interval from above
                        changed |= narrow(pi, v, Interval(bi.lo, inf), "rising")
                        changed |= narrow(ii, v, Interval(-inf, bp.hi), "rising")
                    else:  # point opens the interval from below
                        changed |= narrow(pi, v, Interval(-inf, bi.hi), "rising")
                        changed |= narrow(ii, v, Interval(bp.lo, inf), "rising")
                elif d is Qdir.DEC:
                    if pi > ii:
                        changed |= narrow(pi, v, Interval(-inf, bi.hi), "falling")
                        changed |= narrow(ii, v, Interval(bp.lo, inf), "falling")
                    else:
                        changed |= narrow(pi, v, Interval(bi.lo, inf), "falling")
                        changed |= narrow(ii, v, Interval(-inf, bp.hi), "falling")

        # --- CONSTANT variables intersect across their active states -------
        const_states: dict[str, list[int]] = {}
        for si, node in enumerate(nodes):
            for cc in node.model.constraints_of(node.region):
                if cc.kind == "constant":
                    const_states.setdefault(names[cc.vars[0]], []).append(si)
        for v, sis in const_states.items():
            common = EVERYTHING
            for si in sis:
                common = common.intersect(bounds[si][v])
            for si in sis:
                changed |= narrow(si, v, common, "constant")

        # --- time propagation (MVT through DERIV) --------------------------
        pts = sorted(point_time)
        for k in range(len(pts) - 1):
            pi, pj = pts[k], pts[k + 1]
            changed |= narrow_time(
                pj, Interval(point_time[pi].lo, inf)
            )
            changed |= narrow_time(
                pi, Interval(-inf, point_time[pj].hi)
            )
            if pj != pi + 2:
                continue  # not sandwiching a single interval-state
            ii = pi + 1
            dt = point_time[pj].sub(point_time[pi]).intersect(NON_NEGATIVE)
            for cc in nodes[ii].model.constraints_of(nodes[ii].region):
                if cc.kind != "deriv":
                    continue
                xv, yv = names[cc.vars[0]], names[cc.vars[1]]
                ys = bounds[ii][yv]
                dx = bounds[pj][xv].sub(bounds[pi][xv]).intersect(ys.mul(dt))
                changed |= narrow(pj, xv, bounds[pi][xv].add(ys.mul(dt)), "mvt")
                changed |= narrow(pi, xv, bounds[pj][xv].sub(ys.mul(dt)), "mvt")
                q = dx.div(ys)
                if q is not None:
                    dt = dt.intersect(q.intersect(NON_NEGATIVE))
            changed |= narrow_time(pj, point_time[pi].add(dt))
            changed |= narrow_time(pi, point_time[pj].sub(dt))

        if refuted is not None or not changed:
            break

    times: list[Interval] = []
    for si, node in enumerate(nodes):
        if si in point_time:
            times.append(point_time[si])
        else:
            lo = point_time[si - 1].lo if si - 1 in point_time else t0
            hi = point_time[si + 1].hi if si + 1 in point_time else inf
            times.append(Interval(lo, hi))

    if refuted is not None:
        si, reason = refuted
        return SemiQuantResult(
            False, tuple(bounds), tuple(times), si, reason, passes
        )
    return SemiQuantResult(True, tuple(bounds), tuple(times), passes=passes)


def refine_all(
    result: SimResult,
    *,
    envelopes: Mapping[Constraint, Envelope] | None = None,
    t0: float = 0.0,
    max_passes: int = 30,
) -> tuple[tuple[Behavior, SemiQuantResult], ...]:
    """Refine every behavior of a simulation result."""
    return tuple(
        (
            b,
            refine(
                result.graph,
                b,
                envelopes=envelopes,
                t0=t0,
                max_passes=max_passes,
            ),
        )
        for b in result.behaviors()
    )


def feasible_behaviors(
    result: SimResult,
    *,
    envelopes: Mapping[Constraint, Envelope] | None = None,
    t0: float = 0.0,
    max_passes: int = 30,
) -> tuple[Behavior, ...]:
    """The behaviors that survive numeric refutation — the Q2 pruning."""
    return tuple(
        b
        for b, r in refine_all(
            result, envelopes=envelopes, t0=t0, max_passes=max_passes
        )
        if r.feasible
    )
