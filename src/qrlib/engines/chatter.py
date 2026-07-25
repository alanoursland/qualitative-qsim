"""Structural chatter detection (after Clancy & Kuipers).

Research lineage: Clancy and Kuipers (1997).

A variable **chatters** when nothing pins its direction of change: within
one qualitative magnitude its qdir can wiggle dec/std/inc arbitrarily,
multiplying behaviors that differ in nothing observable. Chatter is a
*structural* property of the active constraints, detectable before any
state explodes:

- ``M+``/``M-``/``MINUS`` tie two variables' directions rigidly (equal or
  opposite); direction-rigid groups form equivalence classes.
- ``ADD``/``MULT`` tie directions only through ambiguous sign arithmetic —
  no rigidity — *except* when an operand is declared ``Constant`` (its
  direction is pinned std), which makes the other two operands rigid
  (two constant operands pin the third outright).
- A class is **anchored** — cannot chatter — when it contains a state
  variable (the ``x`` of a ``Deriv(x, y)``: its direction is the *sign of
  y's magnitude*, so it changes only at the distinguished event of ``y``
  crossing zero) or a ``Constant`` variable.

Everything in an unanchored class is a chatter candidate. Candidacy is a
*permission*, not a sentence: the dynamic side (``SimConfig
.dynamic_chatter``, in the engine) abstracts a candidate's direction only
at states where the wiggle branching actually appears — successors
identical except for candidate directions are merged with those
directions projected to ``Qdir.IGN``, the same sound abstraction as a
hand-written ``ignore_qdir`` but applied exactly where, and only where,
chatter manifests. Where filtering forces a candidate's direction, it
stays concrete.

The analysis is per operating region (each region activates its own
constraint subset, so the same variable can chatter in one region and be
pinned in another).
"""

from __future__ import annotations

from ..model import CompiledModel, Model

__all__ = ["candidates", "chatter_variables"]

_RIGID = ("mplus", "mminus", "minus")


def candidates(frame: CompiledModel, region: str) -> frozenset[int]:
    """Indices of variables that can chatter under ``region``'s active
    constraints."""
    active = frame.constraints_of(region)
    n = len(frame.var_order)
    parent = list(range(n))

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    constants = {c.vars[0] for c in active if c.kind == "constant"}
    anchored = set(constants)
    for c in active:
        if c.kind in _RIGID:
            union(c.vars[0], c.vars[1])
        elif c.kind == "deriv":
            anchored.add(c.vars[0])  # dir(x) = sign of y's magnitude
        elif c.kind in ("add", "mult"):
            free = [v for v in c.vars if v not in constants]
            if len(free) == 2:  # constant operand: the rest move in lockstep
                union(free[0], free[1])
            elif len(free) == 1:  # two constants pin the third
                anchored.add(free[0])

    anchored_roots = {find(v) for v in anchored}
    return frozenset(v for v in range(n) if find(v) not in anchored_roots)


def chatter_variables(
    model: Model | CompiledModel, region: str | None = None
) -> tuple[str, ...]:
    """Name-level convenience: which variables can chatter (in ``region``,
    or the initial region by default)."""
    frame = model.compile() if isinstance(model, Model) else model
    idx = candidates(frame, region or frame.initial_region)
    return tuple(v for i, v in enumerate(frame.var_order) if i in idx)
