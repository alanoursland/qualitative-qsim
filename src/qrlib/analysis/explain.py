"""Behavior explanation: structured step records + prose narration.

Every state in a behavior is symbolic, so narration is cheap and faithful
(docs/host-integration.md, cross-cutting conventions): :func:`explain`
derives, for each step of a behavior, what changed per variable (reaching a
landmark, becoming steady at a discovered value, starting to rise/fall,
crossing into another operating region) as structured :class:`Event`
records; :func:`narrate` renders them as prose. Hosts embed the structured
form in reports and restyle the text freely.

Changes are compared through each node's own frame (descriptions, not raw
ranks — discovered landmarks shift ranks between frames but keep names).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..behavior import Behavior, BehaviorGraph, Node, TerminalClass
from ..quantity import Qdir
from ..state import TimeTag

__all__ = ["Event", "Step", "explain", "narrate"]

_DIRWORD = {
    Qdir.INC: "rising",
    Qdir.STD: "steady",
    Qdir.DEC: "falling",
    Qdir.IGN: "direction untracked",
}

_TERMINAL_PHRASE = {
    TerminalClass.QUIESCENT: "the system is in equilibrium and can remain there indefinitely",
    TerminalClass.CYCLE: "the behavior cycles",
    TerminalClass.DIVERGENT: "a variable grows without bound (the t→∞ limit)",
    TerminalClass.REGION_EXIT: "the system leaves the model's domain of validity",
    TerminalClass.DEADEND: "no consistent continuation exists (this branch is spurious)",
    TerminalClass.TRUNCATED: "exploration stopped at a resource limit",
}


@dataclass(frozen=True)
class Event:
    """One variable's change at one step of a behavior."""

    variable: str
    kind: str  # "reaches" | "steadies" | "starts" | "resumes" | "changes"
    detail: str  # human-readable, e.g. "reaches FULL, still rising"


@dataclass(frozen=True)
class Step:
    """One state of a behavior, described relative to its predecessor."""

    index: int
    time_tag: str  # "point" | "interval"
    region: str
    summary: str
    events: tuple[Event, ...]
    minted: tuple[str, ...]  # landmarks discovered at this step
    terminal: str | None = None


def explain(graph: BehaviorGraph, behavior: Behavior) -> tuple[Step, ...]:
    """Structured step-by-step account of one behavior."""
    nodes = [graph.nodes[i] for i in behavior.node_ids]
    steps: list[Step] = []

    first = nodes[0]
    initial = ", ".join(
        f"{name} {_phrase(first, vi, name, graph)}"
        for vi, name in enumerate(graph.var_order)
    )
    steps.append(
        Step(
            0,
            first.state.time.value,
            first.region,
            _finish(f"initially, {initial}", nodes, 0, behavior),
            (),
            (),
            _terminal_of(nodes, 0, behavior),
        )
    )

    for si in range(1, len(nodes)):
        prev, cur = nodes[si - 1], nodes[si]
        minted = _minted(prev, cur, graph)
        events = tuple(
            e
            for vi, name in enumerate(graph.var_order)
            if (e := _event(prev, cur, vi, name, graph, minted)) is not None
        )
        crossing = cur.region != prev.region
        if crossing:
            prefix = f"crossing into {cur.region}, "
        elif cur.state.time is TimeTag.INTERVAL:
            prefix = "then, over an interval, "
        else:
            prefix = "at the next instant, "
        body = "; ".join(f"{e.variable} {e.detail}" for e in events)
        if not body:
            body = "nothing changes qualitatively"
        steps.append(
            Step(
                si,
                cur.state.time.value,
                cur.region,
                _finish(prefix + body, nodes, si, behavior),
                events,
                minted,
                _terminal_of(nodes, si, behavior),
            )
        )
    return tuple(steps)


def narrate(graph: BehaviorGraph, behavior: Behavior) -> str:
    """Prose narration of one behavior, built from :func:`explain`."""
    steps = explain(graph, behavior)
    name = graph.nodes[behavior.node_ids[0]].model.name
    header = (
        f"Behavior of {name!r}: {len(steps)} states, "
        f"ending in {behavior.terminal.value.replace('_', ' ')}."
    )
    lines = [header]
    for step in steps:
        text = step.summary[0].upper() + step.summary[1:]
        lines.append(f"  {step.index}. {text}")
    return "\n".join(lines)


# --- derivation ------------------------------------------------------------


def _phrase(node: Node, vi: int, name: str, graph: BehaviorGraph) -> str:
    qv = node.state[name]
    space = node.model.spaces[vi]
    where = "at" if qv.mag % 2 == 0 else "in"
    return f"{where} {space.describe(qv.mag)}, {_DIRWORD[qv.dir]}"


def _minted(prev: Node, cur: Node, graph: BehaviorGraph) -> tuple[str, ...]:
    if prev.model == cur.model:
        return ()
    out = []
    for vi in range(len(graph.var_order)):
        before = set(prev.model.spaces[vi].names)
        out.extend(n for n in cur.model.spaces[vi].names if n not in before)
    return tuple(out)


def _event(
    prev: Node,
    cur: Node,
    vi: int,
    name: str,
    graph: BehaviorGraph,
    minted: tuple[str, ...],
) -> Event | None:
    pq, cq = prev.state[name], cur.state[name]
    p_space, c_space = prev.model.spaces[vi], cur.model.spaces[vi]
    p_lab, c_lab = p_space.describe(pq.mag), c_space.describe(cq.mag)
    if p_lab == c_lab and pq.dir == cq.dir:
        return None
    p_at, c_at = pq.mag % 2 == 0, cq.mag % 2 == 0

    if not p_at and c_at:  # arrived at a landmark
        if c_lab in minted:
            return Event(
                name, "steadies", f"becomes steady at {c_lab} (a newly identified value)"
            )
        suffix = {
            Qdir.STD: ", holding steady",
            Qdir.INC: ", still rising",
            Qdir.DEC: ", still falling",
            Qdir.IGN: "",
        }[cq.dir]
        return Event(name, "reaches", f"reaches {c_lab}{suffix}")
    if p_at and not c_at:  # departed into an interval
        verb = {Qdir.INC: "rises", Qdir.DEC: "falls"}.get(cq.dir, "moves")
        return Event(name, "starts", f"{verb} into {c_lab}")
    where = "at" if c_at else "in"
    if p_lab == c_lab:  # direction change only
        if cq.dir is Qdir.STD:
            return Event(name, "steadies", f"levels off {where} {c_lab}")
        if pq.dir is Qdir.STD:
            return Event(
                name, "resumes", f"resumes {_DIRWORD[cq.dir]} {where} {c_lab}"
            )
        return Event(name, "changes", f"turns {_DIRWORD[cq.dir]} {where} {c_lab}")
    return Event(name, "changes", f"is {where} {c_lab}, {_DIRWORD[cq.dir]}")


def _terminal_of(nodes, si: int, behavior: Behavior) -> str | None:
    if si == len(nodes) - 1:
        return behavior.terminal.value
    return None


def _finish(summary: str, nodes, si: int, behavior: Behavior) -> str:
    if si != len(nodes) - 1:
        return summary
    phrase = _TERMINAL_PHRASE[behavior.terminal]
    if behavior.terminal is TerminalClass.CYCLE and behavior.cycle_target is not None:
        try:
            k = behavior.node_ids.index(behavior.cycle_target)
            phrase = f"the state repeats step {k}: {phrase}"
        except ValueError:
            pass
    return f"{summary} — {phrase}"
