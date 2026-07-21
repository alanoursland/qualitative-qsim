"""Queries over behavior graphs.

Host-facing questions (docs/host-integration.md, Surface 4) answered with
plain data and the small algorithms in ``qrlib.graph`` — no external graph
dependencies. All functions take a :class:`~qrlib.behavior.BehaviorGraph`.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

from ..behavior import BehaviorGraph, TerminalClass
from ..state import QState

__all__ = [
    "terminal_census",
    "quiescent_states",
    "cycles",
    "find_states",
]


def terminal_census(graph: BehaviorGraph) -> dict[TerminalClass, int]:
    """How many terminals of each class the graph contains."""
    return dict(
        Counter(n.terminal for n in graph.nodes.values() if n.terminal is not None)
    )


def quiescent_states(graph: BehaviorGraph) -> tuple[int, ...]:
    """Node ids of quiescent terminals — the equilibrium candidates.

    Keyed by node id; the states themselves (``graph.nodes[i].state``) carry
    the variable magnitudes a host can cross-link with its own equilibrium
    analyses.
    """
    return tuple(
        n.id
        for n in graph.nodes.values()
        if n.terminal is TerminalClass.QUIESCENT
    )


def cycles(graph: BehaviorGraph) -> tuple[tuple[int, ...], ...]:
    """The closed loops, as node-id paths from each cycle's re-entry point
    (the matched ancestor) down to the node that closed it."""
    out: list[tuple[int, ...]] = []
    for node in graph.nodes.values():
        if node.terminal is not TerminalClass.CYCLE or node.cycle_target is None:
            continue
        path: list[int] = []
        nid: int | None = node.id
        while nid is not None:
            path.append(nid)
            if nid == node.cycle_target:
                break
            nid = graph.nodes[nid].parent
        out.append(tuple(reversed(path)))
    return tuple(out)


def find_states(
    graph: BehaviorGraph, predicate: Callable[[QState], bool]
) -> tuple[int, ...]:
    """Node ids whose states satisfy ``predicate``."""
    return tuple(n.id for n in graph.nodes.values() if predicate(n.state))
