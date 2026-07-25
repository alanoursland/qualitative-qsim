"""Queries over behavior graphs.

Host-facing questions are answered with plain data and the small algorithms
in ``qrlib.graph`` — no external graph dependencies. All functions take a
:class:`~qrlib.behavior.BehaviorGraph`.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

from ..behavior import BehaviorGraph, TerminalClass
from ..graph import representative_cycles
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
    down to the node whose edge closes it.

    Tree-mode cycles use explicit ``cycle_target`` annotations. Attainable
    envisionments encode recurrence as real graph edges, so their loops are
    discovered structurally from strongly connected components."""
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
    adjacency = {node.id: tuple(node.children) for node in graph.nodes.values()}
    out.extend(
        tuple(int(node) for node in loop)
        for loop in representative_cycles(adjacency)
    )
    return tuple(out)


def find_states(
    graph: BehaviorGraph, predicate: Callable[[QState], bool]
) -> tuple[int, ...]:
    """Node ids whose states satisfy ``predicate``."""
    return tuple(n.id for n in graph.nodes.values() if predicate(n.state))
