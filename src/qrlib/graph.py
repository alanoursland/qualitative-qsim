"""Small dependency-free graph algorithms.

Used by ``qrlib.analysis`` over behavior graphs (and later by envisionment
modes). Graphs are plain adjacency mappings ``node -> iterable of nodes``;
node identity is whatever the caller uses (behavior graphs use int ids).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

__all__ = ["reachable", "tarjan_scc"]


def reachable(adj: Mapping[Hashable, Iterable[Hashable]], starts: Iterable[Hashable]) -> set:
    """Nodes reachable from ``starts`` (inclusive) by following edges."""
    seen: set = set()
    stack = list(starts)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, ()))
    return seen


def tarjan_scc(adj: Mapping[Hashable, Iterable[Hashable]]) -> list[list[Hashable]]:
    """Strongly connected components (iterative Tarjan), in reverse
    topological order. Nodes appearing only as edge targets are included."""
    nodes = set(adj)
    for targets in adj.values():
        nodes.update(targets)

    index: dict = {}
    lowlink: dict = {}
    on_stack: set = set()
    stack: list = []
    sccs: list[list[Hashable]] = []
    counter = 0

    for root in sorted(nodes, key=repr):
        if root in index:
            continue
        work = [(root, iter(sorted(adj.get(root, ()), key=repr)))]
        index[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, it = work[-1]
            advanced = False
            for succ in it:
                if succ not in index:
                    index[succ] = lowlink[succ] = counter
                    counter += 1
                    stack.append(succ)
                    on_stack.add(succ)
                    work.append((succ, iter(sorted(adj.get(succ, ()), key=repr))))
                    advanced = True
                    break
                if succ in on_stack:
                    lowlink[node] = min(lowlink[node], index[succ])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index[node]:
                comp = []
                while True:
                    n = stack.pop()
                    on_stack.discard(n)
                    comp.append(n)
                    if n == node:
                        break
                sccs.append(comp)
    return sccs
