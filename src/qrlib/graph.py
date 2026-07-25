"""Small dependency-free graph algorithms.

Used by ``qrlib.analysis`` over behavior graphs (and later by envisionment
modes). Graphs are plain adjacency mappings ``node -> iterable of nodes``;
node identity is whatever the caller uses (behavior graphs use int ids).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

__all__ = ["reachable", "tarjan_scc", "representative_cycles"]


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


def representative_cycles(
    adj: Mapping[Hashable, Iterable[Hashable]],
) -> tuple[tuple[Hashable, ...], ...]:
    """One deterministic concrete loop from every recurrent component.

    Each returned path starts at its re-entry node and ends at a node with an
    edge back to that start; the start is not repeated. A self-loop is a
    one-node path. This compact representation is bounded by graph size even
    when the number of simple paths through a recurrent component is
    exponential.
    """
    normalized = {
        node: tuple(sorted(targets, key=repr))
        for node, targets in adj.items()
    }
    out: list[tuple[Hashable, ...]] = []
    components = [
        set(component)
        for component in tarjan_scc(normalized)
        if len(component) > 1
        or (
            len(component) == 1
            and next(iter(component)) in normalized.get(next(iter(component)), ())
        )
    ]
    components.sort(key=lambda component: repr(min(component, key=repr)))
    for component in components:
        entry = min(component, key=repr)
        if entry in normalized.get(entry, ()):
            out.append((entry,))
            continue
        queue = [entry]
        previous: dict[Hashable, Hashable | None] = {entry: None}
        closer: Hashable | None = None
        for node in queue:
            if node != entry and entry in normalized.get(node, ()):
                closer = node
                break
            for successor in normalized.get(node, ()):
                if successor in component and successor not in previous:
                    previous[successor] = node
                    queue.append(successor)
        if closer is None:  # defensive: a non-trivial SCC must close
            continue
        path = []
        node: Hashable | None = closer
        while node is not None:
            path.append(node)
            node = previous[node]
        out.append(tuple(reversed(path)))
    return tuple(out)
