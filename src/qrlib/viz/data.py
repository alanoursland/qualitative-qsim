"""Render-agnostic visualization data that hosts can restyle in their own
stacks.

- :func:`timeline_bands`: one behavior as per-variable bands over
  qualitative time — magnitude rank/label + direction per state, with
  numeric bounds and time bounds attached when a
  :class:`~qrlib.semiquant.SemiQuantResult` is supplied (the
  envelope-vs-trajectory plotting data).
- :func:`tree_layout`: node positions for drawing a behavior graph —
  layered by depth, ordered by a first-visit tidy pass (leaves get
  successive x slots, internals center over their children).
"""

from __future__ import annotations

from ..behavior import Behavior, BehaviorGraph

__all__ = ["timeline_bands", "tree_layout"]


def timeline_bands(
    graph: BehaviorGraph, behavior: Behavior, *, refinement=None
) -> dict:
    """Per-variable qualitative timeline of one behavior, as plain data."""
    nodes = [graph.nodes[i] for i in behavior.node_ids]
    variables = []
    for vi, name in enumerate(graph.var_order):
        segments = []
        for si, node in enumerate(nodes):
            qv = node.state[name]
            space = node.model.spaces[vi]
            seg = {
                "state": si,
                "time_tag": node.state.time.value,
                "region": node.region,
                "rank": qv.mag,
                "num_ranks": space.num_ranks,
                "label": space.describe(qv.mag),
                "dir": qv.dir.name.lower(),
            }
            if refinement is not None:
                iv = refinement.bounds[si][name]
                t = refinement.times[si]
                seg["bounds"] = [iv.lo, iv.hi]
                seg["time"] = [t.lo, t.hi]
            segments.append(seg)
        variables.append({"name": name, "segments": segments})
    return {
        "model": nodes[0].model.name,
        "terminal": behavior.terminal.value,
        "regions": [n.region for n in nodes],
        "variables": variables,
    }


def tree_layout(graph: BehaviorGraph) -> dict:
    """Positions for drawing a behavior graph: ``y`` = depth, ``x`` from a
    first-visit tidy ordering. Cross/back edges (envisionment merges,
    cycle closures) are included as edges but do not affect positions."""
    xs: dict[int, float] = {}
    visited: set[int] = set()
    counter = iter(range(len(graph.nodes) + 1))

    def place(nid: int) -> None:
        visited.add(nid)
        node = graph.nodes[nid]
        child_xs = []
        for child in node.children:
            if child in visited:
                continue
            place(child)
            child_xs.append(xs[child])
        xs[nid] = (
            sum(child_xs) / len(child_xs) if child_xs else float(next(counter))
        )

    place(graph.root)
    for nid in graph.nodes:  # nodes unreachable via first-visit (defensive)
        if nid not in xs:
            xs[nid] = float(next(counter))

    nodes = [
        {
            "id": n.id,
            "x": xs[n.id],
            "y": n.depth,
            "label": graph.describe_node(n),
            "region": n.region,
            "time_tag": n.state.time.value,
            "terminal": n.terminal.value if n.terminal else None,
        }
        for n in graph.nodes.values()
    ]
    edges = [[n.id, c] for n in graph.nodes.values() for c in n.children]
    cycle_edges = [
        [n.id, n.cycle_target]
        for n in graph.nodes.values()
        if n.cycle_target is not None
    ]
    return {"nodes": nodes, "edges": edges, "cycle_edges": cycle_edges}
