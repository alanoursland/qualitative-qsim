"""Behavior graphs and simulation results.

The output side of every engine: a :class:`BehaviorGraph` of
:class:`~qrlib.state.QState` nodes with terminal classifications, wrapped in
a :class:`SimResult` that records the config and statistics that produced it
(truncation and filtering are always reported, never silent — see
docs/host-integration.md, cross-cutting conventions).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from .quantity import QuantitySpace
from .state import QState, TimeTag

__all__ = [
    "TerminalClass",
    "SimStatus",
    "SimConfig",
    "Node",
    "Behavior",
    "BehaviorGraph",
    "SimResult",
]


class TerminalClass(Enum):
    """Why a behavior ends at a node.

    - ``QUIESCENT``: all directions steady — an equilibrium; the constant
      continuation is one complete behavior (an unstable equilibrium may
      *also* have departing children).
    - ``CYCLE``: the state matches an ancestor point state; the behavior
      closes into that ancestor (``Node.cycle_target``).
    - ``DIVERGENT``: some variable is at an infinite landmark (the state
      represents the limit t -> infinity).
    - ``REGION_EXIT``: a variable sits at a boundary landmark of its bounded
      quantity space with an outward direction — the system leaves the
      model's domain of validity (e.g. a tank overflowing its space).
    - ``DEADEND``: no consistent successor survived filtering. A dead end
      indicates the state itself is spurious (real behaviors continue), but
      it is reported rather than silently pruned.
    - ``TRUNCATED``: a resource limit stopped exploration here.
    """

    QUIESCENT = "quiescent"
    CYCLE = "cycle"
    DIVERGENT = "divergent"
    REGION_EXIT = "region_exit"
    DEADEND = "deadend"
    TRUNCATED = "truncated"


class SimStatus(Enum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"


@dataclass(frozen=True)
class SimConfig:
    """Engine configuration. Filter toggles exist for experimentation and
    testing; defaults are textbook QSIM behavior."""

    max_states: int = 500
    max_depth: int = 100
    no_change_filter: bool = True
    cycle_detection: bool = True
    infinity_filter: bool = True


@dataclass
class Node:
    """One node of a behavior graph (mutable during construction only)."""

    id: int
    state: QState
    parent: int | None
    depth: int
    children: list[int] = field(default_factory=list)
    terminal: TerminalClass | None = None
    cycle_target: int | None = None


@dataclass(frozen=True)
class Behavior:
    """A root-to-terminal path: one qualitative behavior."""

    node_ids: tuple[int, ...]
    states: tuple[QState, ...]
    terminal: TerminalClass
    cycle_target: int | None = None


@dataclass
class BehaviorGraph:
    """Directed graph over qualitative states, rooted at the initial state."""

    nodes: dict[int, Node]
    root: int
    var_order: tuple[str, ...]
    spaces: tuple[QuantitySpace, ...]

    def behaviors(self) -> tuple[Behavior, ...]:
        """All root-to-terminal paths, in deterministic (DFS) order.

        A node that is both terminal and expanded (an unstable equilibrium)
        contributes the terminated behavior *and* the extended ones.
        """
        out: list[Behavior] = []

        def walk(nid: int, path: tuple[int, ...]) -> None:
            node = self.nodes[nid]
            path = path + (nid,)
            if node.terminal is not None:
                out.append(
                    Behavior(
                        path,
                        tuple(self.nodes[i].state for i in path),
                        node.terminal,
                        node.cycle_target,
                    )
                )
            for child in node.children:
                walk(child, path)

        walk(self.root, ())
        return tuple(out)

    def describe_state(self, state: QState) -> str:
        parts = []
        for name, space in zip(self.var_order, self.spaces):
            parts.append(f"{name}={state[name].describe(space)}")
        tag = "•" if state.time is TimeTag.POINT else "~"
        return f"[{tag}] " + " ".join(parts)

    def export(self) -> dict:
        """Neutral, render-agnostic export: node table + edge lists."""
        nodes = []
        edges: list[list[int]] = []
        cycle_edges: list[list[int]] = []
        for node in self.nodes.values():
            nodes.append(
                {
                    "id": node.id,
                    "parent": node.parent,
                    "depth": node.depth,
                    "time": node.state.time.value,
                    "terminal": node.terminal.value if node.terminal else None,
                    "values": {
                        name: [qv.mag, int(qv.dir)]
                        for name, qv in node.state.values
                    },
                }
            )
            edges.extend([node.id, c] for c in node.children)
            if node.cycle_target is not None:
                cycle_edges.append([node.id, node.cycle_target])
        return {
            "var_order": list(self.var_order),
            "nodes": nodes,
            "edges": edges,
            "cycle_edges": cycle_edges,
        }

    def to_dot(self) -> str:
        lines = ["digraph behaviors {", '  node [shape=box, fontname="monospace"];']
        for node in self.nodes.values():
            label = self.describe_state(node.state).replace('"', r"\"")
            if node.terminal is not None:
                label += rf"\n<{node.terminal.value}>"
            lines.append(f'  n{node.id} [label="{label}"];')
        for node in self.nodes.values():
            for child in node.children:
                lines.append(f"  n{node.id} -> n{child};")
            if node.cycle_target is not None:
                lines.append(f"  n{node.id} -> n{node.cycle_target} [style=dashed];")
        lines.append("}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SimResult:
    """Engine output: graph + status + statistics + the config that made it."""

    graph: BehaviorGraph
    status: SimStatus
    stats: dict
    config: SimConfig

    def behaviors(self) -> tuple[Behavior, ...]:
        return self.graph.behaviors()

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "stats": dict(self.stats),
            "config": asdict(self.config),
            "graph": self.graph.export(),
        }
