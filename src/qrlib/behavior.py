"""Behavior graphs and simulation results.

The output side of every engine: a :class:`BehaviorGraph` of
:class:`~qrlib.state.QState` nodes with terminal classifications, wrapped in
a :class:`SimResult` that records the config and statistics that produced it
(truncation and filtering are always reported, never silent — see
docs/host-integration.md, cross-cutting conventions).

Because QSIM discovers landmarks mid-simulation, quantity spaces are
per-branch: every node carries its **frame** (a
:class:`~qrlib.model.CompiledModel` with that branch's grown spaces and
corresponding values). Nodes on the same branch share frame objects until a
landmark is minted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable

from .model import CompiledModel
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

    - ``QUIESCENT``: all tracked directions steady — an equilibrium; the
      constant continuation is one complete behavior (an unstable
      equilibrium may *also* have departing children).
    - ``CYCLE``: the state matches an ancestor point state in the same
      frame; the behavior closes into that ancestor (``Node.cycle_target``).
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


# (parent_state, candidate_successor_state, frame) -> keep?  The candidate is
# in the *parent's* frame (pre-minting), so the frame's spaces/cvals describe
# both states. Returning False vetoes the successor. Must be pure.
SuccessorFilter = Callable[[QState, QState, CompiledModel], bool]


@dataclass(frozen=True)
class SimConfig:
    """Engine configuration. Filter toggles exist for experimentation and
    testing; defaults are textbook QSIM behavior.

    - ``discover_landmarks``: mint named landmarks where variables become
      steady at unnamed values (per-branch quantity spaces). Off = phase-1
      semantics (steady values stay unnamed; still sound).
    - ``max_landmarks``: per-variable cap on discovered landmarks; beyond
      it, steadiness stays unnamed (bounds the classic landmark explosion).
    - ``ignore_qdir``: variable names whose direction is not tracked
      (chatter abstraction): candidates are generated over all concrete
      directions, filtered normally, then projected to ``Qdir.IGN`` and
      merged — collapsing chatter branching soundly.
    - ``successor_filters``: user vetoes applied to assembled successors —
      the hook for analytic knowledge (e.g. energy arguments) that prunes
      spurious behaviors without touching core semantics.
    - ``envisionment``: merge identical (frame, state) pairs globally,
      producing the attainable envisionment graph instead of a tree.
    """

    max_states: int = 500
    max_depth: int = 100
    no_change_filter: bool = True
    cycle_detection: bool = True
    infinity_filter: bool = True
    discover_landmarks: bool = True
    max_landmarks: int = 6
    ignore_qdir: tuple[str, ...] = ()
    successor_filters: tuple[SuccessorFilter, ...] = ()
    envisionment: bool = False


@dataclass
class Node:
    """One node of a behavior graph (mutable during construction only)."""

    id: int
    state: QState
    parent: int | None
    depth: int
    model: CompiledModel
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
    """Directed graph over qualitative states, rooted at the initial state.

    ``spaces`` are the root frame's quantity spaces; branches that minted
    landmarks carry grown spaces on their nodes (``node.model.spaces``)."""

    nodes: dict[int, Node]
    root: int
    var_order: tuple[str, ...]
    spaces: tuple[QuantitySpace, ...]

    def behaviors(self) -> tuple[Behavior, ...]:
        """All root-to-terminal paths, in deterministic (DFS) order.

        A node that is both terminal and expanded (an unstable equilibrium)
        contributes the terminated behavior *and* the extended ones. In
        envisionment mode, an edge back to a node already on the current
        path closes a CYCLE behavior.
        """
        out: list[Behavior] = []

        def states_of(path: tuple[int, ...]) -> tuple[QState, ...]:
            return tuple(self.nodes[i].state for i in path)

        def walk(nid: int, path: tuple[int, ...], on_path: frozenset[int]) -> None:
            node = self.nodes[nid]
            path = path + (nid,)
            on_path = on_path | {nid}
            if node.terminal is not None:
                out.append(
                    Behavior(path, states_of(path), node.terminal, node.cycle_target)
                )
            for child in node.children:
                if child in on_path:
                    closed = path + (child,)
                    out.append(
                        Behavior(
                            closed, states_of(closed), TerminalClass.CYCLE, child
                        )
                    )
                else:
                    walk(child, path, on_path)

        walk(self.root, (), frozenset())
        return tuple(out)

    def describe_node(self, node: Node) -> str:
        return self._describe(node.state, node.model.spaces)

    def describe_state(self, state: QState) -> str:
        """Describe a state in the *root* frame (pre-discovery spaces)."""
        return self._describe(state, self.spaces)

    def _describe(self, state: QState, spaces: tuple[QuantitySpace, ...]) -> str:
        parts = [
            f"{name}={state[name].describe(space)}"
            for name, space in zip(self.var_order, spaces)
        ]
        tag = "•" if state.time is TimeTag.POINT else "~"
        return f"[{tag}] " + " ".join(parts)

    def export(self) -> dict:
        """Neutral, render-agnostic export: node table + edges + frames."""
        frames: list[CompiledModel] = []
        frame_ids: dict[int, int] = {}  # node id -> frame index

        def frame_index(model: CompiledModel) -> int:
            for i, f in enumerate(frames):
                if f == model:
                    return i
            frames.append(model)
            return len(frames) - 1

        nodes = []
        edges: list[list[int]] = []
        cycle_edges: list[list[int]] = []
        for node in self.nodes.values():
            frame_ids[node.id] = frame_index(node.model)
            nodes.append(
                {
                    "id": node.id,
                    "parent": node.parent,
                    "depth": node.depth,
                    "frame": frame_ids[node.id],
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
            "frames": [
                {
                    name: list(space.effective_landmarks)
                    for name, space in zip(self.var_order, f.spaces)
                }
                for f in frames
            ],
            "nodes": nodes,
            "edges": edges,
            "cycle_edges": cycle_edges,
        }

    def to_dot(self) -> str:
        lines = ["digraph behaviors {", '  node [shape=box, fontname="monospace"];']
        for node in self.nodes.values():
            label = self.describe_node(node).replace('"', r"\"")
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
        config = asdict(self.config)
        # callables are not serializable; record how many were active
        config["successor_filters"] = len(self.config.successor_filters)
        return {
            "status": self.status.value,
            "stats": dict(self.stats),
            "config": config,
            "graph": self.graph.export(),
        }
