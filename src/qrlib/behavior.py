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

from dataclasses import dataclass, field, fields
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
    "RESULT_SCHEMA",
]

RESULT_SCHEMA = "qrlib.result/v2"


class TerminalClass(Enum):
    """Why a behavior ends at a node.

    - ``QUIESCENT``: all tracked directions steady — an equilibrium; the
      constant continuation is one complete behavior (an unstable
      equilibrium may *also* have departing children).
    - ``CYCLE``: the state matches an ancestor point state in the same
      frame; the behavior closes into that ancestor (``Node.cycle_target``).
    - ``DIVERGENT``: some variable is at an infinite landmark (the state
      represents the limit t -> infinity).
    - ``DOMAIN_EXIT``: a variable must leave its bounded quantity space — the
      system leaves the model's domain of validity.
    - ``REGION_EXIT``: a declared operating region is left at a guarded or
      outward boundary for which no target transition applies.
    - ``DEADEND``: no consistent successor survived filtering. A dead end
      indicates the state itself is spurious (real behaviors continue), but
      it is reported rather than silently pruned.
    - ``SPEC_PRUNED``: physically consistent successors existed, but the
      guide specification excluded every one of them (guided simulation
      only) — the spec, not the model, ends this path.
    - ``TRUNCATED``: a resource limit stopped exploration here.
    """

    QUIESCENT = "quiescent"
    CYCLE = "cycle"
    DIVERGENT = "divergent"
    DOMAIN_EXIT = "domain_exit"
    REGION_EXIT = "region_exit"
    DEADEND = "deadend"
    SPEC_PRUNED = "spec_pruned"
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
    - ``dynamic_chatter``: detect chatter automatically instead of naming
      variables in ``ignore_qdir``. Structural analysis (per region) finds
      the variables whose direction nothing pins (``engines.chatter``);
      during expansion, successors identical except for those variables'
      directions are merged, with the wiggling directions projected to
      ``Qdir.IGN`` — the abstraction applies exactly where chatter
      branching manifests, and a candidate whose direction is forced at
      some state keeps it concrete there.
    - ``track_qdir``: variables ``dynamic_chatter`` must never abstract
      (their direction stays tracked even if it chatters). Phase-pair
      variables and directions a guide spec mentions are force-tracked
      automatically.
    - ``successor_filters``: user vetoes applied to assembled successors —
      the hook for analytic knowledge (e.g. energy arguments) that prunes
      spurious behaviors without touching core semantics. A declarative
      ``LyapunovCertificate`` also rejects a recurrent path when its scalar
      made strict progress inside an unchanged qualitative interval; because
      that check is path-dependent, it is incompatible with ``envisionment``.
    - ``envisionment``: merge identical (frame, state) pairs globally,
      producing the attainable envisionment graph instead of a tree.
    - ``backend``: successor-filtering backend: ``"auto"`` selects from
      workload shape, ``"reference"`` forces readable Python predicates,
      and ``"tensor"`` forces tensor lookup tables. Results are identical
      by contract. ``use_tensor`` remains a compatibility override:
      ``True`` forces tensor and ``False`` forces reference.
    - ``guide``: a temporal-logic spec (``qrlib.guide.Formula``) progressed
      along every path; successors whose residual collapses to false are
      pruned (sound: only behaviors none of whose extensions satisfy the
      spec are removed). Usually set via ``qrlib.guide.guided``.
    - ``phase_pairs``: pairs ``(x, y)`` with ``Deriv(x, y)`` declared as
      autonomous phase planes; the non-intersection global filter
      (``engines.phase``, after Lee & Kuipers) prunes successors whose
      path would trace a self-crossing curve in the (x, y) plane —
      successive same-direction crossings of a grid line must be
      monotone or exactly repeating. Declaring a pair asserts that (x, y)
      determine the flow (second-order autonomous system) and that the
      pair's equilibria lie at declared landmarks; a pair violating this
      can prune real behaviors. Path-dependent: incompatible with
      ``envisionment``.
    - ``max_states``: strict upper bound on graph nodes, including the root.
      When admitting another successor would exceed it, the current frontier
      is marked truncated and the result status is ``TRUNCATED``.
    """

    max_states: int = 500
    max_depth: int = 100
    no_change_filter: bool = True
    cycle_detection: bool = True
    infinity_filter: bool = True
    discover_landmarks: bool = True
    max_landmarks: int = 6
    ignore_qdir: tuple[str, ...] = ()
    dynamic_chatter: bool = False
    track_qdir: tuple[str, ...] = ()
    successor_filters: tuple[SuccessorFilter, ...] = ()
    envisionment: bool = False
    backend: str = "auto"
    use_tensor: bool | None = None
    guide: object | None = None
    phase_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.max_states < 1:
            raise ValueError("max_states must be at least 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        valid = ("auto", "reference", "tensor")
        if self.backend not in valid:
            raise ValueError(f"backend must be one of {valid}, got {self.backend!r}")
        if self.use_tensor is not None and not isinstance(self.use_tensor, bool):
            raise TypeError(
                f"use_tensor must be bool or None, got {type(self.use_tensor).__name__}"
            )
        if self.use_tensor is not None:
            legacy = "tensor" if self.use_tensor else "reference"
            if self.backend != "auto" and self.backend != legacy:
                raise ValueError(
                    f"backend={self.backend!r} conflicts with "
                    f"use_tensor={self.use_tensor!r}"
                )

    @property
    def backend_mode(self) -> str:
        """Resolved request, including the legacy ``use_tensor`` override."""
        if self.use_tensor is not None:
            return "tensor" if self.use_tensor else "reference"
        return self.backend


@dataclass
class Node:
    """One node of a behavior graph (mutable during construction only)."""

    id: int
    state: QState
    parent: int | None
    depth: int
    model: CompiledModel
    region: str = "default"
    children: list[int] = field(default_factory=list)
    terminal: TerminalClass | None = None
    cycle_target: int | None = None
    guide: object | None = None  # residual spec formula (guided runs)


@dataclass(frozen=True)
class Behavior:
    """A root-to-terminal path: one qualitative behavior."""

    node_ids: tuple[int, ...]
    states: tuple[QState, ...]
    terminal: TerminalClass
    cycle_target: int | None = None
    regions: tuple[str, ...] = ()


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

        def regions_of(path: tuple[int, ...]) -> tuple[str, ...]:
            return tuple(self.nodes[i].region for i in path)

        def walk(nid: int, path: tuple[int, ...], on_path: frozenset[int]) -> None:
            node = self.nodes[nid]
            path = path + (nid,)
            on_path = on_path | {nid}
            if node.terminal is not None:
                out.append(
                    Behavior(
                        path,
                        states_of(path),
                        node.terminal,
                        node.cycle_target,
                        regions_of(path),
                    )
                )
            for child in node.children:
                if child in on_path:
                    closed = path + (child,)
                    out.append(
                        Behavior(
                            closed,
                            states_of(closed),
                            TerminalClass.CYCLE,
                            child,
                            regions_of(closed),
                        )
                    )
                else:
                    walk(child, path, on_path)

        walk(self.root, (), frozenset())
        return tuple(out)

    def describe_node(self, node: Node, *, ascii: bool = False) -> str:
        text = self._describe(node.state, node.model.spaces, ascii=ascii)
        if node.region != "default":
            text += f" @{node.region}"
        return text

    def describe_state(self, state: QState, *, ascii: bool = False) -> str:
        """Describe a state in the root frame.

        ``ascii=True`` uses console-safe direction and time glyphs.
        """
        return self._describe(state, self.spaces, ascii=ascii)

    def _describe(
        self,
        state: QState,
        spaces: tuple[QuantitySpace, ...],
        *,
        ascii: bool = False,
    ) -> str:
        parts = [
            f"{name}={state[name].describe(space, ascii=ascii)}"
            for name, space in zip(self.var_order, spaces)
        ]
        tag = "*" if ascii and state.time is TimeTag.POINT else (
            "•" if state.time is TimeTag.POINT else "~"
        )
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
                    "region": node.region,
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
    """Engine output with replay-oriented model/config provenance."""

    graph: BehaviorGraph
    status: SimStatus
    stats: dict
    config: SimConfig
    model_hash: str

    def behaviors(self) -> tuple[Behavior, ...]:
        return self.graph.behaviors()

    def to_dict(self) -> dict:
        # Shallow extraction avoids deepcopying arbitrary user callables;
        # the non-data fields are replaced with explicit descriptors below.
        config = {
            item.name: getattr(self.config, item.name)
            for item in fields(self.config)
        }
        config["successor_filters"] = [
            _successor_filter_descriptor(keep)
            for keep in self.config.successor_filters
        ]
        if self.config.guide is not None:
            from .guide import format_formula

            config["guide"] = format_formula(self.config.guide)
        return {
            "schema": RESULT_SCHEMA,
            "model_hash": self.model_hash,
            "status": self.status.value,
            "stats": dict(self.stats),
            "config": config,
            "graph": self.graph.export(),
        }


def _successor_filter_descriptor(keep: SuccessorFilter) -> dict:
    """Stable provenance for built-ins; explicit opacity for user callables."""
    from .energy import EnergyFilter, LyapunovCertificate

    if isinstance(keep, (EnergyFilter, LyapunovCertificate)):
        return {"replayable": True, **keep.to_dict()}
    module = getattr(keep, "__module__", type(keep).__module__)
    qualname = getattr(keep, "__qualname__", type(keep).__qualname__)
    return {
        "kind": "opaque",
        "replayable": False,
        "module": module,
        "qualname": qualname,
    }
