"""Total envisionment: the complete state-transition graph of a model.

Research lineage: [deKleerBrown1984], with QSIM state semantics from
[Kuipers1986] and [Kuipers1994], in ``docs/references.md``.

Attainable envisionment (``SimConfig.envisionment``) explores what is
reachable *from an initial state*. Total envisionment — ENVISION's and
early QSIM's original product — asks the initial-state-free question:
**every** consistent qualitative state of the model, connected by the
transition relation. The result is the model's full qualitative phase
portrait: all equilibria (including ones no particular initial state
reaches), all transient states, and the cycles among them.

Construction is two reuses of existing machinery:

1. **Enumerate**: hand the tuple/Waltz filter and global assembly the
   *full* per-variable domains (every magnitude × every direction);
   the consistent complete assignments are exactly the model's states.
   Each is a POINT node; those that can persist over an open interval
   (no variable sitting at a landmark while moving) are INTERVAL nodes
   too, since QSIM time alternates between the two.
2. **Connect**: successors of each node via the same P-/I-transition
   tables and filters the simulation engine uses — point → interval,
   interval → point (identity excluded: an unchanged closing point is
   not a distinguished time point), the infinity-admissibility rule on
   point candidates, and departures from unstable equilibria. Divergent
   states (a variable at an infinite landmark: the t → ∞ limit) are
   terminal. A quiescent state's constant continuation is itself — its
   classification, not a self-edge.

Landmark discovery is off by construction (discovery is path-relative;
a total envisionment has no paths), and the graph covers one operating
region's constraint set — envision per region for multi-region models;
crossing edges between regional envisionments are not built here.
Chatter abstraction (``ignore_qdir``) is supported and recommended for
the same reasons as in simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..graph import tarjan_scc
from ..model import CompiledModel, Model
from ..quantity import Qdir, QVal
from ..state import QState, TimeTag
from . import filters
from .transitions import interval_successors, point_successors

__all__ = ["EnvNode", "Envisionment", "envision"]

_CONCRETE = (Qdir.DEC, Qdir.STD, Qdir.INC)
_STEADY = (Qdir.STD, Qdir.IGN)


@dataclass(frozen=True)
class EnvNode:
    id: int
    state: QState
    kind: str  # "quiescent" | "divergent" | "transient"


@dataclass(frozen=True)
class Envisionment:
    """All consistent states of one region's constraint set + transitions."""

    frame: CompiledModel
    region: str
    nodes: tuple[EnvNode, ...]
    edges: tuple[tuple[int, int], ...]
    stats: dict

    def successors(self, nid: int) -> tuple[int, ...]:
        return tuple(b for a, b in self.edges if a == nid)

    def quiescent(self) -> tuple[EnvNode, ...]:
        return tuple(n for n in self.nodes if n.kind == "quiescent")

    def cycles(self) -> tuple[tuple[int, ...], ...]:
        """Non-trivial strongly connected components: the recurrent sets."""
        adj: dict[int, list[int]] = {n.id: [] for n in self.nodes}
        for a, b in self.edges:
            adj[a].append(b)
        return tuple(
            tuple(sorted(c)) for c in tarjan_scc(adj) if len(c) > 1
        )

    def describe(self, node: EnvNode) -> str:
        parts = [
            f"{name}={node.state[name].describe(space)}"
            for name, space in zip(self.frame.var_order, self.frame.spaces)
        ]
        tag = "•" if node.state.time is TimeTag.POINT else "~"
        return f"[{tag}] " + " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "schema": "qrlib.envisionment/v1",
            "region": self.region,
            "var_order": list(self.frame.var_order),
            "nodes": [
                {
                    "id": n.id,
                    "kind": n.kind,
                    "time": n.state.time.value,
                    "values": {
                        name: [qv.mag, int(qv.dir)] for name, qv in n.state.values
                    },
                }
                for n in self.nodes
            ],
            "edges": [list(e) for e in self.edges],
            "stats": dict(self.stats),
        }


def envision(
    model: Model | CompiledModel,
    *,
    region: str | None = None,
    ignore_qdir: tuple[str, ...] = (),
    max_states: int = 20000,
) -> Envisionment:
    """Build the total envisionment of ``model`` (one region's constraint
    set). Raises when the enumeration exceeds ``max_states`` — a partial
    total envisionment would be a contradiction in terms."""
    frame = model.compile() if isinstance(model, Model) else model
    if region is None:
        if len(frame.regions) > 1:
            raise ValueError(
                "multi-region model: pass region=... and envision each "
                "region separately"
            )
        region = frame.regions[0].name
    active = frame.constraints_of(region)
    unknown = set(ignore_qdir) - set(frame.var_order)
    if unknown:
        raise ValueError(f"ignore_qdir names unknown variables: {sorted(unknown)}")
    ignored = frozenset(frame.index(n) for n in ignore_qdir)

    # 1. enumerate: all consistent complete assignments
    domains = [
        [
            QVal(mag, d)
            for mag in range(space.num_ranks)
            for d in _CONCRETE
        ]
        for space in frame.spaces
    ]
    pruned = filters.prune_domains(frame, domains, active)
    combos: list[tuple[QVal, ...]] = []
    seen: set[tuple[QVal, ...]] = set()
    considered = 0
    for combo in ([] if pruned is None else filters.assemble(frame, pruned, active)):
        considered += 1
        projected = tuple(
            QVal(qv.mag, Qdir.IGN) if vi in ignored else qv
            for vi, qv in enumerate(combo)
        )
        if projected not in seen:
            seen.add(projected)
            combos.append(projected)
        if len(combos) > max_states:
            raise ValueError(
                f"state space exceeds max_states={max_states}; abstract "
                f"directions (ignore_qdir) or raise the cap"
            )

    def persists(vals: tuple[QVal, ...]) -> bool:
        # a variable at a landmark while moving leaves it immediately
        return all(qv.mag % 2 == 1 or qv.dir in _STEADY for qv in vals)

    nodes: list[EnvNode] = []
    ids: dict[QState, int] = {}
    for time in (TimeTag.POINT, TimeTag.INTERVAL):
        for vals in combos:
            if time is TimeTag.INTERVAL and not persists(vals):
                continue
            state = QState.from_dict(dict(zip(frame.var_order, vals)), time)
            if time is TimeTag.POINT:
                if any(
                    filters._inf_status(qv, frame.inf_ranks(vi))
                    for vi, qv in enumerate(vals)
                ):
                    kind = "divergent"
                elif all(qv.dir in _STEADY for qv in vals):
                    kind = "quiescent"
                else:
                    kind = "transient"
            else:
                kind = "transient"
            ids[state] = len(nodes)
            nodes.append(EnvNode(len(nodes), state, kind))

    # 2. connect: successors via the engine's transition tables + filters
    edges: list[tuple[int, int]] = []
    for node in nodes:
        if node.kind == "divergent":
            continue  # the t -> infinity limit: nothing follows
        vals = tuple(node.state[v] for v in frame.var_order)
        is_point = node.state.time is TimeTag.POINT
        table = point_successors if is_point else interval_successors
        next_time = TimeTag.INTERVAL if is_point else TimeTag.POINT
        succ_domains: list[list[QVal]] = []
        for vi, qv in enumerate(vals):
            if qv.dir is Qdir.IGN:
                union: list[QVal] = []
                for d in _CONCRETE:
                    for cand in table(QVal(qv.mag, d), frame.spaces[vi]):
                        if cand not in union:
                            union.append(cand)
                succ_domains.append(union)
            else:
                succ_domains.append(table(qv, frame.spaces[vi]))
        if any(not d for d in succ_domains):
            continue  # must leave the region's space: no successor here
        pruned_s = filters.prune_domains(frame, succ_domains, active)
        if pruned_s is None:
            continue
        emitted: set[int] = set()
        for combo in filters.assemble(frame, pruned_s, active):
            if next_time is TimeTag.POINT and not filters.admissible_at_infinity(
                frame, combo
            ):
                continue
            projected = tuple(
                QVal(qv.mag, Qdir.IGN) if vi in ignored else qv
                for vi, qv in enumerate(combo)
            )
            # identity is excluded closing an interval (an unchanged point
            # is not a distinguished event) and leaving quiescence (the
            # constant continuation is the classification, not an edge);
            # a transient point's identity interval is its continuation
            if projected == vals and (not is_point or node.kind == "quiescent"):
                continue
            target = QState.from_dict(
                dict(zip(frame.var_order, projected)), next_time
            )
            tid = ids.get(target)
            if tid is not None and tid not in emitted:
                emitted.add(tid)
                edges.append((node.id, tid))

    stats = {
        "assignments": considered,
        "states": len(combos),
        "point_nodes": sum(1 for n in nodes if n.state.time is TimeTag.POINT),
        "interval_nodes": sum(1 for n in nodes if n.state.time is TimeTag.INTERVAL),
        "edges": len(edges),
    }
    return Envisionment(frame, region, tuple(nodes), tuple(edges), stats)
