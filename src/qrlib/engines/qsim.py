"""Reference QSIM engine.

The classic agenda loop (docs/qsim.md §2): pop a state, classify terminals,
generate per-variable candidates from the transition tables, filter
(tuple/Waltz -> global assembly -> global filters), attach surviving
successors, repeat. Written for clarity and auditability; the tensorized
engine (phase 5) must agree with it exactly.

Terminal classification at a point state, in order:

1. DIVERGENT   — some variable at an infinite landmark (t = infinity).
2. QUIESCENT   — all directions steady; the constant continuation is a
                 complete behavior. Departures (unstable equilibria) are
                 still explored and become children if consistent.
3. CYCLE       — state equals an ancestor point state; close the loop.
4. REGION_EXIT — some variable must leave its bounded quantity space
                 (empty P-successor set).
5. DEADEND     — candidates existed but none survived filtering (the state
                 was spurious; reported, never silently dropped).

Resource limits mark nodes TRUNCATED and the result status TRUNCATED.
"""

from __future__ import annotations

from collections import deque

from ..behavior import BehaviorGraph, Node, SimConfig, SimResult, SimStatus, TerminalClass
from ..model import CompiledModel, Model
from ..quantity import Qdir, QVal
from ..state import QState, TimeTag
from . import filters
from .transitions import interval_successors, point_successors

__all__ = ["qsim"]


def qsim(
    model: Model | CompiledModel,
    initial: QState,
    *,
    config: SimConfig | None = None,
    max_states: int | None = None,
    max_depth: int | None = None,
) -> SimResult:
    """Simulate all qualitative behaviors of ``model`` from ``initial``."""
    compiled = model.compile() if isinstance(model, Model) else model
    cfg = config or SimConfig()
    if max_states is not None or max_depth is not None:
        from dataclasses import replace

        cfg = replace(
            cfg,
            **{
                k: v
                for k, v in (("max_states", max_states), ("max_depth", max_depth))
                if v is not None
            },
        )

    _validate_initial(compiled, initial)

    nodes: dict[int, Node] = {0: Node(0, initial, None, 0)}
    frontier: deque[int] = deque([0])
    stats = {
        "nodes": 1,
        "expanded": 0,
        "candidates": 0,
        "no_change_filtered": 0,
        "infinity_filtered": 0,
        "deadends": 0,
    }
    truncated = False

    def add_child(parent: Node, state: QState) -> None:
        nid = len(nodes)
        nodes[nid] = Node(nid, state, parent.id, parent.depth + 1)
        parent.children.append(nid)
        frontier.append(nid)
        stats["nodes"] += 1

    while frontier:
        nid = frontier.popleft()
        node = nodes[nid]
        state = node.state
        vals = tuple(state[v] for v in compiled.var_order)
        is_point = state.time is TimeTag.POINT

        if len(nodes) > cfg.max_states or node.depth >= cfg.max_depth:
            node.terminal = TerminalClass.TRUNCATED
            truncated = True
            continue

        if is_point and any(
            filters._inf_status(qv, compiled.inf_ranks(vi))
            for vi, qv in enumerate(vals)
        ):
            node.terminal = TerminalClass.DIVERGENT
            continue

        if all(qv.dir is Qdir.STD for qv in vals):
            node.terminal = TerminalClass.QUIESCENT
            if is_point:
                # Explore departures (unstable equilibria); the identity
                # continuation is the quiescent behavior itself.
                for succ in _successors(compiled, state, cfg, stats, exclude=vals):
                    add_child(node, succ)
                stats["expanded"] += 1
            continue

        if is_point and cfg.cycle_detection:
            anc = _matching_ancestor(nodes, node)
            if anc is not None:
                node.terminal = TerminalClass.CYCLE
                node.cycle_target = anc
                continue

        if is_point and any(
            not point_successors(qv, compiled.spaces[vi])
            for vi, qv in enumerate(vals)
        ):
            node.terminal = TerminalClass.REGION_EXIT
            continue

        exclude = vals if (not is_point and cfg.no_change_filter) else None
        children = list(_successors(compiled, state, cfg, stats, exclude=exclude))
        stats["expanded"] += 1
        if not children:
            node.terminal = TerminalClass.DEADEND
            stats["deadends"] += 1
            continue
        for succ in children:
            add_child(node, succ)

    graph = BehaviorGraph(nodes, 0, compiled.var_order, compiled.spaces)
    status = SimStatus.TRUNCATED if truncated else SimStatus.COMPLETE
    return SimResult(graph, status, stats, cfg)


def _successors(
    compiled: CompiledModel,
    state: QState,
    cfg: SimConfig,
    stats: dict,
    *,
    exclude: tuple[QVal, ...] | None,
) -> list[QState]:
    """Filtered successor states of ``state`` (the opposite time tag).

    ``exclude`` drops the assignment identical to those values: the
    no-change filter for interval states, the identity continuation for
    quiescent points.
    """
    is_point = state.time is TimeTag.POINT
    table = point_successors if is_point else interval_successors
    next_time = TimeTag.INTERVAL if is_point else TimeTag.POINT

    domains = [
        table(state[v], compiled.spaces[i])
        for i, v in enumerate(compiled.var_order)
    ]
    if any(not d for d in domains):
        return []
    pruned = filters.prune_domains(compiled, domains)
    if pruned is None:
        return []

    out: list[QState] = []
    for combo in filters.assemble(compiled, pruned):
        stats["candidates"] += 1
        if exclude is not None and combo == exclude:
            stats["no_change_filtered"] += 1
            continue
        if (
            next_time is TimeTag.POINT
            and cfg.infinity_filter
            and not filters.admissible_at_infinity(compiled, combo)
        ):
            stats["infinity_filtered"] += 1
            continue
        out.append(
            QState.from_dict(dict(zip(compiled.var_order, combo)), next_time)
        )
    return out


def _matching_ancestor(nodes: dict[int, Node], node: Node) -> int | None:
    pid = node.parent
    while pid is not None:
        anc = nodes[pid]
        if anc.state == node.state:
            return pid
        pid = anc.parent
    return None


def _validate_initial(compiled: CompiledModel, initial: QState) -> None:
    if initial.time is not TimeTag.POINT:
        raise ValueError("the initial state must be a time-point state")
    missing = set(compiled.var_order) - set(initial.variables)
    extra = set(initial.variables) - set(compiled.var_order)
    if missing or extra:
        raise ValueError(
            f"initial state variables do not match the model "
            f"(missing: {sorted(missing)}, unknown: {sorted(extra)})"
        )
    for vi, name in enumerate(compiled.var_order):
        rank = initial[name].mag
        if not 0 <= rank < compiled.spaces[vi].num_ranks:
            raise ValueError(f"initial value of {name!r} has out-of-range rank {rank}")
    violated = filters.check_state(compiled, initial)
    if violated is not None:
        raise ValueError(
            f"initial state violates constraint {violated.source!r}"
        )
