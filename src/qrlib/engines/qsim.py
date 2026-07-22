"""Reference QSIM engine.

The classic agenda loop (docs/qsim.md §2): pop a state, classify terminals,
generate per-variable candidates from the transition tables, filter
(tuple/Waltz -> global assembly -> global filters), attach surviving
successors, repeat. Written for clarity and auditability; the tensorized
engine (phase 5) must agree with it exactly.

Phase-2 machinery (docs/qsim.md §8): landmark discovery (per-branch
frames), chatter abstraction (``ignore_qdir``), user successor filters,
and envisionment mode.

Phase-4 machinery — operating regions: every node carries its region; the
region's constraint subset governs filtering. At a point state, if a
region transition's guards hold (landmark predicates on magnitudes), the
behavior crosses instantaneously: **entry states** are created in the
target region with the same magnitudes and re-derived directions (the
vector field may change discontinuously at the boundary), producing a
point -> point edge. A boundary with no declared transition still ends in
``REGION_EXIT``.

Terminal classification at a point state, in order:

1. DIVERGENT   — some variable at an infinite landmark (t = infinity).
2. (region transitions fire here: entry children, no terminal)
3. QUIESCENT   — all tracked directions steady; the constant continuation
                 is a complete behavior. Departures (unstable equilibria)
                 are still explored and become children if consistent.
4. CYCLE       — state equals an ancestor point state in an equal frame
                 and region (tree mode only; envisionment merges instead).
5. REGION_EXIT — some variable must leave its bounded quantity space
                 (empty P-successor set, no transition declared).
6. DEADEND     — candidates existed but none survived filtering (the state
                 was spurious; reported, never silently dropped).

Resource limits mark nodes TRUNCATED and the result status TRUNCATED.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from ..behavior import BehaviorGraph, Node, SimConfig, SimResult, SimStatus, TerminalClass
from ..model import CompiledModel, CompiledTransition, Model
from ..quantity import Qdir, QVal
from ..state import QState, TimeTag
from . import filters, phase
from .landmarks import introduce_landmarks
from .transitions import interval_successors, point_successors

__all__ = ["qsim"]

_TRACKED_STEADY = (Qdir.STD, Qdir.IGN)


def qsim(
    model: Model | CompiledModel,
    initial: QState,
    *,
    config: SimConfig | None = None,
    max_states: int | None = None,
    max_depth: int | None = None,
) -> SimResult:
    """Simulate all qualitative behaviors of ``model`` from ``initial``."""
    root_frame = model.compile() if isinstance(model, Model) else model
    cfg = config or SimConfig()
    overrides = {
        k: v
        for k, v in (("max_states", max_states), ("max_depth", max_depth))
        if v is not None
    }
    if overrides:
        cfg = replace(cfg, **overrides)

    unknown = set(cfg.ignore_qdir) - set(root_frame.var_order)
    if unknown:
        raise ValueError(f"ignore_qdir names unknown variables: {sorted(unknown)}")
    ignored = frozenset(root_frame.index(name) for name in cfg.ignore_qdir)
    phase_pairs = phase.validate_pairs(root_frame, cfg)

    root_region = root_frame.initial_region
    _validate_initial(root_frame, initial, root_region)
    init_state = _project(root_frame, initial, ignored)

    progress_fn = None
    root_guide = None
    if cfg.guide is not None:
        from ..guide import FALSE as _SPEC_FALSE
        from ..guide import progress as progress_fn

        root_guide = progress_fn(cfg.guide, init_state, root_frame)

    nodes: dict[int, Node] = {
        0: Node(0, init_state, None, 0, root_frame, root_region, guide=root_guide)
    }
    frontier: deque[int] = deque([0])
    seen = (
        {(root_frame, init_state, root_region, root_guide): 0}
        if cfg.envisionment
        else None
    )
    stats = {
        "nodes": 1,
        "expanded": 0,
        "candidates": 0,
        "no_change_filtered": 0,
        "infinity_filtered": 0,
        "user_filtered": 0,
        "merged": 0,
        "landmarks_minted": 0,
        "region_crossings": 0,
        "spec_filtered": 0,
        "phase_filtered": 0,
        "deadends": 0,
    }
    truncated = False
    if root_guide is not None and root_guide == _SPEC_FALSE:
        nodes[0].terminal = TerminalClass.SPEC_PRUNED
        frontier.clear()

    def attach(parent: Node, state: QState, frame: CompiledModel, region: str) -> str:
        child_guide = None
        if progress_fn is not None:
            child_guide = progress_fn(parent.guide, state, frame)
            if child_guide == _SPEC_FALSE:
                # bad prefix: no extension can satisfy the spec
                stats["spec_filtered"] += 1
                return "spec"
        if phase_pairs and not phase.admits(
            nodes, parent, state, frame, phase_pairs, root_frame.spaces
        ):
            # the path's phase-plane curve would have to cross itself
            stats["phase_filtered"] += 1
            return "phase"
        if seen is not None:
            key = (frame, state, region, child_guide)
            hit = seen.get(key)
            if hit is not None:
                if hit not in parent.children:
                    parent.children.append(hit)
                stats["merged"] += 1
                return "merged"
        nid = len(nodes)
        nodes[nid] = Node(
            nid, state, parent.id, parent.depth + 1, frame, region,
            guide=child_guide,
        )
        parent.children.append(nid)
        frontier.append(nid)
        stats["nodes"] += 1
        if seen is not None:
            seen[(frame, state, region, child_guide)] = nid

    while frontier:
        nid = frontier.popleft()
        node = nodes[nid]
        frame = node.model
        state = node.state
        vals = tuple(state[v] for v in frame.var_order)
        is_point = state.time is TimeTag.POINT
        active_idx = frame.region_named(node.region).constraint_idx

        if len(nodes) > cfg.max_states or node.depth >= cfg.max_depth:
            node.terminal = TerminalClass.TRUNCATED
            truncated = True
            continue

        if is_point and any(
            filters._inf_status(qv, frame.inf_ranks(vi))
            for vi, qv in enumerate(vals)
        ):
            node.terminal = TerminalClass.DIVERGENT
            continue

        if is_point:
            firing = [
                tr
                for tr in frame.region_named(node.region).transitions
                if _guards_hold(frame, vals, tr)
            ]
            if firing:
                stats["region_crossings"] += len(firing)
                spec_killed = False
                for tr in firing:
                    for entry in _entry_states(
                        frame, vals, tr.target, cfg, stats, ignored
                    ):
                        spec_killed |= attach(node, entry, frame, tr.target) == "spec"
                if not node.children:
                    if spec_killed:  # the spec excluded every entry
                        node.terminal = TerminalClass.SPEC_PRUNED
                    else:  # none existed, or the phase filter refuted them
                        node.terminal = TerminalClass.DEADEND
                        stats["deadends"] += 1
                continue

        if all(qv.dir in _TRACKED_STEADY for qv in vals):
            node.terminal = TerminalClass.QUIESCENT
            if is_point:
                # Explore departures (unstable equilibria); the identity
                # continuation is the quiescent behavior itself.
                succ = _expand(
                    frame, state, vals, cfg, stats, ignored, active_idx, exclude=vals
                )
                for child_state, child_frame in succ or []:
                    attach(node, child_state, child_frame, node.region)
                stats["expanded"] += 1
            continue

        if is_point and cfg.cycle_detection and not cfg.envisionment:
            anc = _matching_ancestor(nodes, node)
            if anc is not None:
                node.terminal = TerminalClass.CYCLE
                node.cycle_target = anc
                continue

        exclude = vals if (not is_point and cfg.no_change_filter) else None
        children = _expand(
            frame, state, vals, cfg, stats, ignored, active_idx, exclude=exclude
        )
        if children is None:  # some variable has no legal transition
            node.terminal = (
                TerminalClass.REGION_EXIT if is_point else TerminalClass.DEADEND
            )
            continue
        stats["expanded"] += 1
        if not children:
            node.terminal = TerminalClass.DEADEND
            stats["deadends"] += 1
            continue
        spec_killed = False
        for child_state, child_frame in children:
            spec_killed |= attach(node, child_state, child_frame, node.region) == "spec"
        if not node.children:
            if spec_killed:  # the spec excluded every consistent successor
                node.terminal = TerminalClass.SPEC_PRUNED
            else:  # the phase filter refuted every one: the state is spurious
                node.terminal = TerminalClass.DEADEND
                stats["deadends"] += 1

    graph = BehaviorGraph(nodes, 0, root_frame.var_order, root_frame.spaces)
    status = SimStatus.TRUNCATED if truncated else SimStatus.COMPLETE
    return SimResult(graph, status, stats, cfg)


def _guards_hold(
    frame: CompiledModel, vals: tuple[QVal, ...], tr: CompiledTransition
) -> bool:
    for vi, op, landmark in tr.guards:
        rank = vals[vi].mag
        ref = 2 * frame.spaces[vi].effective_landmarks.index(landmark)
        holds = {
            "==": rank == ref,
            "<": rank < ref,
            ">": rank > ref,
            "<=": rank <= ref,
            ">=": rank >= ref,
        }[op]
        if not holds:
            return False
    return True


def _filtered_combos(frame, domains, active_idx, cfg):
    """Consistent complete assignments — reference pipeline or the
    tensorized tables (identical results by contract)."""
    if cfg.use_tensor:
        from ..tensor.engine import filtered_combos

        return filtered_combos(frame, domains, active_idx)
    active = tuple(frame.constraints[i] for i in active_idx)
    pruned = filters.prune_domains(frame, domains, active)
    if pruned is None:
        return []
    return filters.assemble(frame, pruned, active)


def _entry_states(
    frame: CompiledModel,
    vals: tuple[QVal, ...],
    target: str,
    cfg: SimConfig,
    stats: dict,
    ignored: frozenset[int],
) -> list[QState]:
    """Region-entry states: magnitudes carry over, directions re-derive
    under the target region's constraints (the vector field may change
    discontinuously at the boundary)."""
    active_idx = frame.region_named(target).constraint_idx
    domains = [
        [QVal(qv.mag, d) for d in (Qdir.DEC, Qdir.STD, Qdir.INC)] for qv in vals
    ]
    out: list[QState] = []
    emitted: set[QState] = set()
    for combo in _filtered_combos(frame, domains, active_idx, cfg):
        stats["candidates"] += 1
        projected = tuple(
            QVal(qv.mag, Qdir.IGN) if vi in ignored else qv
            for vi, qv in enumerate(combo)
        )
        entry = QState.from_dict(
            dict(zip(frame.var_order, projected)), TimeTag.POINT
        )
        if entry in emitted:
            continue
        emitted.add(entry)
        out.append(entry)
    return out


def _expand(
    frame: CompiledModel,
    state: QState,
    vals: tuple[QVal, ...],
    cfg: SimConfig,
    stats: dict,
    ignored: frozenset[int],
    active_idx: tuple[int, ...],
    *,
    exclude: tuple[QVal, ...] | None,
) -> list[tuple[QState, CompiledModel]] | None:
    """Filtered successors of ``state`` with their (possibly grown) frames.

    Returns None when some variable has no legal transition at all (the
    region-exit condition at points); an empty list when candidates existed
    but none survived filtering.
    """
    is_point = state.time is TimeTag.POINT
    table = point_successors if is_point else interval_successors
    next_time = TimeTag.INTERVAL if is_point else TimeTag.POINT

    domains: list[list[QVal]] = []
    for vi, qv in enumerate(vals):
        if qv.dir is Qdir.IGN:
            # direction untracked: candidates from every concrete direction
            union: list[QVal] = []
            for d in (Qdir.DEC, Qdir.STD, Qdir.INC):
                for cand in table(QVal(qv.mag, d), frame.spaces[vi]):
                    if cand not in union:
                        union.append(cand)
            domains.append(union)
        else:
            domains.append(table(qv, frame.spaces[vi]))
    if any(not d for d in domains):
        return None

    out: list[tuple[QState, CompiledModel]] = []
    emitted: set[QState] = set()
    for combo in _filtered_combos(frame, domains, active_idx, cfg):
        stats["candidates"] += 1
        if (
            next_time is TimeTag.POINT
            and cfg.infinity_filter
            and not filters.admissible_at_infinity(frame, combo)
        ):
            stats["infinity_filtered"] += 1
            continue
        projected = tuple(
            QVal(qv.mag, Qdir.IGN) if vi in ignored else qv
            for vi, qv in enumerate(combo)
        )
        if exclude is not None and projected == exclude:
            stats["no_change_filtered"] += 1
            continue
        cand_state = QState.from_dict(
            dict(zip(frame.var_order, projected)), next_time
        )
        if cand_state in emitted:  # chatter branches collapse here
            continue
        if cfg.successor_filters and not all(
            keep(state, cand_state, frame) for keep in cfg.successor_filters
        ):
            stats["user_filtered"] += 1
            continue
        emitted.add(cand_state)

        child_frame = frame
        child_state = cand_state
        if next_time is TimeTag.POINT and cfg.discover_landmarks:
            child_frame, minted_vals, minted = introduce_landmarks(
                frame, vals, projected, cfg.max_landmarks
            )
            if minted:
                stats["landmarks_minted"] += len(minted)
                child_state = QState.from_dict(
                    dict(zip(frame.var_order, minted_vals)), next_time
                )
        out.append((child_state, child_frame))
    return out


def _project(
    frame: CompiledModel, state: QState, ignored: frozenset[int]
) -> QState:
    if not ignored:
        return state
    values = {
        name: (QVal(state[name].mag, Qdir.IGN) if vi in ignored else state[name])
        for vi, name in enumerate(frame.var_order)
    }
    return QState.from_dict(values, state.time)


def _matching_ancestor(nodes: dict[int, Node], node: Node) -> int | None:
    pid = node.parent
    while pid is not None:
        anc = nodes[pid]
        if (
            anc.state == node.state
            and anc.model == node.model
            and anc.region == node.region
            and anc.guide == node.guide
        ):
            return pid
        pid = anc.parent
    return None


def _validate_initial(
    compiled: CompiledModel, initial: QState, region: str
) -> None:
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
        qv = initial[name]
        if not 0 <= qv.mag < compiled.spaces[vi].num_ranks:
            raise ValueError(
                f"initial value of {name!r} has out-of-range rank {qv.mag}"
            )
        if qv.dir is Qdir.IGN:
            raise ValueError(
                f"initial value of {name!r} must use a concrete direction "
                f"(the engine projects ignored directions itself)"
            )
    violated = filters.check_state(
        compiled, initial, compiled.constraints_of(region)
    )
    if violated is not None:
        raise ValueError(f"initial state violates constraint {violated.source!r}")
