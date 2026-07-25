"""Reference QSIM engine.

Research lineage: Kuipers (1986; 1994; 2001).

The classic agenda loop pops a state, classifies terminals, generates
per-variable candidates from the transition tables, and filters
(tuple/Waltz -> global assembly -> global filters), attach surviving
successors, repeat. Written for clarity and auditability; the tensorized
engine (phase 5) must agree with it exactly.

Additional machinery includes landmark discovery (per-branch frames),
chatter abstraction (``ignore_qdir``), user successor filters,
path-aware Lyapunov recurrence pruning, and envisionment mode.

Phase-4 machinery — operating regions: every node carries its region; the
region's constraint subset governs filtering. At a point state, if a
region transition's guards hold (landmark predicates on magnitudes), the
behavior crosses instantaneously: **entry states** are created in the
target region with carried magnitudes plus any declared transition resets,
and re-derived directions (the vector field may change discontinuously at
the boundary), producing a point -> point edge. A declared region boundary
with no transition ends in ``REGION_EXIT``.

Terminal classification at a point state, in order:

1. DIVERGENT   — some variable at an infinite landmark (t = infinity).
2. (region transitions fire here: entry children, no terminal)
3. QUIESCENT   — all tracked directions steady; the constant continuation
                 is a complete behavior. Departures (unstable equilibria)
                 are still explored and become children if consistent.
4. CYCLE       — state equals an ancestor point state in an equal frame
                 and region (tree mode only; envisionment merges instead).
5. DOMAIN_EXIT  — an implicit/default model must leave a bounded quantity
                  space.
   REGION_EXIT  — a declared operating region has no applicable transition.
6. DEADEND     — candidates existed but none survived filtering (the state
                 was spurious; reported, never silently dropped).

Resource limits mark nodes TRUNCATED and the result status TRUNCATED.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from math import prod

from ..behavior import BehaviorGraph, Node, SimConfig, SimResult, SimStatus, TerminalClass
from ..energy import EnergyFilter, LyapunovCertificate
from ..model import CompiledModel, CompiledTransition, Model
from ..quantity import Qdir, QVal
from ..state import QState, TimeTag
from . import chatter, filters, phase
from .landmarks import introduce_landmarks
from .transitions import interval_successors, point_successors

__all__ = ["qsim"]

_TRACKED_STEADY = (Qdir.STD, Qdir.IGN)
AUTO_TENSOR_PRODUCT = 1 << 11


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

    # EnergyFilter's finite-amplitude argument compares named turning points.
    # Supplying it is an explicit request for that stronger representation,
    # so make the effective configuration honest and enable discovery rather
    # than silently running an inert filter under the practical profile.
    energy_discovery_enabled = (
        not cfg.discover_landmarks
        and any(
            isinstance(keep, EnergyFilter)
            for keep in cfg.successor_filters
        )
    )
    if energy_discovery_enabled:
        cfg = replace(cfg, discover_landmarks=True)

    unknown = set(cfg.ignore_qdir) - set(root_frame.var_order)
    if unknown:
        raise ValueError(f"ignore_qdir names unknown variables: {sorted(unknown)}")
    ignored = frozenset(root_frame.index(name) for name in cfg.ignore_qdir)
    phase_pairs = phase.validate_pairs(root_frame, cfg)
    lyapunov = tuple(
        keep
        for keep in cfg.successor_filters
        if isinstance(keep, LyapunovCertificate)
    )
    for certificate in lyapunov:
        certificate.validate(root_frame)
    if cfg.envisionment and lyapunov:
        raise ValueError(
            "LyapunovCertificate is path-dependent and incompatible with "
            "envisionment=True"
        )

    chatter_by_region: dict[str, frozenset[int]] = {}
    if cfg.dynamic_chatter:
        unknown = set(cfg.track_qdir) - set(root_frame.var_order)
        if unknown:
            raise ValueError(f"track_qdir names unknown variables: {sorted(unknown)}")
        # never abstract force-tracked, already-ignored, or phase-pair vars
        keep = {root_frame.index(n) for n in cfg.track_qdir} | ignored
        keep |= {vi for pair in phase_pairs for vi in pair}
        chatter_by_region = {
            r.name: chatter.candidates(root_frame, r.name) - keep
            for r in root_frame.regions
        }

    root_region = root_frame.initial_region
    _validate_initial(root_frame, initial, root_region)
    init_state = _project(root_frame, initial, ignored)
    for certificate in lyapunov:
        if not certificate(init_state, init_state, root_frame):
            raise ValueError(
                f"initial state violates {certificate.describe()}"
            )

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
        "lyapunov_cycles_filtered": 0,
        "merged": 0,
        "landmarks_minted": 0,
        "region_crossings": 0,
        "region_resets": 0,
        "spec_filtered": 0,
        "phase_filtered": 0,
        "chatter_merged": 0,
        "deadends": 0,
        "limit_hits": {"max_states": 0, "max_depth": 0},
        "backend": _new_backend_stats(cfg),
    }
    if energy_discovery_enabled:
        stats["config_adjustments"] = [
            "enabled landmark discovery for EnergyFilter"
        ]
    if cfg.dynamic_chatter:
        stats["chatter_candidates"] = {
            r: sorted(root_frame.var_order[i] for i in s)
            for r, s in chatter_by_region.items()
        }
    truncated = False
    if root_guide is not None and root_guide == _SPEC_FALSE:
        nodes[0].terminal = TerminalClass.SPEC_PRUNED
        frontier.clear()

    def attach(parent: Node, state: QState, frame: CompiledModel, region: str) -> str:
        nonlocal truncated
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
        if state.time is TimeTag.POINT and lyapunov:
            hit = _matching_candidate_ancestor(
                nodes, parent, state, frame, region, child_guide
            )
            if hit is not None and any(
                not certificate.admits_cycle(
                    _candidate_cycle(nodes, parent, hit, state, frame)
                )
                for certificate in lyapunov
            ):
                stats["lyapunov_cycles_filtered"] += 1
                stats["user_filtered"] += 1
                return "lyapunov"
        if seen is not None:
            key = (frame, state, region, child_guide)
            hit = seen.get(key)
            if hit is not None:
                if hit not in parent.children:
                    parent.children.append(hit)
                stats["merged"] += 1
                return "merged"
        if len(nodes) >= cfg.max_states:
            truncated = True
            stats["limit_hits"]["max_states"] += 1
            if parent.terminal is None:
                parent.terminal = TerminalClass.TRUNCATED
            return "limit"
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

        if len(nodes) >= cfg.max_states:
            stats["limit_hits"]["max_states"] += 1
            node.terminal = TerminalClass.TRUNCATED
            truncated = True
            continue
        if node.depth >= cfg.max_depth:
            stats["limit_hits"]["max_depth"] += 1
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
                and not _bounces(nodes, node, tr, vals)
            ]
            if firing:
                stats["region_crossings"] += len(firing)
                spec_killed = False
                for tr in firing:
                    stats["region_resets"] += len(tr.resets)
                    for entry in _entry_states(
                        frame, vals, tr, cfg, stats, ignored,
                        chatter_by_region.get(tr.target, frozenset()),
                    ):
                        spec_killed |= attach(node, entry, frame, tr.target) == "spec"
                if not node.children and node.terminal is not TerminalClass.TRUNCATED:
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
                    frame, state, vals, cfg, stats, ignored, active_idx,
                    chatter_by_region.get(node.region, frozenset()),
                    exclude=vals,
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
            frame, state, vals, cfg, stats, ignored, active_idx,
            chatter_by_region.get(node.region, frozenset()),
            exclude=exclude,
        )
        if children is None:  # some variable has no legal transition
            node.terminal = (
                (
                    TerminalClass.REGION_EXIT
                    if frame.explicit_regions
                    else TerminalClass.DOMAIN_EXIT
                )
                if is_point
                else TerminalClass.DEADEND
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
        if not node.children and node.terminal is not TerminalClass.TRUNCATED:
            if spec_killed:  # the spec excluded every consistent successor
                node.terminal = TerminalClass.SPEC_PRUNED
            else:  # the phase filter refuted every one: the state is spurious
                node.terminal = TerminalClass.DEADEND
                stats["deadends"] += 1

    graph = BehaviorGraph(nodes, 0, root_frame.var_order, root_frame.spaces)
    status = SimStatus.TRUNCATED if truncated else SimStatus.COMPLETE
    stats["distinct_frames"] = len({node.model for node in nodes.values()})
    if truncated:
        stats["truncation"] = _truncation_diagnostic(cfg, stats)
    return SimResult(graph, status, stats, cfg, root_frame.model_hash)


def _truncation_diagnostic(cfg: SimConfig, stats: dict) -> dict:
    """Plain-data explanation and safe next actions for a truncated run."""
    limits = [
        name for name, count in stats["limit_hits"].items() if count
    ]
    if cfg.discover_landmarks and stats["landmarks_minted"]:
        likely_cause = "landmark_growth"
        energy_enabled_discovery = any(
            "EnergyFilter" in adjustment
            for adjustment in stats.get("config_adjustments", ())
        )
        prefix = (
            "EnergyFilter enabled landmark discovery, which"
            if energy_enabled_discovery
            else "Landmark discovery"
        )
        message = (
            f"{prefix} minted {stats['landmarks_minted']} landmarks across "
            f"{stats['distinct_frames']} distinct frames before the "
            f"{', '.join(limits)} limit."
        )
        if energy_enabled_discovery:
            suggestions = [
                f"Raise max_states above {cfg.max_states} to retain the "
                "energy-filtered landmark detail.",
                "Remove EnergyFilter only if its physical premise is not "
                "required; the practical profile will then keep extrema "
                "unnamed.",
            ]
        else:
            suggestions = [
                "Use SimConfig.practical() when named intermediate extrema "
                "are not required.",
                "Use SimConfig.classic() only when full landmark discovery "
                "is intentional.",
                "Add trusted energy, Lyapunov, or phase constraints only when "
                "the model justifies them.",
            ]
    elif "max_depth" in limits and "max_states" not in limits:
        likely_cause = "depth_limit"
        message = f"Simulation reached max_depth={cfg.max_depth}."
        suggestions = [
            "Raise max_depth if the additional event history is required.",
            "Inspect repeated qualitative phases before increasing the limit.",
        ]
    else:
        likely_cause = "state_space_growth"
        message = (
            f"Simulation exhausted {', '.join(limits)} after creating "
            f"{stats['nodes']} nodes."
        )
        suggestions = []
        if not cfg.dynamic_chatter:
            suggestions.append(
                "Enable dynamic chatter abstraction with SimConfig.practical()."
            )
        suggestions.append(
            "Add trusted successor filters only when domain knowledge justifies them."
        )
    return {
        "limits_hit": limits,
        "likely_cause": likely_cause,
        "message": message,
        "suggestions": suggestions,
    }


def _bounces(
    nodes: dict[int, Node], node: Node, tr: CompiledTransition, vals: tuple[QVal, ...]
) -> bool:
    """Would this transition instantaneously return to the region the
    behavior just left, with unchanged magnitudes? Both regions' guards
    hold on a shared boundary, so without this check two boundary-crossing
    transitions ping-pong forever at the same instant (a Zeno loop). The
    behavior is already sitting on the boundary: staying put in the
    current region is the real continuation."""
    if node.parent is None:
        return False
    parent = nodes[node.parent]
    if parent.region != tr.target or parent.model != node.model:
        return False
    pvals = tuple(parent.state[v].mag for v in parent.model.var_order)
    return pvals == tuple(qv.mag for qv in vals)


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


def _filtered_combos(frame, domains, active_idx, cfg, stats):
    """Consistent assignments with workload-aware backend selection."""
    backend, reason = _select_backend(cfg, domains, active_idx)
    backend_stats = stats["backend"]
    _increment(backend_stats["selection_reasons"], reason)
    if backend == "tensor":
        try:
            from ..tensor.engine import filtered_combos
        except (ImportError, ModuleNotFoundError) as exc:
            if cfg.backend_mode == "tensor":
                raise RuntimeError(
                    "tensor backend requested but PyTorch tensor filtering "
                    "is unavailable"
                ) from exc
            backend_stats["reference_filter_calls"] += 1
            _increment(backend_stats["fallback_reasons"], "tensor_unavailable")
            return _reference_filtered_combos(frame, domains, active_idx)
        telemetry: dict = {}
        result = filtered_combos(
            frame,
            domains,
            active_idx,
            telemetry=telemetry,
        )
        backend_stats["tensor_filter_calls"] += 1
        if "fallback" in telemetry:
            _increment(backend_stats["fallback_reasons"], telemetry["fallback"])
        return result
    backend_stats["reference_filter_calls"] += 1
    return _reference_filtered_combos(frame, domains, active_idx)


def _select_backend(cfg, domains, active_idx) -> tuple[str, str]:
    """Return the backend and auditable reason for one filter workload."""
    requested = cfg.backend_mode
    if requested == "reference":
        return "reference", "explicit_reference"
    if requested == "tensor":
        return "tensor", "explicit_tensor"
    if not active_idx:
        return "reference", "auto_no_active_constraints"
    interpretations = prod(len(domain) for domain in domains)
    if interpretations < AUTO_TENSOR_PRODUCT:
        return "reference", "auto_below_threshold"
    return "tensor", "auto_at_or_above_threshold"


def _reference_filtered_combos(frame, domains, active_idx):
    active = tuple(frame.constraints[i] for i in active_idx)
    pruned = filters.prune_domains(frame, domains, active)
    if pruned is None:
        return []
    return filters.assemble(frame, pruned, active)


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _new_backend_stats(cfg: SimConfig) -> dict:
    return {
        "requested": cfg.backend_mode,
        "auto_tensor_product_threshold": AUTO_TENSOR_PRODUCT,
        "reference_filter_calls": 0,
        "tensor_filter_calls": 0,
        "selection_reasons": {},
        "fallback_reasons": {},
    }


def _entry_states(
    frame: CompiledModel,
    vals: tuple[QVal, ...],
    transition: CompiledTransition,
    cfg: SimConfig,
    stats: dict,
    ignored: frozenset[int],
    chatter: frozenset[int] = frozenset(),
) -> list[QState]:
    """Region-entry states after resets, with target directions re-derived."""
    target = transition.target
    entry_vals = list(vals)
    for vi, landmark in transition.resets:
        entry_vals[vi] = QVal(frame.spaces[vi].rank_of(landmark), Qdir.STD)
    active_idx = frame.region_named(target).constraint_idx
    domains = [
        [QVal(qv.mag, d) for d in (Qdir.DEC, Qdir.STD, Qdir.INC)]
        for qv in entry_vals
    ]
    out: list[QState] = []
    emitted: set[QState] = set()
    survivors = []
    for combo in _filtered_combos(frame, domains, active_idx, cfg, stats):
        stats["candidates"] += 1
        survivors.append(
            tuple(
                QVal(qv.mag, Qdir.IGN) if vi in ignored else qv
                for vi, qv in enumerate(combo)
            )
        )
    if chatter:
        survivors = _merge_chatter(survivors, chatter, stats)
    for projected in survivors:
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
    chatter: frozenset[int] = frozenset(),
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

    survivors: list[tuple[QVal, ...]] = []
    for combo in _filtered_combos(frame, domains, active_idx, cfg, stats):
        stats["candidates"] += 1
        if (
            next_time is TimeTag.POINT
            and cfg.infinity_filter
            and not filters.admissible_at_infinity(frame, combo)
        ):
            stats["infinity_filtered"] += 1
            continue
        survivors.append(
            tuple(
                QVal(qv.mag, Qdir.IGN) if vi in ignored else qv
                for vi, qv in enumerate(combo)
            )
        )
    if chatter:
        survivors = _merge_chatter(survivors, chatter, stats)

    out: list[tuple[QState, CompiledModel]] = []
    emitted: set[QState] = set()
    for projected in survivors:
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
                frame, vals, projected, cfg.max_landmarks_per_variable
            )
            if minted:
                stats["landmarks_minted"] += len(minted)
                child_state = QState.from_dict(
                    dict(zip(frame.var_order, minted_vals)), next_time
                )
        out.append((child_state, child_frame))
    return out


def _merge_chatter(
    combos: list[tuple[QVal, ...]], chatter: frozenset[int], stats: dict
) -> list[tuple[QVal, ...]]:
    """Dynamic chatter abstraction: successors identical except in chatter
    candidates' directions merge into one, the wiggling directions
    projected to IGN. A candidate whose direction is uniform across a
    group (filtering pinned it) stays concrete."""
    groups: dict[tuple, list[tuple[QVal, ...]]] = {}
    for combo in combos:
        key = tuple(
            (qv.mag, None if vi in chatter else qv.dir)
            for vi, qv in enumerate(combo)
        )
        groups.setdefault(key, []).append(combo)
    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        varying = {
            vi for vi in chatter if len({m[vi].dir for m in members}) > 1
        }
        out.append(
            tuple(
                QVal(qv.mag, Qdir.IGN) if vi in varying else qv
                for vi, qv in enumerate(members[0])
            )
        )
        stats["chatter_merged"] += len(members) - 1
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


def _matching_candidate_ancestor(
    nodes: dict[int, Node],
    parent: Node,
    state: QState,
    frame: CompiledModel,
    region: str,
    guide: object | None,
) -> int | None:
    """Ancestor matched by a not-yet-attached point successor."""
    pid: int | None = parent.id
    while pid is not None:
        anc = nodes[pid]
        if (
            anc.state == state
            and anc.model == frame
            and anc.region == region
            and anc.guide == guide
        ):
            return pid
        pid = anc.parent
    return None


def _candidate_cycle(
    nodes: dict[int, Node],
    parent: Node,
    ancestor: int,
    state: QState,
    frame: CompiledModel,
) -> tuple[tuple[QState, CompiledModel], ...]:
    """States on the candidate recurrence, including its closing state."""
    path: list[tuple[QState, CompiledModel]] = []
    node = parent
    while True:
        path.append((node.state, node.model))
        if node.id == ancestor:
            break
        if node.parent is None:  # defensive: ancestor came from this chain
            return ()
        node = nodes[node.parent]
    path.reverse()
    path.append((state, frame))
    return tuple(path)


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
