"""The coverage oracle: is an observed behavior a path in a behavior graph?

The flagship validation surface (docs/host-integration.md, Surface 3): given
an abstracted trajectory and a QSIM-predicted :class:`BehaviorGraph`, decide
whether the observation is consistent with the prediction, returning a
witness path on success and a localized diagnosis on failure.

Matching semantics:

- Observed states are encoded in the model's *root* quantity spaces (the
  abstraction cannot see landmarks the engine discovered). A node covers an
  observed value when the node's (finer, per-branch-frame) magnitude lies
  inside the observed (coarser, root-space) magnitude, and directions agree
  (a node's untracked ``IGN`` direction matches anything).
- The observed sequence may start anywhere in the graph and is treated as a
  **prefix**: a finite sample window ends mid-behavior (equilibrium arrival
  is the t->infinity limit and is never observed exactly).
- CYCLE closures are followed (matching continues from the cycle target's
  successors), QUIESCENT nodes absorb constant continuations, and matching
  into a TRUNCATED node succeeds vacuously (an unexplored frontier cannot
  refute an observation) — reported in ``note``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from ..behavior import BehaviorGraph, Node, TerminalClass
from ..quantity import Qdir, QuantitySpace, QVal
from ..state import QState
from .abstraction import AbstractedBehavior, AbstractionConfig

__all__ = ["CoverageResult", "check", "score"]


@dataclass(frozen=True)
class CoverageResult:
    """Outcome of one coverage check.

    ``witness`` is the matched node-id path (through cycle re-entries, ids
    may repeat). On failure, ``divergence_index`` is the first observed
    state that could not be matched and ``diagnosis`` localizes the
    mismatch. ``note`` records vacuous/absorbed endings. ``config`` embeds
    the abstraction parameters the observation was produced with — a
    coverage claim is not interpretable without them.
    """

    covered: bool
    witness: tuple[int, ...] | None
    matched: int
    total: int
    divergence_index: int | None = None
    diagnosis: str | None = None
    note: str | None = None
    config: AbstractionConfig | None = None


def check(
    observed: AbstractedBehavior | Sequence[QState], graph: BehaviorGraph
) -> CoverageResult:
    if isinstance(observed, AbstractedBehavior):
        states = observed.states
        cfg = observed.config
    else:
        states = tuple(observed)
        cfg = None
    total = len(states)
    if total == 0:
        return CoverageResult(True, (), 0, 0, config=cfg)

    parents: dict[tuple[int, int], tuple[int, int] | None] = {}
    frontier: deque[tuple[int, int]] = deque()
    best_matched = 0
    # (observed index, candidate node id, mismatched variable names)
    best_miss: tuple[int, int, tuple[str, ...]] | None = None

    def note_miss(oi: int, nid: int, mism: tuple[str, ...]) -> None:
        nonlocal best_miss
        if (
            best_miss is None
            or oi > best_miss[0]
            or (oi == best_miss[0] and len(mism) < len(best_miss[2]))
        ):
            best_miss = (oi, nid, mism)

    for nid, node in graph.nodes.items():
        mism = _mismatches(node, states[0], graph)
        if not mism:
            key = (0, nid)
            parents[key] = None
            frontier.append(key)
        else:
            note_miss(0, nid, mism)

    def witness_of(key: tuple[int, int]) -> tuple[int, ...]:
        path: list[int] = []
        cur: tuple[int, int] | None = key
        while cur is not None:
            path.append(cur[1])
            cur = parents[cur]
        return tuple(reversed(path))

    while frontier:
        key = frontier.popleft()
        oi, nid = key
        node = graph.nodes[nid]
        best_matched = max(best_matched, oi + 1)

        if oi == total - 1:
            return CoverageResult(True, witness_of(key), total, total, config=cfg)
        if node.terminal is TerminalClass.TRUNCATED:
            return CoverageResult(
                True,
                witness_of(key),
                oi + 1,
                total,
                note="matched into a truncated frontier; remaining states are "
                "unverifiable (an unexplored graph cannot refute them)",
                config=cfg,
            )
        if node.terminal is TerminalClass.QUIESCENT and _quiescent_absorbs(
            node, states[oi + 1 :], graph
        ):
            return CoverageResult(
                True,
                witness_of(key),
                total,
                total,
                note="constant quiescent continuation",
                config=cfg,
            )

        for snid in _successors(graph, node):
            mism = _mismatches(graph.nodes[snid], states[oi + 1], graph)
            if not mism:
                skey = (oi + 1, snid)
                if skey not in parents:
                    parents[skey] = key
                    frontier.append(skey)
            else:
                note_miss(oi + 1, snid, mism)

    diagnosis = None
    if best_miss is not None:
        oi, nid, mism = best_miss
        diagnosis = (
            f"first divergence at observed state {oi} "
            f"({graph.describe_state(states[oi])}): closest graph candidate "
            f"n{nid} differs in {', '.join(mism) if mism else 'time tag'}"
        )
    return CoverageResult(
        False,
        None,
        best_matched,
        total,
        divergence_index=best_matched,
        diagnosis=diagnosis,
        config=cfg,
    )


def score(
    observations: Sequence[AbstractedBehavior | Sequence[QState]],
    graph: BehaviorGraph,
) -> tuple[float, tuple[CoverageResult, ...]]:
    """Fraction of observations covered, with per-observation results.

    The aggregate is the qualitative-consistency score used for model
    validation and selection (docs/host-integration.md, Surface 3)."""
    results = tuple(check(obs, graph) for obs in observations)
    fraction = (
        sum(1 for r in results if r.covered) / len(results) if results else 1.0
    )
    return fraction, results


# --- matching internals ----------------------------------------------------


def _successors(graph: BehaviorGraph, node: Node) -> list[int]:
    succ = list(node.children)
    if node.cycle_target is not None:
        succ.extend(graph.nodes[node.cycle_target].children)
    return succ


def _frame_range(
    obs: QVal, root_space: QuantitySpace, frame_space: QuantitySpace
) -> tuple[int, int]:
    """Inclusive frame-rank range consistent with an observed root value."""
    eff_root = root_space.effective_landmarks
    eff_frame = frame_space.effective_landmarks
    if obs.mag % 2 == 0:
        r = 2 * eff_frame.index(eff_root[obs.mag // 2])
        return (r, r)
    lo = 2 * eff_frame.index(eff_root[obs.mag // 2])
    hi = 2 * eff_frame.index(eff_root[obs.mag // 2 + 1])
    return (lo + 1, hi - 1)


def _mismatches(
    node: Node, obs: QState, graph: BehaviorGraph
) -> tuple[str, ...]:
    """Names of variables where the node fails to cover the observed state
    (empty = covers). A time-tag mismatch reports as ``('<time>',)``."""
    if node.state.time is not obs.time:
        return ("<time>",)
    out = []
    for vi, name in enumerate(graph.var_order):
        nq = node.state[name]
        oq = obs[name]
        lo, hi = _frame_range(oq, graph.spaces[vi], node.model.spaces[vi])
        if not (lo <= nq.mag <= hi):
            out.append(name)
        elif nq.dir is not Qdir.IGN and nq.dir != oq.dir:
            out.append(name)
    return tuple(out)


def _quiescent_absorbs(
    node: Node, rest: Sequence[QState], graph: BehaviorGraph
) -> bool:
    """A quiescent state persists forever: the rest of the observation must
    hold the same values, steady."""
    for obs in rest:
        for vi, name in enumerate(graph.var_order):
            nq = node.state[name]
            oq = obs[name]
            lo, hi = _frame_range(oq, graph.spaces[vi], node.model.spaces[vi])
            if not (lo <= nq.mag <= hi):
                return False
            if oq.dir is not Qdir.STD:
                return False
    return True
