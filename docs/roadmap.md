# Roadmap

Phases are ordered by dependency, not calendar. Each phase ends with tests
green and docs updated; later phases never begin on top of an untested core.

## Phase 0 — Scaffolding *(this commit)*

- Design docs (`docs/`), package skeleton, core data types
  (`QuantitySpace`, `QVal`, `QState`, constraints, `Model`) with unit tests.
- Decisions recorded in `architecture.md`; unknowns in `open-questions.md`.

## Phase 1 — Reference QSIM (pure Python)

- Transition tables, constraint predicates with corresponding values,
  tuple/Waltz/global filtering, behavior tree construction.
- Global filters: no-change, quiescence, cycle match, divergence.
- `BehaviorGraph` result object with iteration + dot export.
- **Exit criteria:** bathtub, U-tube, frictionless spring golden tests match
  the literature; spring behaviors verified terminal-by-terminal.

## Phase 2 — Full-fidelity QSIM

- New-landmark introduction (per-branch quantity spaces).
- Chatter mitigation: `ignore-qdir`, then chatter-box abstraction.
- Resource limits with explicit `TRUNCATED` terminals.
- Attainable-envisionment mode (state dedup / graph instead of tree).
- Damped-spring golden test incl. spurious-behavior regression checks.

## Phase 3 — Numeric bridge, upward direction

- Trajectory abstraction pipeline (quantize → direction → segment →
  canonicalize), batched, ragged-output handling.
- Soundness harness: sample concrete instances of a model (monotone
  splines), integrate, abstract, assert containment in the QSIM graph.
  This doubles as randomized testing for Phases 1–2.
- Landmark-discovery-from-data prototype.

## Phase 4 — Tensorized engine

- State/frontier codecs (`tensor/encoding.py`), compiled constraint tables.
- Batched successor generation + filtering; fallback backtracker for
  oversized interpretation sets.
- Equivalence property tests vs. reference engine; benchmark suite
  (frontier scaling, ensemble scaling); measured GPU-vs-CPU report.

## Phase 5 — Semi-quantitative layer (Q2-style)

- Interval annotations on landmarks; envelope annotations on M-constraints.
- Interval propagation along behaviors; pruning of numerically impossible
  behaviors; guaranteed bounds output.

## Phase 6 — Model structure & beyond (exploratory)

- Operating regions / mode transitions (piecewise QDEs with guards) — also
  the landing pad for QPT-style process activation later.
- Downward-bridge consistency checker; envelope sampling as a public API.
- Candidates, order to be decided with integration details in hand:
  total envisionment, comparative analysis, temporal-logic queries over
  behavior graphs, QDE induction from abstracted trajectories, soft
  differentiable constraint losses.

## Deliberately not scheduled

GUI/model-building environments, spatial reasoning, probabilistic quantity
semantics, real-time use. Revisit only with a concrete driving use case.
