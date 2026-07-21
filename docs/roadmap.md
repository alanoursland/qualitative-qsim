# Roadmap

Phases are ordered by dependency, not calendar. Each phase ends with tests
green and docs updated; later phases never begin on top of an untested core.
Reordered after the host-integration requirements landed
(`docs/host-integration.md`): operating regions and the coverage oracle
moved up; analysis/queries attach to the phases that make them possible.

## Phase 0 — Scaffolding *(done, revised)*

- Design docs (`docs/`), package skeleton, core data types
  (`Landmark`/`QuantitySpace`, `QVal`, `QState`, constraints, `Model`)
  with unit tests. Landmarks carry optional numeric values/bounds.

## Phase 1 — Reference QSIM (pure Python)

- Transition tables, constraint predicates with corresponding values,
  tuple/Waltz/global filtering, behavior graph construction.
- Built-in global filters: no-change, quiescence, cycle match, divergence.
- Result conventions from day one: `SimConfig`, `SimResult` with
  `status`/pruning stats/model hash, `TerminalClass` on every terminal,
  neutral graph export + dot; `graph.py` (BFS/SCC/cycles) and basic
  `analysis.queries` (reachability, quiescent states, cycles).
- **Exit criteria:** bathtub, U-tube, frictionless spring golden tests match
  the literature; results round-trip through `to_dict()`.

## Phase 2 — Full-fidelity QSIM

- New-landmark introduction (per-branch quantity spaces).
- Chatter mitigation: `ignore-qdir`, then chatter-box abstraction.
- Resource limits with explicit `TRUNCATED` terminals.
- Attainable-envisionment mode (state dedup / graph instead of tree).
- Pluggable global-filter hook (user path/state predicates — the
  energy-argument slot for killing spurious oscillation growth).
- Damped-spring golden test incl. spurious-behavior regression checks
  (spurious branches present with the analytic filter off, gone with it on).

## Phase 3 — Upward bridge + coverage oracle

- Trajectory abstraction pipeline (quantize → direction → segment →
  canonicalize), batched, ragged-output handling, explicit
  abstraction-config values.
- **Coverage oracle** with the full contract (witness / divergence
  diagnosis / embedded config), single and batched; coverage aggregate as a
  consistency score.
- Soundness harness: envelope-sample concrete instances (monotone splines),
  integrate, abstract, assert coverage — randomized property test over
  Phases 1–2. **This invariant is the exit criterion.**
- Landmark intake/dedup (`bridge.harvest`) + landmark proposal from data.

## Phase 4 — Structure intake, regions, and priors

- Sign-matrix intake → models (`bridge.signs`), incl. region-dependent
  structure; sign estimation from data with confidence masks.
- **Operating regions in core**: region declarations on `Model`, boundary
  landmark predicates, region-transition simulation, `REGION_EXIT`
  terminals, region-tagged states; mode channel respected end-to-end
  (abstraction + coverage).
- Region/guard data export per the mapping contract
  (`host-integration.md`, Surface 5).
- `Model.sign_structure()` export + downward consistency checker
  (data vs. constraints).
- Versioned JSON schema for models/results frozen at the end of this phase.

## Phase 5 — Tensorized engine

- State/frontier codecs (`tensor/encoding.py`), compiled constraint tables.
- Batched successor generation + filtering; fallback backtracker for
  oversized interpretation sets.
- Equivalence property tests vs. reference engine; benchmark suite
  (frontier scaling, ensemble scaling, abstraction throughput); measured
  GPU-vs-CPU report.

## Phase 6 — Semi-quantitative layer (Q2-style)

- Interval bounds on landmarks (already in the schema) propagated through
  constraints along behaviors; envelope annotations on M-constraints.
- Pruning of numerically impossible behaviors; guaranteed bounds output;
  envelope-vs-trajectory data export for host plotting.

## Phase 7 — Analysis polish & exploratory

- Explanation layer (`analysis.explain`): structured step records + prose.
- Viz data exports (timeline bands, tree layouts) + optional plotting extra.
- Candidates, order to be decided by demand: total envisionment,
  comparative analysis, temporal-logic queries over behavior graphs, QDE
  induction from abstracted trajectories, soft differentiable constraint
  losses.

## Deliberately not scheduled

GUI/model-building environments, spatial reasoning, probabilistic quantity
semantics, real-time use. Revisit only with a concrete driving use case.
