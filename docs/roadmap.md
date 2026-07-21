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

## Phase 1 — Reference QSIM (pure Python) *(done)*

- Transition tables, constraint predicates with corresponding values
  (including the algebra of infinite landmarks in `ADD`), tuple/Waltz
  filtering, global interpretation assembly, behavior graph construction.
- Built-in global filters: no-change, cycle match, quiescence handling
  (with departure exploration for unstable equilibria), region-exit
  detection, and the infinity-admissibility rule (a point state at an
  infinite landmark is the t→∞ limit; every variable must be steady or at
  infinity there). Toggling the infinity filter off restores the classic
  spurious reach-infinity-in-finite-time behaviors — kept as a regression
  test.
- Result conventions: `SimConfig`, `SimResult` with `status` + stats,
  `TerminalClass` on every terminal, neutral graph export + dot;
  `graph.py` (reachability, Tarjan SCC) and `analysis.queries`
  (terminal census, quiescent states, cycles, state search).
- **Exit criteria met:** bathtub (3 behaviors: equilibrium below FULL, at
  FULL, overflow region-exit), U-tube (single equilibrium behavior), and
  frictionless spring (single sustained oscillation, 8-transition cycle)
  match the literature.
- Deferred within phase: model hash in results (with the phase-4 schema);
  `to_dict()` round-trip tests.

## Phase 2 — Full-fidelity QSIM *(done)*

- New-landmark introduction (default on): I5/I9 arrivals mint named
  landmarks (`x*0`, …) into per-branch **frames** (compiled models with
  grown spaces and rank-shifted constraint references); minting also
  records new corresponding values for M±/MINUS/ADD constraints whose
  variables all sit at landmarks. `max_landmarks` caps per-variable
  discovery (beyond it, steadiness stays unnamed — sound).
- Chatter mitigation: `ignore_qdir` — untracked directions (`Qdir.IGN`)
  generated over all concrete directions, filtered normally, then
  projected and merged (damped spring: 403 truncated nodes → 15 complete).
  Automatic chatter-box *detection* deferred.
- Successor-filter hook (`SimConfig.successor_filters`): user vetoes over
  `(parent, candidate, frame)`. Regression-tested with the classic energy
  argument: undamped spring with discovery branches intractably
  (spurious growing/shrinking amplitudes, truncation); with the energy
  filter it completes as the single true cycle (17 nodes: one
  landmark-discovery period, then closure).
- Attainable-envisionment mode (`SimConfig.envisionment`): global
  (frame, state) merging; cycles become back-edges enumerated by
  `behaviors()` (spring: 8-node cycle graph).
- Cycle matching is frame-aware (states in different quantity spaces never
  match). Resource limits with explicit `TRUNCATED` terminals were done in
  phase 1.

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
