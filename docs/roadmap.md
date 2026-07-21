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

## Phase 3 — Upward bridge + coverage oracle *(done)*

- Trajectory abstraction pipeline (`bridge.abstraction`): quantize →
  estimate directions → segment → debounce → emit alternating
  point/interval behaviors with sample spans; explicit
  `AbstractionConfig` travels with every result. Reference implementation
  (pure Python, per-trajectory; `abstract_batch` loops — tensorized in
  phase 5); accepts any array-like incl. torch/numpy via `tolist()`.
  Crossings between samples are synthesized as point states;
  multi-landmark jumps raise (undersampling is reported, not papered
  over). Numerics that mattered: second-order one-sided endpoint
  differences (first-order is O(h)-biased exactly at critical points) and
  a value-scale floor on the relative direction threshold (constant
  variables otherwise hallucinate directions from rounding noise).
- **Coverage oracle** (`bridge.coverage`): witness path on success;
  longest-prefix, divergence index, and per-variable diagnosis on
  failure; embedded abstraction config. Observed states (root-space
  ranks) match discovered-landmark frames by containment translation.
  Prefix semantics (finite windows end mid-behavior); CYCLE closures
  followed; QUIESCENT constant continuations absorbed; matching into
  TRUNCATED frontiers succeeds vacuously with a note. `score()` gives the
  batched consistency fraction.
- **Soundness harness** (tests/test_soundness.py, the exit criterion):
  randomized monotone power-law bathtubs (equilibrium prefixes + overflow
  paths matched through to REGION_EXIT), springs over 2.2 periods
  (matched through cycle closure, against both the discovery-off graph
  and the energy-filtered discovery graph via frame translation), U-tube,
  and a fabricated violating trajectory refuted with diagnosis.
- Landmark intake by value with conflict reporting
  (`bridge.harvest.harvest_into_model`) + steady-stretch landmark
  proposal from data (`propose_landmarks`), round-trippable.

## Phase 4 — Structure intake, regions, and priors *(done)*

- **Operating regions in core**: `Model.region()` declares named
  constraint subsets; `Model.transition()` declares guarded crossings
  (conjunctions of landmark predicates on magnitudes, e.g.
  `amount == FULL and netflow > 0`). Region entry is instantaneous
  (point→point edge): magnitudes carry over, **directions re-derive**
  under the target region's constraints (the vector field may change
  discontinuously at the boundary). Nodes/behaviors/exports are
  region-tagged; cycle matching and envisionment merging are
  region-aware; a boundary without a declared transition still ends in
  `REGION_EXIT`.
- **Mode channel end-to-end**: `abstract_trajectory(modes=...)` labels
  observed states with regions (mode changes force segment boundaries;
  boundary instants belong to the region being left); coverage requires
  region agreement and skips across the instantaneous transition/entry
  double-point pairs (an observation captures the boundary as one
  instant). Wrong mode channels are refuted.
- Sign-matrix intake → models (`bridge.signs.model_from_signs`):
  Deriv + M± through named auxiliary term/sum variables; `UNKNOWN`
  entries stay unconstrained; no zero-crossing cvals are asserted (a
  sign matrix doesn't pin where influences vanish). Region-dependent
  matrices compose via the region API.
- Sign estimation from data (`estimate_signs` + `signs_with_threshold`):
  least-squares first cut with t-like per-entry confidences — exact on
  linear systems, average monotonicity on nonlinear ones; asserts signs
  or `UNKNOWN`, never a confident zero (open-questions #10 refinement
  still open).
- `Model.sign_structure()` export (monotone pairs, derivative couplings,
  sums/products, constants, corresponding values) + downward
  consistency checker (`check_consistency`: per-constraint violation
  masses against data; localizes corrupted structure).
- **Versioned schemas frozen**: `qrlib.model/v1`
  (`Model.to_dict/from_dict`, JSON round-trip preserving semantics,
  regions and landmark values included) and `qrlib.result/v1`
  (region-tagged graph export).

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
