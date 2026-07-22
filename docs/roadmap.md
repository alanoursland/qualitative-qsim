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

## Phase 5 — Tensorized engine *(done; GPU measurements pending a GPU box)*

- `tensor/encoding.py`: qcode packing, canonical `(B, 2V)` frontier
  codecs, and per-frame dense constraint tables **built by exhaustively
  evaluating the reference predicates** — agreement by construction,
  re-verified by tests; cached per (content-hashable) frame.
- `tensor/engine.py`: tensorized prune + interpretation filtering, single
  and batched-frontier (`filtered_combos_batch`, shared padded grid,
  order-preserving), with the reference generator as fallback above an
  interpretation-product cap. Activated via `SimConfig(use_tensor=True)`.
- `tensor/abstraction.py`: batched quantization/direction estimation over
  `(B, T, V)` tensors mirroring the reference arithmetic
  expression-for-expression (float64, bit-identical ranks/dirs); run
  boundaries detected in tensor land so Python touches O(runs), not O(T).
- **Equivalence tests**: identical behavior-graph exports and stats
  across nine engine configurations (goldens, discovery, energy filter,
  chatter, envisionment, regions); batched ≡ per-state ≡ reference
  (order included); abstraction parity on soundness-harness trajectories.
- **Measured (CPU, this environment)** via `benchmarks/bench_tensor.py`:
  trajectory abstraction ~×22 (0.07 → 1.6M samples/s, B=8 × T=50k);
  batched frontier filtering ×1.5 at B=2048; single-state engine
  expansion ×0.25 — the tensor path *loses* on one small model at a
  time, exactly as `docs/gpu-tensorization.md` predicted, which is why
  `use_tensor` defaults off. GPU runs of the same benchmark are the
  remaining item, pending an environment with CUDA.

## Phase 6 — Semi-quantitative layer (Q2-style) *(done)*

- `qrlib.semiquant`: interval propagation along behaviors to a fixpoint —
  within-state ADD/MINUS/MULT interval arithmetic and M± **envelope**
  forward/inverse maps; cross-state continuity/monotonic persistence,
  region-entry value/time sharing, CONSTANT intersection; and MVT time
  propagation through DERIV (region-aware throughout; discovered
  landmarks bounded by their neighbors via ordering-tightened ranges).
  Conservative closed-interval arithmetic with the standard `0·∞ = 0`
  convention; all deductions only shrink intervals, so real trajectories
  of consistent instances always lie inside the reported bounds.
- **Guaranteed bounds output** (`SemiQuantResult`, plain-data
  `to_dict()`): per-state variable bands + time bounds, ready to plot
  against numeric trajectories. Classic Q2 results reproduced exactly:
  the overflow crossing lands in `[FULL/IF, FULL/(IF−OMAX)]` (verified
  to contain a concrete instance's true crossing time), ±10% envelopes
  pin the discovered equilibrium landmark to `[1/1.1², 1/0.9²]`, and
  asymptotic equilibrium arrival is correctly unbounded.
- **Numeric refutation / pruning** (`feasible_behaviors`): behaviors
  whose intervals empty are impossible under the annotations, with the
  state/variable localized. Demonstrated in both directions: a strong
  drain (OMAX > inflow) kills the overflow and at-FULL branches; a weak
  drain (OMAX < inflow) kills both equilibria. Unannotated models are
  never refuted (sound no-op).
- Deferred: batched/tensorized interval propagation `(B, V, 2)` (rides
  the phase-5 layer when a workload demands it).

## Phase 7 — Analysis polish *(done)*

- Explanation layer (`analysis.explain`): per-step structured `Event`
  records (reaches / steadies / starts / resumes / region crossings,
  with discovered landmarks called out) + `narrate()` prose. Changes are
  compared through each node's own frame (descriptions, not raw ranks),
  so landmark discovery reads naturally ("becomes steady at amount*0, a
  newly identified value").
- Visualization (`qrlib.viz`): data-first exports — `timeline_bands`
  (per-variable qualitative timelines, with numeric bounds and time
  bounds attached when a `SemiQuantResult` is supplied) and
  `tree_layout` (layered positions with tidy first-visit ordering) —
  plus dependency-free SVG renderers (`timeline_svg`, `tree_svg`; cycle
  closures dashed). Dot export remains on `BehaviorGraph`.
- README rewritten around a verified working example; version 0.1.0a0.

## Backlog (demand-driven — pick up when a use case asks)

Fuller assessments (what each does, whether we subsume it, cost/benefit)
are in `docs/literature-survey.md`; the highest value-per-effort items,
in recommended order:

- ~~Causal ordering~~ **(done)** — `qrlib.analysis.causal`: structural
  matching + SCC over the constraint graph, with integral causality for
  `DERIV` (state variables given by integration). Reports exogenous
  inputs, the instantaneous causal chain with depth levels, integration
  feedback edges, feedback loops (SCCs), and structural singularity
  (under-/over-determined); prose via `narrate_causes`, plain-data
  `to_dict`. Per-region for multi-region models.
- ~~Model-based diagnosis~~ **(done)** — `qrlib.diagnosis`: components
  with behavioral modes (constraint sets), candidates checked by
  simulate-and-cover (the coverage oracle is the consistency check;
  refutations are the conflicts), cardinality-ordered search with
  minimal-fault-set pruning, sound-refutation / consistent-not-proven
  semantics, vacuous-evidence flagging on truncation. Added the `At`
  operating-point constraint (magnitude pinned at a landmark) — what
  makes "stuck at zero" distinguishable from "normally zero" — wired
  through predicates, schema, causal ordering, semiquant, and the
  consistency checker.
- ~~Guided simulation / exogenous inputs~~ **(done)** — `qrlib.guide`:
  LTL over qualitative states (magnitude-vs-landmark and direction atoms;
  X/G/F/U with operator sugar) progressed along every path
  (Bacchus-Kabanza); bad prefixes pruned soundly at attach time with a
  distinct `SPEC_PRUNED` terminal; exact per-terminal verdicts (lasso
  evaluation for cycles, constant-suffix for quiescent/divergent,
  finite-trace for exits, three-valued for truncation); residual formula
  in node identity gives correct temporal unrolling of loops.
  `classify()` doubles as temporal-logic model checking over any
  behavior graph (the backlog's TL-queries item). Exogenous inputs work
  as advertised: dropping `Constant(inflow)` and guiding with
  `G(inflow==IF* ∧ dir std)` reproduces the golden graph export-for-
  export. `universal` is a sound all-real-behaviors proof (empty
  violated + undetermined); satisfied behaviors are possible, not
  proven.
- ~~**Decomposition / scaling** (DecSIM)~~ **(done — `qrlib.decompose`)**:
  variable partitioner (user-declared or connected-components), one
  constraint owner per constraint (foreign variables become the
  component's interface), per-component simulation with upstream
  components guiding downstream ones — each upstream behavior's
  shared-variable episode word is compiled into a `qrlib.guide` spec, so
  DecSIM's coordination rides the TeQSIM machinery — and a post-hoc join
  on shared-variable magnitude episodes (declared-landmark projection
  dissolves per-branch discovered landmarks). Cyclic coupling falls back
  to chatter-abstracted interfaces + join only. Twin-bathtub headline:
  10 component nodes / 9 joint tuples vs 33 monolithic nodes / 23
  interleaved behaviors, coverage-tested; guided cascade: downstream
  drops from budget-blowing to 15 nodes, joint behaviors == monolithic.
- **Piecewise-affine qualitative analysis** (see `docs/piecewise-affine.md`):
  focal-point front-end + Filippov sliding-mode derivation; exact (not
  merely sound) phase portraits for PWA-structured systems. **Not an
  in-repo implementation target.** The valuable half — identifying the
  threshold partition, per-box affine fields, eigenstructure, and Filippov
  tangency — is numeric dynamical-systems modeling this library has no home
  for, and the ordinal envisionment half has no natural supply of PWA
  models here (inputs would be hand-authored). The natural owner is a
  numeric host that produces the PWA model and consumes qrlib across the
  existing bridge (thresholds → landmark values, per-box signs →
  `bridge.signs`, box adjacency → operating regions). The note stays as
  portable design for that host; the seam it describes is the deliverable,
  not an engine feature to build here.
- ~~Qualitative-phase-space non-intersection global filter (Lee &
  Kuipers)~~ **(done — `engines.phase`, `SimConfig.phase_pairs`)**: per
  declared autonomous phase pair (x, ẋ), path crossings of each directed
  grid-line transversal must be strictly monotone or exactly repeating
  (Poincaré return-map monotonicity), and a provably revisited crossing
  point closes the orbit, forcing every crossing group periodic. Prunes
  only provable violations (unseparated magnitudes never prune); states
  with all successors refuted become DEADENDs. Headlines: damped spring
  amplitude wobbles and reopened orbits pruned (16 → 6 crossing
  families); undamped spring with discovery drops 199 → 119 behaviors
  while the golden no-discovery cycle graph is untouched;
  numeric-trajectory coverage retained.
- ~~Dynamic chatter abstraction (Clancy & Kuipers)~~ **(done —
  `engines.chatter`, `SimConfig.dynamic_chatter`)**: per-region
  structural detection of direction-unanchored variables + merge-at-
  manifestation (successors identical modulo candidate directions
  collapse, wiggling directions to `IGN`; forced directions stay
  concrete). Removes the hand-written `ignore_qdir` lists: the cascade
  completes automatically (TRUNCATED → COMPLETE, 19 nodes, the golden 5
  behaviors), the damped spring is tamed to the hand-tuned budget, and
  candidate-but-never-wiggling models (U-tube) come out export-
  identical. Composes with the guide (dir-atoms auto-tracked via
  `track_qdir`), DecSIM (guided interface vars auto-tracked), and the
  phase filter (pair vars never abstracted).
- ~~Order-of-magnitude disambiguation (FOG)~~ **(done —
  `qrlib.Negligible`)**: `Negligible(small, large)` declares
  ``|small| < |large|`` everywhere (FOG's Ne in its sound instantaneous
  form). Compile closes the relation transitively (cycles are
  contradictions), and an `Add` whose operands are so ordered gains a
  dominant operand: its zero-referenced sign-sum resolves to the
  dominant sign exactly instead of forking `{-1,0,1}` — the
  perturbed-sum demo drops 3 behaviors to the 1 real one. The relation
  is itself checked (large at zero forces small to zero; an infinite
  small vs finite large is refuted), region-gated (dominance applies
  only where every region activating the Add also activates the
  declarations), rides the schema/sign-structure exports, is skipped by
  causal ordering (an inequality, not an equation), and reaches the
  tensor tables by construction.
- ~~QPT / ENVISION-confluences model front-ends~~ **(done —
  `qrlib.frontends`)**: `qpt.System` authors quantities, qualitative
  proportionalities, and processes with direct influences; influence
  resolution emits `q'` derivative variables (sole-mechanism `q' = 0`
  when nothing is active) and activation conditions compile to operating
  regions with boundary transitions — supported by a new engine Zeno
  guard suppressing instantaneous return-to-region ping-pong on shared
  boundaries. `devices.Library`/`Device` compose models by wiring
  reusable component types; connected terminals unify into shared
  variables and the netlist compiles to a `Model` identical to its
  handwritten equivalent (cascade regression). Both are pure front-ends:
  the engine consumes ordinary models.
- ~~Total envisionment~~ **(done — `qrlib.envision`)**: all consistent
  states of one region's constraint set (full per-variable domains through
  the existing Waltz/assembly machinery), connected via the engine's
  transition tables and filters — point/interval node duality
  (persistability), infinity-admissibility, identity semantics matching
  the simulator, divergent limit states terminal, quiescence as
  classification. Yields the model's full qualitative phase portrait:
  the bathtub shows all 5 equilibria (the standard initial state only
  ever reaches 2), the attainable envisionment is verified to be a
  subgraph, and the spring's oscillation is the portrait's one
  recurrent SCC. Hard `max_states` cap (a partial total envisionment is
  a contradiction in terms); `ignore_qdir` supported.
- QDE induction from abstracted trajectories (GENMODEL/MISQ lineage) —
  pairs naturally with the trajectory-abstraction pipeline.
- Comparative analysis. (Temporal-logic queries over behavior graphs:
  covered by `qrlib.guide.classify`.)
- Soft differentiable constraint losses (layered above the exact core).
- Batched/tensorized interval propagation; GPU benchmark runs (needs a
  CUDA box); first-class declarative `EnergyFilter`
  (docs/open-questions.md #7).
- Not yet assessed (worth a follow-up survey): monotone dynamical-systems
  theory as a rigor foundation for M+/M- reasoning; symbolic-abstraction /
  reachability tooling; hybrid-system falsification / conformance testing.

## Deliberately not scheduled

GUI/model-building environments, spatial reasoning, probabilistic quantity
semantics, real-time use. Revisit only with a concrete driving use case.
