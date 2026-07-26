# Implemented research references

This page maps the research that directly shaped qrlib to the corresponding
library code, behavioral evidence, and tutorial material. It answers a narrower
question than the [annotated bibliography](references.md): **which cited ideas
are implemented, where are they implemented, and how closely does qrlib claim
to follow them?**

The labels in [`references.md`](references.md) remain authoritative:

- **Direct lineage** means an algorithm, semantics, or modeling vocabulary
  materially shaped an implemented feature.
- **Background** and **adjacent work** provide context but are not
  implementation claims, so they are not listed as implemented references
  here.
- “Implemented” does not necessarily mean a line-for-line reproduction.
  Scope notes below identify narrower implementations and qrlib-specific
  extensions.

For feature-oriented navigation, see the
[tutorial feature map](tutorial/feature-map.md). For implementation history,
see the [roadmap](roadmap.md).

## At a glance

| Reference | Implemented contribution | Primary qrlib surface |
|---|---|---|
| [Kuipers1986](references.md#kuipers1986) | QDE semantics, qualitative simulation, transitions, behavior coverage | `Model`, constraints, `qsim`, `BehaviorGraph`, `bridge.coverage` |
| [Kuipers1994](references.md#kuipers1994) | Quantity spaces, corresponding values, landmark discovery, full reference semantics | `QuantitySpace`, `CompiledModel`, `SimConfig.classic()` |
| [KuipersBerleant1988](references.md#kuipersberleant1988) | Q2-style interval refinement and behavior refutation | `qrlib.semiquant` |
| [FoucheKuipers1992](references.md#fouchekuipers1992) | Energy premises for pruning spurious oscillations | `EnergyFilter`, `LyapunovCertificate` |
| [ClancyKuipers1997Chatter](references.md#clancykuipers1997chatter) | Structural chatter detection and dynamic abstraction | `SimConfig.dynamic_chatter`, `engines.chatter` |
| [LeeKuipers1988](references.md#leekuipers1988) | Phase-space trajectory non-intersection | `SimConfig.phase_pairs`, `engines.phase` |
| [BrajnikClancy1998](references.md#brajnikclancy1998) | Temporal-logic focusing of qualitative simulation | `guide.guided` |
| [ShultsKuipers1997](references.md#shultskuipers1997) | Temporal classification over qualitative behavior graphs | `guide.classify`, `guide.verdict` |
| [SubramanianMooney1996](references.md#subramanianmooney1996) | Behavioral modes and minimal qualitative diagnosis | `diagnosis.diagnose` |
| [ClancyKuipers1997](references.md#clancykuipers1997) | Component decomposition, guidance, and compatible-history joining | `decompose.decsim` |
| [Forbus1984](references.md#forbus1984) | Process-centered modeling, proportionalities, and direct influences | `frontends.qpt` |
| [deKleerBrown1984](references.md#dekleerbrown1984) | Device composition and total envisionment | `frontends.devices`, `envision` |
| [IwasakiSimon1986](references.md#iwasakisimon1986) | Equation-based causal ordering and integral causality | `analysis.causal` |
| [ChiuKuipers1992](references.md#chiukuipers1992) | Qualitative comparative analysis | `analysis.compare` |
| [Raiman1986](references.md#raiman1986) | Order-of-magnitude negligibility | `Negligible` |
| [Harary1953](references.md#harary1953) | Signed-graph balance certificates | `analysis.monotonicity` |
| [RichardsKraanKuipers1992](references.md#richardskraankuipers1992) | Abduction of qualitative model structure from observations | `qrlib.induce` |

## Core qualitative simulation

### Kuipers1986 — QSIM semantics and sound over-approximation

**Research contribution.** Qualitative differential equations, qualitative
states, point/interval transitions, constraint filtering, behavior trees, and
the central interpretation that QSIM soundly covers real behaviors while
possibly admitting spurious ones.

**Implemented in qrlib.**

- [`quantity.py`](../src/qrlib/quantity.py), [`state.py`](../src/qrlib/state.py),
  and [`constraints.py`](../src/qrlib/constraints.py) implement qualitative
  magnitudes, directions, states, and the QSIM constraint vocabulary.
- [`engines/transitions.py`](../src/qrlib/engines/transitions.py),
  [`engines/filters.py`](../src/qrlib/engines/filters.py), and
  [`engines/qsim.py`](../src/qrlib/engines/qsim.py) implement transition
  generation, constraint filtering, and reachable simulation.
- [`behavior.py`](../src/qrlib/behavior.py) and
  [`graph.py`](../src/qrlib/graph.py) represent branching behaviors, terminal
  classes, and cycles.
- [`bridge/coverage.py`](../src/qrlib/bridge/coverage.py) turns the coverage
  interpretation into a witness-producing trajectory checker.

**Evidence and tutorial.** Golden QSIM examples are in
[`test_qsim_golden.py`](../tests/test_qsim_golden.py), transition-table
properties in [`test_transitions.py`](../tests/test_transitions.py), and
numeric trajectory coverage in
[`test_soundness.py`](../tests/test_soundness.py). Tutorial lessons
[2–5](tutorial/README.md) introduce the state and simulation semantics;
[lesson 7](tutorial/07-soundness.md) exercises the coverage claim.

**Scope.** qrlib preserves QSIM's sound-over-approximation contract; it does not
claim that every predicted behavior is physically realizable.

### Kuipers1994 — authoritative QSIM specification

**Research contribution.** The book-level specification of quantity spaces,
corresponding values, landmark introduction, global filtering, and worked
qualitative simulation patterns.

**Implemented in qrlib.**

- [`model.py`](../src/qrlib/model.py) compiles authored models into stable
  variable, quantity-space, constraint, region, and transition frames.
- [`engines/landmarks.py`](../src/qrlib/engines/landmarks.py) implements
  per-branch landmark discovery and corresponding-value propagation.
- `SimConfig.classic()` in
  [`behavior.py`](../src/qrlib/behavior.py) exposes the discovery-oriented
  textbook profile, while the practical default bounds the same semantics for
  ordinary use.

**Evidence and tutorial.** Model and predicate semantics are exercised by
[`test_model.py`](../tests/test_model.py) and
[`test_predicates.py`](../tests/test_predicates.py); landmark discovery and
cycle closure by [`test_qsim_phase2.py`](../tests/test_qsim_phase2.py).
[Lesson 3](tutorial/03-constraints-and-models.md) covers model vocabulary and
[lesson 6](tutorial/06-spurious-behaviors.md) covers discovery and filtering.

**Scope.** The public API and practical/classic profiles are qrlib engineering;
the underlying qualitative semantics follow the book.

## Semi-quantitative and global behavior refinement

### KuipersBerleant1988 — Q2-style incomplete quantitative knowledge

**Research contribution.** Interval-valued landmark knowledge, constraint
propagation, transition-time bounds, and numeric refutation of qualitatively
possible behaviors.

**Implemented in qrlib.**

- [`semiquant.py`](../src/qrlib/semiquant.py) provides `Interval`, monotone
  `Envelope` bounds, within- and cross-state propagation, `refine`,
  `refine_all`, and `feasible_behaviors`.
- [`tensor/interval.py`](../src/qrlib/tensor/interval.py) provides batched
  interval narrowing and feasibility checks with the same interval semantics.

**Evidence and tutorial.** [`test_semiquant.py`](../tests/test_semiquant.py)
checks value bounds, time bounds, and behavior pruning;
[`test_interval.py`](../tests/test_interval.py) checks interval soundness and
tensor/reference parity. See
[lesson 8](tutorial/08-semi-quantitative.md).

**Scope.** This is the Q2-style layer. qrlib does not claim to implement the
later Q3 system described by the background reference
[BerleantKuipers1997](references.md#berleantkuipers1997).

### FoucheKuipers1992 — energy-based pruning

**Research contribution.** Global energy premises that eliminate locally
consistent but globally impossible oscillatory behaviors.

**Implemented in qrlib.**

- [`energy.py`](../src/qrlib/energy.py) implements conserved and
  nonincreasing-amplitude `EnergyFilter` premises.
- The same module's `LyapunovCertificate` generalizes the scalar-premise idea
  to conditional strict decrease and recurrence pruning.

**Evidence and tutorial.** [`test_energy.py`](../tests/test_energy.py) checks
amplitude pruning, turning points, real-trajectory coverage, conditional
descent, and recurrence rejection. See
[lesson 6](tutorial/06-spurious-behaviors.md).

**Scope.** `EnergyFilter` is the direct implementation-lineage claim.
`LyapunovCertificate` is a qrlib extension built around the broader
energy/Lyapunov descent principle, not a claimed reproduction of a distinct
algorithm from the paper.

### LeeKuipers1988 — non-intersection in qualitative phase space

**Research contribution.** For autonomous systems, uniqueness prevents two
trajectories from crossing in phase space; repeated directed crossings can
therefore refute spurious qualitative paths.

**Implemented in qrlib.**

- [`engines/phase.py`](../src/qrlib/engines/phase.py) detects directed
  phase-pair crossing events and rejects provable non-intersection violations.
- `SimConfig.phase_pairs` in
  [`behavior.py`](../src/qrlib/behavior.py) opts declared variable pairs into
  the path-dependent filter.

**Evidence and tutorial.** [`test_phase.py`](../tests/test_phase.py) checks
validation, pruning effect, determinism, filter composition, and numeric
trajectory coverage. See
[lesson 13](tutorial/13-advanced-analysis.md).

**Scope.** qrlib implements the non-intersection filter, not the full
QPORTRAIT construction method described by the background reference
[LeeKuipers1993](references.md#leekuipers1993).

### ClancyKuipers1997Chatter — dynamic chatter abstraction

**Research contribution.** Static and dynamic abstraction of distinctions
whose repeated qualitative direction changes cause combinatorial chatter.

**Implemented in qrlib.**

- [`engines/chatter.py`](../src/qrlib/engines/chatter.py) detects structurally
  abstractable direction variables by operating region.
- [`engines/qsim.py`](../src/qrlib/engines/qsim.py) dynamically merges only
  distinctions that actually manifest as chatter; `track_qdir` preserves
  variables required by guides, phase pairs, or decomposition interfaces.

**Evidence and tutorial.** [`test_chatter.py`](../tests/test_chatter.py)
checks detection, no-op cases, automatic completion, composition with other
filters, and real-trajectory coverage. See
[lesson 6](tutorial/06-spurious-behaviors.md).

## Temporal reasoning, diagnosis, and decomposition

### BrajnikClancy1998 — temporally focused simulation

**Research contribution.** TeQSIM-style temporal constraints that focus
qualitative simulation on behaviors relevant to a temporal specification.

**Implemented in qrlib.**

- [`guide.py`](../src/qrlib/guide.py) defines qualitative-state atoms, Boolean
  operators, `X`, `G`, `F`, and `U`, formula progression, and `guided`.
- Guided simulation carries progressed formulas during graph generation so
  already-impossible branches can be pruned early.

**Evidence and tutorial.** [`test_guide.py`](../tests/test_guide.py) checks
progression, focusing, safety and until formulas, exogenous input, and result
export. See [lessons 9](tutorial/09-regions-and-reasoning.md) and
[13](tutorial/13-advanced-analysis.md).

**Scope.** qrlib implements the focusing semantics over its own behavior
graph and formula API; it does not claim API compatibility with TeQSIM.

### ShultsKuipers1997 — temporal classification of QSIM behaviors

**Research contribution.** Sound temporal conclusions over a qualitative
behavior graph, including universal claims that must account for every
possible qualitative behavior.

**Implemented in qrlib.**

- [`guide.py`](../src/qrlib/guide.py) implements lasso-aware `verdict` and
  post-hoc `classify` in addition to guided generation.
- `GuidedResult.universal` reports whether every represented behavior
  satisfies the specification.

**Evidence and tutorial.** The lasso, quiescent-suffix, truncation, universal,
and pure-classification contracts are covered in
[`test_guide.py`](../tests/test_guide.py). See
[lesson 13](tutorial/13-advanced-analysis.md).

**Scope.** Conclusions inherit QSIM's sound-over-approximation interpretation:
a universal property over the complete graph is sound, while truncation
remains explicitly undetermined.

### SubramanianMooney1996 — behavioral-mode diagnosis

**Research contribution.** Qualitative multiple-fault diagnosis by assigning
normal or fault modes to components, checking consistency with observations,
and preferring minimal fault sets.

**Implemented in qrlib.**

- [`diagnosis.py`](../src/qrlib/diagnosis.py) provides behavioral `Component`
  declarations, candidate checks, cardinality-ordered search, and
  `diagnose`.
- Operating-point observations use the `At` constraint from
  [`constraints.py`](../src/qrlib/constraints.py).

**Evidence and tutorial.** [`test_diagnosis.py`](../tests/test_diagnosis.py)
checks nominal exoneration, single and double faults, minimality, multiple
observations, search budgets, and serialization. See
[lesson 12](tutorial/12-learning-and-diagnosis.md).

**Scope.** This is a QDOCS-style behavioral-mode implementation over qrlib
models, not a reproduction of the paper's complete surrounding system.

### ClancyKuipers1997 — DecSIM-style decomposition

**Research contribution.** Partition a qualitative model into components,
simulate components with interface guidance, and join compatible component
histories to improve scaling.

**Implemented in qrlib.**

- [`decompose.py`](../src/qrlib/decompose.py) implements partition suggestion,
  explicit decomposition, interface streams, compatibility, component runs,
  history joining, and `decsim`.
- Interface guidance reuses [`guide.py`](../src/qrlib/guide.py); cyclic
  coupling falls back to chatter-abstracted interfaces and compatible joining.

**Evidence and tutorial.** [`test_decsim.py`](../tests/test_decsim.py) covers
partitioning, one-way guidance, cyclic fallback, stream compatibility, and
serialization. [`test_research_integration.py`](../tests/test_research_integration.py)
checks that device composition and decomposition preserve monolithic
behaviors. See
[lesson 14](tutorial/14-composition-and-scale.md).

**Scope.** The public decomposition API and its integration with qrlib's guide
and chatter machinery are library-specific.

## Modeling vocabularies and envisionment

### Forbus1984 — Qualitative Process Theory

**Research contribution.** Process-centered modeling with activation
conditions, qualitative proportionalities, and direct positive or negative
influences on quantities.

**Implemented in qrlib.**

- [`frontends/qpt.py`](../src/qrlib/frontends/qpt.py) provides `System`,
  `Process`, proportionalities, direct influences, process activation
  regions, and influence resolution.
- The frontend compiles to the same ordinary [`Model`](../src/qrlib/model.py)
  consumed by all engines.

**Evidence and tutorial.** [`test_frontends.py`](../tests/test_frontends.py)
compares generated models with hand-written QDEs, checks activation regions,
and simulates the result. See
[lesson 14](tutorial/14-composition-and-scale.md).

**Scope.** This is a compact QPT-inspired authoring frontend, not a complete
implementation of every QPT modeling construct.

### deKleerBrown1984 — device composition and envisionment

**Research contribution.** Component-centered qualitative physics and total
envisionment of the qualitatively possible states of a device.

**Implemented in qrlib.**

- [`frontends/devices.py`](../src/qrlib/frontends/devices.py) implements
  reusable component types, instances, terminal wiring, and compilation to an
  ordinary model.
- [`engines/envision.py`](../src/qrlib/engines/envision.py) enumerates the
  complete consistent-state portrait for one operating region, with
  quiescence and recurrent-component queries.

**Evidence and tutorial.** Device composition is covered by
[`test_frontends.py`](../tests/test_frontends.py), total and attainable
portraits by [`test_envision.py`](../tests/test_envision.py). See
[lessons 13](tutorial/13-advanced-analysis.md) and
[14](tutorial/14-composition-and-scale.md).

**Scope.** qrlib borrows the component/envisionment vocabulary while using
QSIM states and constraints internally; it does not claim to reproduce the
original ENVISION implementation.

## Structural and comparative analysis

### IwasakiSimon1986 — causal ordering

**Research contribution.** Derive causal direction from equations and
distinguish instantaneous causal structure from integral causality.

**Implemented in qrlib.**

- [`analysis/causal.py`](../src/qrlib/analysis/causal.py) performs structural
  equation matching, identifies exogenous and state variables, orients
  instantaneous and integration edges, condenses feedback loops, and reports
  singular or redundant structures.

**Evidence and tutorial.** [`test_causal.py`](../tests/test_causal.py) checks
textbook feedback systems, algebraic chains, integral causality, singularity,
regional models, narration, and export. See
[lesson 9](tutorial/09-regions-and-reasoning.md).

**Scope.** [deKleerBrown1986](references.md#dekleerbrown1986) is supporting
background for the relation among equation ordering, component topology, and
feedback; the direct implemented algorithmic claim is the Iwasaki–Simon
lineage.

### ChiuKuipers1992 — comparative analysis

**Research contribution.** Qualitatively determine how system behavior changes
under parameter perturbations.

**Implemented in qrlib.**

- [`analysis/compare.py`](../src/qrlib/analysis/compare.py) propagates signed
  changes through constraints at an operating point and returns increases,
  decreases, unchanged quantities, contradictions, and indeterminate results.

**Evidence and tutorial.** [`test_compare.py`](../tests/test_compare.py)
checks equilibrium shifts, monotone and additive propagation, operating-point
dependence, indeterminacy, and narration. See
[lesson 9](tutorial/09-regions-and-reasoning.md).

**Scope.** qrlib implements the narrower equilibrium comparative-statics case,
not the paper's full qualitative integral representation or transient
comparative analysis.

### Raiman1986 — order-of-magnitude negligibility

**Research contribution.** FOG-style reasoning about a quantity being
negligible relative to another.

**Implemented in qrlib.**

- `Negligible` in [`constraints.py`](../src/qrlib/constraints.py) is a
  first-class constraint with serialization, syntax, reference filtering, and
  tensor filtering.
- [`semiquant.py`](../src/qrlib/semiquant.py) additionally checks the relation
  when numeric landmark bounds are available.

**Evidence and tutorial.** [`test_oom.py`](../tests/test_oom.py) covers
declaration, closure, contradiction detection, regional semantics,
reference/tensor agreement, serialization, and trajectory coverage. See
[lesson 13](tutorial/13-advanced-analysis.md).

**Scope.** qrlib implements the relation in a sound instantaneous form, not
the full FOG reasoning system.

### Harary1953 — signed-graph balance

**Research contribution.** A signed graph is balanced exactly when its
vertices admit a two-polarity assignment satisfying every signed edge;
unbalanced graphs contain a negative cycle.

**Implemented in qrlib.**

- [`analysis/monotonicity.py`](../src/qrlib/analysis/monotonicity.py)
  translates `M+`, `M-`, and `Minus` constraints into signed relations and
  returns deterministic polarity components or a concrete negative-cycle
  witness.

**Evidence and tutorial.** [`test_monotonicity.py`](../tests/test_monotonicity.py)
checks hand-built cases, regional selection, certificate mechanics, and 120
random graphs against an exhaustive polarity oracle. See
[lesson 13](tutorial/13-advanced-analysis.md) and the
[monotonicity note](monotonicity.md).

**Scope.** This certifies consistency of the declared signed relationships. It
is not a proof that an external vector field defines a monotone dynamical
system.

## Model induction

### RichardsKraanKuipers1992 — qualitative model abduction

**Research contribution.** Infer qualitative differential model structure
from observed behavior by abducting candidate relationships and preferring
models that explain the observations.

**Implemented in qrlib.**

- [`induce.py`](../src/qrlib/induce.py) constructs a parsimony ladder of
  signed influence structures, builds candidate QDEs, checks consistency
  against trajectories, and returns ranked candidates with diagnostics.
- [`bridge/signs.py`](../src/qrlib/bridge/signs.py) provides the sign-estimation
  and calibration intake used alongside induction.

**Evidence and tutorial.** [`test_induce.py`](../tests/test_induce.py) checks
decay, independent systems, oscillators, damping, parsimony, multiple
trajectories, explicit derivatives, and validation. Sign calibration is
covered by [`test_signs.py`](../tests/test_signs.py). See
[lesson 12](tutorial/12-learning-and-diagnosis.md).

**Scope.** qrlib uses its own parsimony-ranked, data-consistency-checked
procedure. It claims the GENMODEL/MISQ/QDE-abduction lineage, not a reproduction
of MISQ. Region-dependent model learning from
[RamachandranMooneyKuipers1994](references.md#ramachandranmooneykuipers1994)
remains background rather than an implemented capability.

## Engineering dependencies are not implementation-lineage claims

[PaszkeEtAl2019](references.md#paszkeetal2019) documents PyTorch, qrlib's
tensor execution dependency. Tensor encoding, backend selection, batched
abstraction, differentiable losses, schemas, provenance records, compact
constraint syntax, and SVG rendering are qrlib engineering contributions.
They are implemented and tested, but they are not presented here as
reimplementations of qualitative-reasoning research references.

