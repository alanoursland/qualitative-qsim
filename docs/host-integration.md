# Embedding qrlib in a dynamical-systems toolkit

`qrlib` is a general-purpose qualitative reasoning library, but it is designed
to be *embeddable*: a larger numeric/symbolic dynamical-systems stack should
be able to build a thin adapter module on top of it and get a full QR
capability — abstraction of its models into QDEs, behavior enumeration,
validation of numeric/learned models against qualitative structure, and
priors for system identification — without qrlib knowing anything about that
stack.

This document fixes the **requirements** such hosts impose and the **seams**
qrlib exposes to meet them. It is the result of surveying what mature
dynamical-systems toolkits typically already have (symbolic system
representations with Jacobians and equilibrium finding, ODE solvers and
trajectory objects, hybrid automata with guarded modes, trajectory
classifiers, system-identification pipelines, evidence/provenance record
formats, pluggable visualization) and asking what a QR engine must provide —
and must *not* duplicate — to slot in beside them.

## Division of labor

| Host toolkit owns | qrlib owns |
|---|---|
| System representations (ODEs, symbolic expressions, learned models) | QR semantics: quantity spaces, constraints, QDE models, operating regions |
| Symbolic analysis: differentiation, Jacobians, equilibrium solving, sign derivation via its CAS | Behavior enumeration engines (QSIM reference + tensorized), filters, soundness guarantees |
| Numeric solvers, trajectory/event machinery, precise crossing refinement | Trajectory abstraction from sampled tensors and host-supplied refined event states; coverage checking; sign estimation from data |
| Hybrid-automaton runtime (modes, guards, execution) | Region/mode-structure as data + the mapping contract to guarded-mode form |
| Identification/regression machinery | Exported sign/monotonicity structure and qualitative-consistency scores as plain data |
| Provenance/evidence record formats, reporting | Stable, serializable result objects with witnesses, statuses, and config capture |
| Rendering backends, plotting style | Render-agnostic graph/timeline data exports (+ optional built-in plotting extra) |

Two hard rules follow:

1. **No CAS dependency in qrlib.** Symbolic expressions never cross the seam.
   Where a host would use symbolic knowledge (signs of ∂f/∂x, equilibrium
   values), the seam accepts the *conclusions* — sign matrices, numeric
   landmark values, orderings — not the expressions. Hosts keep symbolic
   objects on their side, keyed by qrlib names (below).
2. **Tensors are the only numeric interchange.** Trajectories, sample sets,
   sign matrices, and interval bounds cross the seam as plain tensors/arrays
   with documented layouts (`docs/numeric-bridge.md`).

## Surface 1 — Model identity and authoring

- **Names are canonical.** Variables and landmarks are identified by strings.
  A host that models with symbolic objects keeps a `symbol ↔ name` registry
  on its side; every qrlib artifact (states, behaviors, results) refers to
  names only, so round-tripping is lossless for the host.
- **Landmarks carry optional numeric knowledge.** A `Landmark` is a name plus
  optionally: an exact numeric `value` (when the host knows it, e.g. a
  computed equilibrium level) and/or interval bounds `(lo, hi)` (for
  semi-quantitative reasoning, or when the host only has an estimate). All
  purely-qualitative machinery ignores these; abstraction, coverage, and
  interval propagation use them. A landmark whose value the host knows only
  symbolically is passed with `value=None` (or numeric bounds) — the symbolic
  form stays host-side under the same name.
- **Models are programmatically authored, serializable data.** The `Model`
  schema (variables, spaces, constraints, corresponding values, regions) has
  a versioned JSON form so hosts can generate, store, and ship QDEs as
  artifacts. The convenience text syntax for constraints
  (`"M+(level, outflow)"`) is implemented as a safe, literal-only parser into
  the existing constraint objects; it never evaluates code and does not
  change the model schema (`docs/constraint-syntax.md`).
- **Corresponding values and orderings** are first-class in the schema — they
  are the main channel by which host knowledge (e.g. "f(0)=0", "the two
  equilibria are ordered") sharpens qualitative predictions.

## Surface 2 — Abstraction inputs (host model → QDE)

Hosts want `their_system → Model` with minimal ceremony. qrlib cannot (and
should not) differentiate the host's models, so the seam is defined at the
level of *structure summaries*:

- **Sign-structure intake** (`qrlib.bridge.signs`): the primary input is an
  **interaction sign matrix** `S[i][j] ∈ {-1, 0, +1, UNKNOWN}` giving the
  sign of ∂(dx_i/dt)/∂x_j — however the host obtained it (CAS with
  assumptions, manual domain knowledge, data). qrlib turns it into a QDE:
  `Deriv` constraints plus `M+`/`M-` constraints through named auxiliary
  variables, with `UNKNOWN` entries left unconstrained (more branching,
  still sound). Region-dependent structure is passed as a list of
  (region-condition, matrix) pairs and produces an operating-region model
  (Surface 5).
- **Sign estimation from data** (fallback the host may not have): given
  trajectory or `(x, dx/dt)` sample tensors, qrlib estimates the sign matrix.
  The calibrated path reports deterministic bootstrap sign agreement in
  `[0, 1]` with explicit seed/resample metadata and maps unstable or fitted
  zero effects to `UNKNOWN`. This is a sample-stability measure, not a
  posterior truth probability. Hosts with a CAS will prefer their exact
  route; the estimator also serves as a cross-check on it.
- **Landmark intake** (`qrlib.bridge.harvest`): hosts contribute candidate
  landmark values from wherever they get them — equilibrium finders, guard
  thresholds, nullcline intersections, domain knowledge — as
  `(variable, name, value?, bounds?)` records. qrlib deduplicates, orders,
  and inserts them into quantity spaces, and reports conflicts (two named
  landmarks that cannot be ordered) instead of guessing. qrlib additionally
  proposes landmarks from data (steady-point clustering in trajectory
  batches) as *suggestions* the host can accept.

## Surface 3 — Validation outputs (the coverage oracle)

The single most valuable host-facing product: **is this numeric behavior
consistent with this qualitative model?** Used to validate learned/neural
surrogates against known structure, to sanity-check identified models, and
to test qrlib itself.

- `qrlib.bridge.coverage.check(trajectory_or_behavior, graph) → CoverageResult`
  with a strict contract:
  - `covered: bool`;
  - `witness`: the matching path (sequence of state ids) when covered — the
    positive evidence a host provenance system can record;
  - on failure: the longest matched prefix, the first divergent segment
    (with its time-index range in the source trajectory), and a diagnosis of
    *what* diverged (which variable's magnitude/direction, and which
    constraint or transition rule excludes it);
  - the abstraction parameters used (landmark tolerance, direction
    hysteresis, debounce) — coverage claims are meaningless without them, so
    they are embedded in the result, not left implicit.
- **Batched form**: `(B, T, V)` trajectories in → per-trajectory coverage
  mask + aggregate statistics out, so "what fraction of this model's
  rollouts are qualitatively consistent?" is one call. The aggregate is
  directly usable as a **model-selection score** in identification loops.
- All results are frozen, `to_dict()`-serializable objects that carry the
  model hash and engine config that produced the behavior graph — designed
  to be wrapped verbatim into host evidence/provenance records (claim,
  scope, assumptions, witness).

## Surface 4 — Behavior graphs: queries and export

Hosts have graph machinery (mode topologies, reachability, SCC/cycle
analysis) and will want to reuse it; hosts also shouldn't *need* it for
common questions.

- **Built-in queries** (`qrlib.analysis.queries`, no external graph deps):
  reachability from initial states, quiescent-state enumeration (equilibrium
  candidates, keyed by variable magnitudes so hosts can cross-link them to
  their own equilibrium analyses), cycle enumeration (oscillation
  candidates), and path-predicate filtering ("behaviors that never reach
  landmark FULL").
- **Terminal classification enum** on every leaf/closure:
  `QUIESCENT | CYCLE | DIVERGENT | REGION_EXIT | TRUNCATED`. The enum is
  stable and documented with its intended mapping onto the common
  trajectory-classification vocabulary (quiescent ≈ converging, cycle ≈
  limit cycle, divergent ≈ unbounded), so host classifiers and qrlib speak
  translatable dialects.
- **Neutral export**: `BehaviorGraph.export()` yields plain arrays — node
  table (encoded states + labels), edge list, initial/terminal marks — the
  lowest-common-denominator form every host graph library can ingest.
  Dot/graphviz export rides on the same data. Rendering conventions
  (`qrlib.viz`) produce data first (timeline bands per variable, tree
  layouts) with plotting as an optional extra, so hosts can restyle in
  their own visualization stacks.

## Surface 5 — Operating regions ↔ hybrid modes

Piecewise qualitative models (a tank that overflows, a valve that opens) and
hybrid-automaton modes are the same idea at different rigor. Requirements
this places on qrlib core (not an afterthought — promoted to an early phase):

- A `Model` may declare **regions**: each with its own active constraint
  subset, boundary conditions expressed as landmark predicates
  (`level == FULL`, `netflow > 0`), and a region-transition map. The engine
  simulates across region changes and tags states with their region;
  `REGION_EXIT` terminals appear where no transition is declared.
- The region structure is exported as data (regions, boundary predicates
  over named landmarks, transition map) in a shape that maps one-to-one onto
  guarded-mode representations: region → mode, landmark predicate →
  threshold guard on the numeric landmark value. The mapping is a host-side
  adapter, but qrlib guarantees the export contains everything needed
  (including numeric landmark values when known) to build it mechanically —
  and, in reverse, that a host's per-mode structure summaries + guard
  thresholds are sufficient input to assemble a region model via Surfaces
  1–2.
- Trajectory abstraction accepts an optional **mode channel** `(B, T)` of
  integer region tags alongside the trajectory tensor, so behaviors of
  hybrid executions abstract into region-tagged qualitative behaviors and
  coverage checking respects mode sequences.
- Event-aware hosts may pass `CrossingEvent` records containing the precise
  crossing time and complete solver state. qrlib validates the declared
  landmark, preserves original sample-index spans, and exposes exact
  physical-time bounds for those point states. Inferred crossings retain an
  enclosing time bracket instead.

## Surface 6 — Priors for system identification

The downward direction: hand-authored or abstracted qualitative models as
*constraints on model fitting*.

- `Model.sign_structure() → SignStructure`: the closure of what the
  constraints imply — per-pair signs of ∂y/∂x, derivative couplings, zero
  crossings pinned by corresponding values — as plain data. A host maps this
  onto its regression parameterization (e.g. sign constraints on library
  coefficients) itself; qrlib does not know what a coefficient is.
- **Post-fit checking before constrained fitting**: the cheap integration is
  running Surface 3 coverage (and Surface 2 sign estimation) on a fitted
  model's rollouts and reporting violations; constrained regression is a
  host concern that consumes the same exported structure.
- **Deferred by design**: differentiable relaxations of qualitative
  constraints (soft losses for training) remain out of the near-term scope;
  the exact, discrete semantics stay the product. The export format is
  chosen so a relaxation layer could be added above it without touching core
  (see `docs/open-questions.md`).

## Cross-cutting conventions the seams rely on

- **Frozen result objects** with `status` fields (`COMPLETE | TRUNCATED`),
  explicit per-filter pruning statistics, and `to_dict()` — truncation and
  filtering are always reported, never silent, because a host treating a
  truncated behavior set as exhaustive would void the soundness guarantee it
  is buying.
- **Config objects** (`SimConfig`: limits, filter toggles, region policy;
  abstraction configs: tolerances, hysteresis, debounce) are values,
  captured inside the results they produced.
- **Determinism**: same model + config → identical graphs, across runs and
  devices; state ids are content-derived (encoded-state hashes), so host
  records referencing them stay valid.
- **Explanation** (`qrlib.analysis.explain`): behaviors render to structured
  step records (variable, from, to, driver constraints) with a prose
  formatter on top — hosts embed the structured form in reports and restyle
  the text freely.

## What this changes elsewhere in the design

- `Landmark` becomes an object with optional `value`/bounds (was: bare
  names) — reflected in `docs/architecture.md` and the skeleton.
- Operating regions move up the roadmap (phase 4, was exploratory phase 6),
  and the coverage oracle becomes the explicit exit criterion of the
  upward-bridge phase — see `docs/roadmap.md`.
- New planned packages: `qrlib.analysis` (queries, explanation) and
  `qrlib.bridge.signs` / `qrlib.bridge.harvest` / `qrlib.bridge.coverage`.
- Several former open questions are now resolved — see
  `docs/open-questions.md`.
