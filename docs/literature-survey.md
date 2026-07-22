# QR literature survey: what else is worth incorporating

A survey of qualitative-reasoning techniques *not* currently in the library,
each assessed for what it does, whether something we already have subsumes
it, and what incorporating it would take and buy. Findings are ranked by
value, weighting **new capability classes** (diagnosis, guided simulation,
decomposition/scaling, causal reasoning, phase-space reasoning) over
incremental variants of what exists.

## How this was produced

Multi-source literature search with adversarial verification: each claim
below was checked by independent skeptics against primary sources (AAAI/
IJCAI proceedings, Springer/IEEE journals, author archives) and kept only on
a majority-confirm vote. The ten findings here survived that filter; the
honest coverage gaps are listed in §"Not yet assessed". Everything is
assessed against the library as of v0.1.0a0 (full QSIM + landmark discovery,
chatter abstraction, analytic filters, attainable envisionment, operating
regions; the numeric bridge + coverage oracle; sign-structure intake; Q2
semi-quantitative refinement; explanation; the tensor engine).

## Governing caveat (shapes everything below)

QSIM's guarantee is **one-directional** (the Guaranteed-Coverage Theorem):
it predicts *every* real behavior but may also emit spurious ones. So
anything built on the behavior graph — model-checking, diagnosis, guided
simulation — can soundly prove **universal** properties ("in all behaviors
…") but **not existential** ones ("there exists a behavior …"), because a
witnessing behavior might be spurious. The coverage oracle already lives on
the sound side of this line (it refutes, it doesn't certify existence). Any
capability added below inherits this asymmetry and should state which side
it reasons on.
*Kuipers, "Qualitative Simulation" (encyclopedia article, 2001).*

---

## Tier 1 — high-value new capability classes

### 1. Decomposition / scaling — DecSIM — **implemented**

> Built in `qrlib.decompose` (variable partitioner, per-constraint
> ownership with interface variables, upstream-guided component runs via
> `qrlib.guide` words, episode-sequence join with terminal-aware suffix
> semantics). The rest is the original assessment.

**What it does.** Partitions a model's variables into components (tightly
coupled together, loosely coupled apart), simulates each component
separately with its own state graph, and reconciles cross-component
influences through a separate "interacting histories" representation
consulted only when needed. This attacks QSIM's core scaling wall: `n`
simultaneously-moving-but-unconstrained variables produce up to `2^n`
successor states; events internal to one component no longer force branches
in the others, giving exponential reduction on decomposable models while
yielding a behavior set equivalent to monolithic QSIM (modulo
cross-component event ordering).
*Clancy & Kuipers, "Model decomposition and simulation: a component-based
qualitative simulation algorithm," AAAI-97.*

**Subsumed by us?** No. Our tensor engine makes each expansion *faster* but
does nothing about the *number* of states; this is the one technique that
attacks the combinatorial ceiling itself.

**Incorporation.** Medium-large. Needs (a) a partitioner over the
constraint graph (connected-components / tight-coupling heuristic), (b) a
per-component simulation loop — the existing engine, restricted to a
variable subset with the rest as exogenous, (c) an interacting-histories
coordinator that re-expands a component only when a partner's event can
affect it. The behavior-graph and frame machinery mostly carry over; the new
work is the coordinator and its equivalence proof against the monolithic
result. **Buys:** models that currently blow the state budget become
tractable — the single biggest lever on what size of system the library can
handle. Highest-priority addition.

### 2. Model-based diagnosis — QDOCS — **implemented**

> Built in `qrlib.diagnosis` (mode-carrying components, coverage-oracle
> consistency, cardinality-ordered minimal-diagnosis search) plus the
> `At` operating-point constraint. The rest is the original assessment.


**What it does.** Consistency-based multiple-fault diagnosis over
*continuous dynamic* systems: attach behavioral/fault modes to components,
then given observations find the minimal sets of mode assignments under
which the qualitative model is consistent with what was seen. Extends the
GDE/Sherlock lineage (static logic circuits) to dynamics by checking
temporal consistency of an observed behavior against the model and
propagating conflicts to candidate fault sets.
*Subramanian & Mooney, "Qualitative multiple-fault diagnosis of continuous
dynamic systems using behavioral modes," AAAI-96 / IJCAI-95.*

**Subsumed by us?** No — but we already have the hard half. The **coverage
oracle** ("is this observed behavior a path in the predicted graph?") is
exactly the consistency check diagnosis needs; today it returns a localized
refutation, which is a *conflict* in diagnosis terms.

**Incorporation.** Medium. Add (a) fault-mode model variants (a component
with alternative constraint sets — the operating-regions machinery already
expresses "different constraints active in different modes"), (b) a
conflict→candidate layer (hitting-set / GDE-style candidate generation) fed
by coverage refutations across mode assignments. **Buys:** a genuinely new
product-grade capability — "given these sensor traces, which components are
faulty, and in what mode?" — reusing the coverage oracle and regions we
already shipped. One of the highest-value additions, and the kind of
capability that differentiates tooling rather than replicating textbook
simulation.

### 3. Guided / constrained simulation — TeQSIM — **implemented**

> Built in `qrlib.guide`: formula progression interleaved with the agenda
> loop (sound bad-prefix pruning, `SPEC_PRUNED` terminals), exact lasso /
> constant-suffix / finite-trace verdicts, `classify()` as standalone
> temporal-logic model checking. The rest is the original assessment.


**What it does.** Treats **temporal-logic trajectory constraints** as part
of the model: interleaves simulation with model-checking so only behaviors
satisfying an externally-supplied linear-temporal-logic specification (plus
the QDE constraints and continuity) are generated. Its three-valued variant
couples the temporal guidance with semi-quantitative refinement. Enables:
focusing a large simulation on behaviors of interest, **non-autonomous /
piecewise-continuous** systems (time-varying exogenous inputs expressed as
temporal constraints), boundary-condition problems, and folding observations
into the simulation.
*Brajnik & Clancy, "Focusing Qualitative Simulation Using Temporal Logic,"
Annals of Mathematics and AI (1998).*

**Subsumed by us?** No. Our `successor_filters` are *internal, physics-based*
vetoes (e.g. energy). TeQSIM adds *external, specification-based* guidance
over whole trajectories — a different mechanism (a temporal spec + a
behavior-graph model-checker), and the standard way to handle exogenous
time-varying inputs, which the current engine has no first-class story for.

**Incorporation.** Medium. Add (a) a small LTL-over-behaviors parser, (b) an
incremental model-checker that prunes the frontier as behaviors violate the
spec — this rides on the existing agenda loop and the `analysis.queries`
graph machinery. **Buys:** steer-able simulation, exogenous inputs, and
observation-driven pruning. Sound for universal conclusions (see governing
caveat); an existential match must be treated as "possible", not "proven".

---

## Tier 2 — new reasoning layers

### 4. Causal ordering (Simon; Iwasaki & Simon) — **implemented**

> Built in `qrlib.analysis.causal` (structural matching + SCC, integral
> causality for `DERIV`). The rest of this section is the original
> assessment.


**What it does.** Derives *causal* structure — which variable determines
which — purely from the structural equations, by finding minimal
self-contained equation subsets and ordering them. Concrete algorithm: find
a minimal variable set whose equations mention only those variables, pick
the subset maximizing `|E|−|V|`, "plunk" (fix) a variable, and iterate; for
`n` independent equations in `n` unknowns this recovers Simon's causal
order. Feedback loops correctly yield *no* internal ordering.
*Iwasaki & Simon (1986); de Kleer & Brown; "Theories of Causal Ordering".*

**Subsumed by us?** No. Our explanation layer narrates *what* changes, not
*what causes what*. Causal ordering is a distinct structural analysis over
the model, independent of simulation.

**Incorporation.** Small-medium, and it's a self-contained algorithm over
the compiled constraint graph we already have. **Buys:** a causal-structure
export (and a much stronger explanation layer — "inflow rising *causes*
level to rise *causes* outflow to rise"). Caveat: equation-based ordering
gives nothing inside feedback loops; a component-topology method (ENVISION,
below) is the alternative there.

### 5. Order-of-magnitude reasoning — FOG — **Ne implemented**

> Built as `qrlib.Negligible(small, large)`: FOG's Ne relation in its
> sound instantaneous form (``|small| < |large|`` everywhere),
> transitively closed at compile with contradiction detection, feeding
> the ADD sign algebra exactly as assessed below (a dominant operand
> resolves the zero-referenced `{-1,0,1}` fork), checked as a constraint
> in its own right, and excluded from causal ordering. `Vo`/`Co` and the
> wider rule set remain unimplemented — add them if a model needs more
> than dominance. The rest is the original assessment.

**What it does.** A disambiguation layer *above* sign reasoning: operators
`Ne` (negligible vs.), `Vo` (close to), `Co` (same sign and order), with
~30 inference rules. Resolves ambiguities pure sign algebra can't — the
canonical example: a sign-only analysis of a large-mass/small-mass elastic
collision leaves five possibilities; adding `m Ne M` derives the unique
physical answer.
*Raiman, "Order of Magnitude Reasoning," AAAI-86.*

**Subsumed by us?** Partly overlaps the semi-quantitative layer (interval
bounds also disambiguate), but FOG works *symbolically* from ordering
assumptions without needing numeric bounds — usable exactly where numbers
are unavailable.

**Incorporation.** Medium. A relation store (`Ne`/`Vo`/`Co` among
quantities) + a rule-propagation pass feeding the ADD sign algebra (the
current source of `{-1,0,1}` ambiguity in `filters._qsum`). **Buys:**
branch reduction in envisionment where sign reasoning currently forks, with
no numeric input required.

### 6. Qualitative phase-space reasoning (Lee & Kuipers) — **non-intersection implemented**

> The non-intersection constraint is built in `engines.phase`
> (`SimConfig.phase_pairs`): per declared autonomous pair (x, ẋ),
> grid-line crossings along each path must be monotone or exactly
> repeating per directed transversal, plus the closure rule (a provably
> revisited crossing point makes the orbit periodic, so every crossing
> group must be all-equal). QPORTRAIT remains unimplemented (optional).
> The rest is the original assessment.

**What it does.** Two phase-portrait results: a global
**non-intersection-of-trajectories** constraint in qualitative phase space
(a real trajectory can't cross itself except at closure — a sound global
pruning rule), and **QPORTRAIT**, constructing 2-D phase portraits from a
QDE.
*Lee & Kuipers, AAAI-88 (non-intersection); Lee & Kuipers, AAAI-93
(QPORTRAIT).*

**Subsumed by us?** No. The non-intersection constraint is a *global path*
property; our filters are largely local/per-successor.

**Incorporation.** The non-intersection rule is a medium-effort global
filter over the behavior graph (a new kind — path-level, not
successor-level). QPORTRAIT is a larger, narrower (2-D) construction.
**Buys:** extra spurious-behavior pruning for oscillatory/2-D systems (a
principled sibling of the energy filter). Prioritize the non-intersection
rule; treat QPORTRAIT as optional.

---

## Tier 3 — alternative model intake & a scaling lesson

### 7. Qualitative Process Theory (Forbus) as a model front-end — **implemented**

> Built as `qrlib.frontends.qpt`: quantities, qualitative
> proportionalities, and processes with direct influences; influence
> resolution emits derivative variables and per-activation-combination
> constraints; activation conditions compile to operating regions with
> boundary transitions (an engine Zeno guard suppresses instantaneous
> boundary ping-pong). Strengthenings inherent to targeting a QDE engine
> are documented in the module, and unsupported forms raise rather than
> degrade. The rest is the original assessment.

**What it does.** A **process-centered** modeling ontology — processes
(flow, transfer, motion) are the primitives that create influences — versus
our constraint/QDE ontology. Widely compiled *down* to QDEs, so it can feed
the existing engine.
*Forbus, "Qualitative Process Theory," Artificial Intelligence 24 (1984).*

**Subsumed by us?** The *engine* subsumes compiled QPT; the *authoring
ontology* is new. Value is ergonomic — some domains are far more natural to
state as processes than as constraints.

**Incorporation.** Medium, and cleanly layered: a QPT front-end that emits
our `Model`/regions, with influence resolution producing per-region sign
structure (which `bridge.signs` already consumes). No engine change.
**Buys:** a more natural authoring path and automatic process-activation
regions.

### 8. ENVISION / confluences (de Kleer & Brown) — **composition implemented**

> Built as `qrlib.frontends.devices`: a `Library` of `ComponentType`s
> (terminals + internals + a law over local names), instantiated and
> wired by a `Device`; connected terminals unify into shared variables
> and the composed result is an ordinary `Model` ("no function in
> structure": types know nothing about their wiring). Component modes
> defer to operating regions / `diagnosis.Component`; **total
> envisionment is now `qrlib.envision`** (all consistent states of a
> region's constraint set, connected — the full qualitative phase
> portrait). The rest is the original assessment.

**What it does.** A **device-centered** ontology: components + conduits
(topology) + boundary conditions, behavior composed from a reusable
generic-component library ("no function in structure"). Confluences are
QDEs over a three-valued sign space; multiplication is exact but addition is
inherently ambiguous — the structural root of qualitative ambiguity — and
*total* envisionment enumerates all states.
*de Kleer & Brown, "A Qualitative Physics Based on Confluences," AI 24
(1984).*

**Subsumed by us?** Confluences ≈ our sign-matrix intake; what's new is
**component-topology composition** (build a model by wiring library parts)
and **total** (vs. attainable) envisionment.

**Incorporation.** Medium. A component library + netlist compiler onto
`Model`; a total-envisionment mode (already on the backlog). **Buys:**
compositional model construction from reusable parts — valuable if models
are built from a fixed catalog of devices.

### 9. Polynomial-time self-explanatory compilation — SIMGEN Mk3

**What it does.** Compiles self-explanatory simulators from compositional
domain theories in *polynomial* (empirically quadratic) time, scaling to
thousands of parameters, by **minimizing** qualitative reasoning: no
transitivity closure, no influence resolution, no limit analysis — because
full envisionment is exponential and unnecessary *when quantitative
information is available*.
*Forbus & Falkenhainer; Forbus, "Polynomial-time compilation of
self-explanatory simulators," QR-94 / IJCAI-95.*

**Subsumed by us?** It's a strategic lesson more than a module: when numbers
exist, skip the expensive qualitative steps. Dovetails with our Q2 layer.

**Incorporation.** N/A as a drop-in; adopt the *principle* in the
explanation/semiquant path — attach explanations to a numerically-driven run
without paying for envisionment. **Buys:** a scaling story for the
self-explanatory/narration path on large, numerically-instantiated models.

---

## Incremental (not a new class)

- ~~**Dynamic chatter abstraction** (Clancy & Kuipers, AAAI-97)~~ —
  **implemented** (`engines.chatter` + `SimConfig.dynamic_chatter`).
  Structural per-region analysis finds direction-unanchored constraint
  classes (rigid M+/M-/MINUS links; ADD/MULT rigid only via Constant
  operands; anchors = state variables and constants); during expansion,
  successors identical except in candidates' directions merge with the
  wiggling directions projected to `Qdir.IGN`. Candidacy is permission,
  not sentence: a candidate whose direction filtering pins stays
  concrete (the U-tube's flow class is a candidate yet its graph is
  untouched — precision static abstraction loses). `track_qdir`
  force-tracks; guide dir-atoms and DecSIM guided interface variables
  are force-tracked automatically.

## Not yet assessed (honest gaps — candidates for a follow-up pass)

These were named in the search but did **not** produce verified findings, so
their fit is genuinely open, not judged:

- **Monotone dynamical-systems theory** (Hirsch; Angeli & Sontag) as a
  *rigorous foundation* for M+/M- reasoning — potentially the strongest
  soundness upgrade, since it characterizes exactly when monotonicity
  constraints have provable consequences. Worth a targeted read.
- **Symbolic abstraction / reachability tooling** in formal methods (finite
  abstractions of continuous systems; validated reachability). Overlaps our
  coverage-oracle and region machinery; may offer more rigorous abstraction
  guarantees.
- **Temporal-logic falsification / conformance testing** of hybrid systems.
  Adjacent to the coverage oracle (search for a violating trajectory vs.
  check a given one).
- **Semi-quantitative variants** (Q3, NSIM, SQSIM) and **comparative
  analysis / exaggeration** (Weld) — named but unverified here.
  **QDE induction from data** (the GENMODEL/MISQ/QSI/ILP lineage) is now
  built as `qrlib.induce` (structure selection over a parsimony ladder,
  validated by the data-consistency checker).

## Recommended sequence

By value-per-effort against the current architecture:

1. ~~**Causal ordering** (§4)~~ — **done** (`qrlib.analysis.causal`).
2. ~~**Model-based diagnosis** (§2)~~ — **done** (`qrlib.diagnosis`).
3. ~~**Guided simulation / exogenous inputs** (§3)~~ — **done**
   (`qrlib.guide`).
4. ~~**DecSIM decomposition** (§1)~~ — **done** (`qrlib.decompose`).
5. Opportunistic: ~~the **non-intersection global filter** (§6)~~ — **done**
   (`engines.phase`, `SimConfig.phase_pairs`); ~~**FOG** disambiguation
   (§5)~~ — **done** (`qrlib.Negligible`); ~~a **QPT/ENVISION** front-end
   (§7–8)~~ — **done** (`qrlib.frontends`).

See also `docs/piecewise-affine.md` — a related near-term candidate
(exact qualitative phase portraits for piecewise-affine systems) documented
separately.
