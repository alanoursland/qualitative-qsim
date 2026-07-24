# QSIM deep-dive and implementation plan

Working notes on the algorithm we are committing to implement first.
Primary sources:
[Kuipers1986](references.md#kuipers1986) and
[Kuipers1994](references.md#kuipers1994) — the 1994 book is the authoritative
spec, including
the transition tables.

## 1. Semantics in one paragraph

A QDE constrains a set of "reasonable" real-valued functions of time. A
solution of the QDE is an assignment of reasonable functions to variables
satisfying all constraints. QSIM computes a description of *all* solutions:
time is broken at **distinguished time points** (where some variable reaches
a landmark or becomes steady); between consecutive distinguished time points
every variable has a constant qualitative value. A **behavior** is the
resulting alternating sequence `state(t0), state(t0,t1), state(t1), ...` of
point-states and interval-states. QSIM's output is the tree of all such
sequences from the initial state.

## 2. The successor generation loop

```
frontier = {initial_state}
while frontier and under limits:
    s = pop(frontier)
    1. Per-variable candidates: for each variable, look up possible next
       (qmag, qdir) values in the P-transition table (if s is a point-state)
       or I-transition table (if s is an interval-state).
    2. Constraint filtering ("tuple filter"): for each constraint, keep only
       those tuples of candidate values for its variables that satisfy the
       constraint's sign/ordering rules and its corresponding values.
    3. Pairwise (Waltz) filtering: constraints sharing a variable must agree
       on its surviving candidates; propagate deletions to fixpoint.
    4. Global interpretations: assemble complete next-states from surviving
       per-constraint tuples (backtracking cross-product with consistency).
    5. Global filters (each optional, on by default):
       - no-change filter: drop successors identical to s (time must advance);
       - quiescence: all qdirs STD -> mark as quiescent leaf (equilibrium);
       - cycle match: successor identical to an ancestor -> close a cycle;
       - divergence: a variable at ±inf at a point-state -> terminal;
       - new-landmark introduction (see §4).
    6. Add surviving successors as children of s; push non-terminal ones.
```

Both transition tables are tiny (per-variable, ≤ 4 rows apply per current
value) and come straight from the intermediate value theorem + mean value
theorem applied to reasonable functions. They are data, not code: we encode
them as tables in `engines/transitions.py` with the book's row labels
(P1..P7, I1..I9 style) preserved in comments for auditability.

Key structural fact for implementation: **point→interval transitions (P) are
nearly deterministic; interval→point transitions (I) branch.** Branching is
where behavior trees fan out and where filtering earns its keep.

## 3. Constraint filtering details

Per-constraint consistency rules:

- `ADD(x,y,z)` / `MULT(x,y,z)` / `MINUS(x,y)`: sign algebra on qmag (relative
  to 0) and on qdir, plus corresponding-value orderings (e.g. if (x*,y*,z*)
  is a corresponding triple and x > x*, y = y* then z > z*).
- `DERIV(x,y)`: sign(y's qmag) must equal x's qdir.
- `M+(x,y)`: qdirs equal; qmags consistent with every corresponding pair
  (x above/at/below x* iff y above/at/below y*). `M-` dual.
- `CONSTANT(x)`: qdir = STD, qmag fixed.

Implementation note: all of these are functions of (a) sign of qmag, (b)
order relation of qmag vs. each corresponding landmark, (c) qdir. With the
rank encoding of magnitudes, each reduces to integer comparisons — which is
what makes the dense-table compilation (`architecture.md`) and later
tensorization mechanical rather than clever.

## 4. New landmarks

When a variable becomes steady (qdir hits STD) at a magnitude that is *not* a
landmark, QSIM introduces a **new landmark** there (e.g. the level at which
the tank equilibrates), splitting the interval in its quantity space for all
subsequent states of that behavior branch. Consequences we must design for:

- Quantity spaces are **per-behavior-branch**, not global; the rank encoding
  of a variable can grow along a path. The compiled landmark→rank map is
  therefore versioned; a branch carries its space version.
- Corresponding values gain new entries (the new landmark corresponds via
  active constraints, e.g. `M+` partners equilibrate together).
- This is the main source of divergence between "toy QSIM" and real QSIM;
  the reference engine implements it before the tensor engine does
  (tensorized landmark introduction likely means re-encoding the affected
  branch's frontier — acceptable because introductions are rare events).

## 5. Chatter

Weakly-constrained variables (typically higher derivatives) can oscillate
their qdir spuriously, exploding the tree with distinctions that carry no
information. Standard mitigations, in the order we'll implement them:

1. **ignore-qdir** per variable (user annotation): drop direction distinctions
   for named variables.
2. **chatter-box abstraction**: detect the chattering subspace and collapse it
   into a single abstract state with `qdir = unknown` for chattering
   variables.

Chatter handling is a *filter module*, not core semantics — toggleable, and
off means textbook behavior.

## 6. Limits and termination

QSIM need not terminate (new landmarks can be introduced forever). The engine
takes explicit resource limits: max states, max depth, and max landmarks per
variable. `max_states` is a strict graph-node bound including the root.
Hitting a limit marks affected leaves as
`TRUNCATED` in the behavior graph — never silently dropped, per the soundness
commitment.

## 7. Acceptance tests (from the literature)

| Model | Expected qualitative outcomes |
|---|---|
| Bathtub (constant inflow) | rise to equilibrium (decelerating); or reach FULL/overflow depending on landmarks — matches Kuipers' worked example |
| U-tube | levels converge monotonically to equal-pressure equilibrium |
| Frictionless spring | sustained oscillation (cycle detected) |
| Spring + friction (monotonic damping) | includes decaying oscillation; known spurious behaviors (e.g. increasing oscillation) appear **iff** the corresponding filters are off — a good filter regression test |
| Two cascaded tanks | composition sanity; larger tree, checks scaling knobs |

Each becomes a golden test with the expected behavior set checked
structurally (tree shape + terminal classifications), not by string
comparison.

## 8. Phase-1 implementation notes (as built)

Semantic decisions made in the reference engine, recorded for review:

- **Infinity admissibility.** A point state with a variable at `±inf`
  denotes the limit t→∞; it is admitted only if every variable is steady or
  itself at an infinite landmark (a finite limit or an infinite one — no
  variable can be left mid-motion at t=∞). `ADD` additionally applies the
  algebra of infinities (one infinite operand forces the sum; a finite sum
  of one infinity needs the opposite infinity). Together these kill the
  classic reach-infinity-in-finite-time spurious behaviors: the spring's 16
  divergent candidates and the U-tube's "tank fills to infinity" branch.
  Toggling `infinity_filter` off restores them (regression-tested).
- **Steady inside an interval without landmark discovery.** I5/I9
  transitions (becoming steady at an unnamed value) are admitted and yield
  quiescent states whose magnitude is the open interval — the equilibrium
  exists but is unnamed until phase-2 landmark introduction mints it.
- **Quiescent states are terminal and explorable.** The constant
  continuation is one complete behavior; departure candidates (unstable
  equilibria) are still generated and survive only if constraint-consistent.
- **Domain exit.** A point state where a variable sits at a boundary
  landmark of its *bounded* space with an outward direction has no legal
  P-transition; it is classified `DOMAIN_EXIT` (leaving the model's domain
  of validity).
- **Dead ends are reported, not pruned.** A state with candidates but no
  consistent successor is classified `DEADEND`; deleting it (or its
  ancestors) retroactively is a later refinement that must preserve
  soundness bookkeeping.

Phase-2 additions:

- **Frames.** Landmark discovery makes quantity spaces per-branch: each
  node carries a frame (a `CompiledModel` with grown spaces and
  rank-shifted constraint references). Inserting a landmark into interval
  rank `r` shifts that variable's ranks above `r` by +2; the steady value
  re-encodes to `r+1`. Frames are content-hashable, so cycle matching and
  envisionment merging compare `(frame, state)` pairs — states in
  different spaces never spuriously match.
- **Discovered corresponding values.** When minting, any M+/M-/MINUS/ADD
  constraint touching a minted variable whose variables are all at
  landmarks records the observed rank tuple as a corresponding value
  (valid: monotonic functions and sums pin co-occurring values). This is
  what ties `x*0` to `a*0` in the spring and sharpens later filtering on
  that branch.
- **Chatter abstraction is projection, not restriction.** For
  `ignore_qdir` variables, candidates are generated over all concrete
  directions and constraint-filtered normally (soundness), then projected
  to `IGN` and merged (collapse). Ignored variables never mint landmarks
  and never block quiescence.
- **The energy-argument slot.** `successor_filters` receive
  `(parent_state, candidate, frame)` with the candidate in the parent's
  (pre-mint) frame. `EnergyFilter` supplies conserved and nonincreasing
  amplitude policies through discovered extrema. `LyapunovCertificate`
  supplies conditional strict decrease for an explicit scalar variable:
  it enforces the scalar's minimum at a declared equilibrium, checks its
  direction locally, and rejects a recurrent path if strict descent occurred
  somewhere around the proposed cycle. This last check matters when the
  numeric scalar falls without leaving one open qualitative interval.
  Lyapunov recurrence checking is path-dependent and therefore cannot be
  combined with `envisionment=True`.

Phase-4 additions — operating regions:

- **Guards fire at points, on magnitudes.** A region transition is a
  conjunction of landmark predicates (`(var, op, landmark)`) evaluated on
  a point state's magnitude ranks in the node's frame (guards reference
  landmark *names*, so discovered landmarks shifting ranks cannot break
  them). Direction is expressed by adding atoms (e.g. `netflow > 0`
  distinguishes overflow arrival from equilibrium arrival at the same
  landmark).
- **Region entry is instantaneous and re-derives directions.** The
  transition state (source region) gets entry children (target region)
  with carried magnitudes, any explicitly reset variables assigned to target
  landmarks, and directions enumerated afresh under the target's constraint
  subset — the vector field may change
  discontinuously at the boundary, so carrying directions over would be
  unsound. This produces a point→point edge; alternation resumes
  immediately after. When a transition fires, normal in-region expansion
  is suppressed (the guard marks the boundary of the source region's
  validity); a boundary with no declared transition remains
  `REGION_EXIT`.
- **Identity is (frame, state, region).** Cycle matching and envisionment
  merging compare all three.

## 9. Order of implementation

1. Transition tables + per-variable candidate generation (pure data + lookup).
2. Constraint predicates + corresponding values (reference form).
3. Tuple/Waltz/global filtering, no new landmarks — bathtub & U-tube green.
4. Global filters: quiescence, cycle, no-change, divergence — spring green.
5. New-landmark introduction — equilibrium landmarks appear correctly.
6. Chatter mitigations, limits/truncation polish.
7. Then and only then: tensorized re-implementation (`gpu-tensorization.md`).
