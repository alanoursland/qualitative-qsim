# QSIM deep-dive and implementation plan

Working notes on the algorithm we are committing to implement first.
Primary sources: Kuipers 1986 (AIJ 29) and Kuipers 1994 (*Qualitative
Reasoning*, MIT Press) — the 1994 book is the authoritative spec, including
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
takes explicit resource limits: max states, max depth, max landmarks per
variable, wall-clock budget. Hitting a limit marks affected leaves as
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

## 8. Order of implementation

1. Transition tables + per-variable candidate generation (pure data + lookup).
2. Constraint predicates + corresponding values (reference form).
3. Tuple/Waltz/global filtering, no new landmarks — bathtub & U-tube green.
4. Global filters: quiescence, cycle, no-change, divergence — spring green.
5. New-landmark introduction — equilibrium landmarks appear correctly.
6. Chatter mitigations, limits/truncation polish.
7. Then and only then: tensorized re-implementation (`gpu-tensorization.md`).
