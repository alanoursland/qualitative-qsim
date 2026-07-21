# The qualitative reasoning landscape

A short survey of the main QR formalisms, what each contributes, and what this
library plans to do about them. This is orientation material, not a
literature review.

## 1. QSIM — qualitative simulation (Kuipers, 1986; 1994)

**The centerpiece of this library.**

- **Model form:** *Qualitative Differential Equations* (QDEs). Variables are
  "reasonable functions" of time (continuously differentiable, finitely many
  critical points). Each variable has a **quantity space**: a totally ordered
  set of landmark values, typically including `0` and possibly `±∞`. A
  variable's qualitative value is a pair *(qmag, qdir)*: magnitude (at a
  landmark, or in an open interval between adjacent landmarks) and direction
  of change (`dec`/`std`/`inc`).
- **Constraints:** `ADD(x,y,z)` (x+y=z), `MULT(x,y,z)`, `MINUS(x,y)`,
  `DERIV(x,y)` (dx/dt = y), `M+(x,y)` / `M-(x,y)` (y = f(x) for some monotonic
  f), `CONSTANT(x)`. Constraints may carry **corresponding values** — tuples
  of landmarks known to co-occur (e.g., f(0)=0) — which sharpen filtering.
- **Algorithm:** from a state, enumerate each variable's possible next values
  from small **transition tables** (P-transitions leaving a time point,
  I-transitions leaving an interval), then filter combinations: per-constraint
  tuple filtering, Waltz-style pairwise propagation, global cross-product
  consistency, then global filters (no-change, quiescence, cycle
  identification, divergence at infinity). Time alternates between instants
  and open intervals. Output is a **behavior tree/graph**.
- **Guarantee:** *sound* (every actual behavior of every instance ODE is
  predicted) but *incomplete* (spurious behaviors occur). Landmark discovery
  (introducing new landmarks where a variable becomes steady) and the
  **chatter** problem (spurious branching on directions of weakly-constrained
  variables) are the classic practical issues; chatter-box abstraction and
  ignore-qdir treatments are the standard mitigations.

**Plan:** full implementation — reference pure-Python engine first, then a
tensorized engine. Corresponding values, new-landmark introduction, and
chatter mitigation are all in scope (phased; see `roadmap.md`).

## 2. Envisionment / confluences (de Kleer & Brown, 1984)

Qualitative physics based on **confluences** (qualitative differential
equations over signs), with the **envisionment**: the graph of *all* possible
qualitative states and transitions of a device, not just those reachable from
one initial state. Device-centric: models are compositions of component
models with local behavior rules.

**Plan:** an *attainable envisionment* (reachable-graph) mode falls out of
QSIM almost for free (memoize states, share successors). A *total
envisionment* mode (enumerate all consistent states, then connect) is a
natural batch/GPU workload — generating and filtering the full state
cross-product is exactly a tensor job. Component/device composition is
lower priority than the state-graph machinery.

## 3. Qualitative Process Theory (Forbus, 1984)

Physics organized around **processes** (heat flow, liquid flow, boiling) that
are active when their preconditions hold and impose **influences** (direct
`I+`/`I−` on derivatives, indirect `∝Q+`/`∝Q−` on magnitudes). Influence
resolution combines all active processes to determine derivative signs. QPT
shines at *model formulation* — deciding which equations apply when — where
QSIM assumes the QDE is already given.

**Plan:** medium-term. The clean landing spot in this architecture is
**operating regions / mode transitions**: a piecewise QDE whose regions have
guards, which QSIM already needs for models that change structure (tank
overflows, valve opens). Full QPT process libraries and influence resolution
can build on that later without changing the core.

## 4. Semi-quantitative reasoning (Q2/Q3, NSIM; Kuipers & Berleant et al.)

Annotate landmarks with **numeric interval bounds** and monotonic functions
with **envelopes**; propagate intervals along the qualitative behaviors to
(a) prune behaviors whose numbers can't work out and (b) produce guaranteed
numeric bounds on trajectories. This is the classic path from "pure symbols"
toward numbers, and the most direct synergy with numeric dynamical systems.

**Plan:** in scope after the core engine works. Interval propagation is also
naturally tensorizable (interval arithmetic on stacked `(lo, hi)` tensors).

## 5. Model learning / QDE induction (GENMODEL, MISQ, QSI lineage)

Inducing qualitative models from (numeric or qualitative) trajectory data:
propose constraints consistent with observed qualitative behaviors, search
model space. Attractive here because the upward bridge (trajectory
abstraction) produces exactly the input these methods need.

**Plan:** stretch goal; the abstraction pipeline (`numeric-bridge.md`) is a
prerequisite and is planned regardless.

## 6. Other relatives (tracked, not planned)

- **Comparative analysis** (Weld): how does behavior change if a parameter
  increases? Pairs well with ensembles-of-models batching.
- **Temporal-logic queries over behavior graphs** (model checking behaviors
  against CTL-ish specs): behavior graphs are Kripke structures; a small
  query layer may appear once graphs exist.
- **Order-of-magnitude reasoning** (O(M), ROM): different algebra, same
  "constraint tables over small enums" implementation shape.
- **Qualitative spatial reasoning:** out of scope.

## Implementation-relevant common structure

Every formalism above reduces, computationally, to:

1. small finite value domains per variable (signs, landmarks×directions),
2. local consistency tables (per constraint / transition / influence),
3. combinatorial search over labelings, pruned by (2),
4. graphs over the surviving labelings.

That shared shape is what the core package layout (`architecture.md`) and the
tensor encoding (`gpu-tensorization.md`) are built around — QSIM is the first
client, not a special case.
