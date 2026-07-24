# The qualitative reasoning landscape

A short survey of the main QR formalisms, what each contributes, and how this
library currently represents them. This is orientation material, not a
literature review. Stable bibliographic records and implementation-lineage
notes are in [`references.md`](references.md).

## 1. QSIM — qualitative simulation

Sources: [Kuipers1986](references.md#kuipers1986) and
[Kuipers1994](references.md#kuipers1994).

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

**In qrlib:** implemented by the reference engine and behavior-equivalent
tensor filtering. Corresponding values, new-landmark introduction, analytic
and phase filters, and chatter mitigation are supported.

## 2. Envisionment / confluences

Source: [deKleerBrown1984](references.md#dekleerbrown1984).

Qualitative physics based on **confluences** (qualitative differential
equations over signs), with the **envisionment**: the graph of *all* possible
qualitative states and transitions of a device, not just those reachable from
one initial state. Device-centric: models are compositions of component
models with local behavior rules.

**In qrlib:** attainable envisionment is a QSIM configuration; total
envisionment is `qrlib.envision`; reusable component/device composition is
`qrlib.frontends.devices`.

## 3. Qualitative Process Theory

Source: [Forbus1984](references.md#forbus1984).

Physics organized around **processes** (heat flow, liquid flow, boiling) that
are active when their preconditions hold and impose **influences** (direct
`I+`/`I−` on derivatives, indirect `∝Q+`/`∝Q−` on magnitudes). Influence
resolution combines all active processes to determine derivative signs. QPT
shines at *model formulation* — deciding which equations apply when — where
QSIM assumes the QDE is already given.

**In qrlib:** `qrlib.frontends.qpt` compiles process-centered descriptions,
activation conditions, and influence resolution into ordinary QDE constraints
and operating regions.

## 4. Semi-quantitative reasoning

Sources: [KuipersBerleant1988](references.md#kuipersberleant1988) for Q2 and
[BerleantKuipers1997](references.md#berleantkuipers1997) for Q3.

Annotate landmarks with **numeric interval bounds** and monotonic functions
with **envelopes**; propagate intervals along the qualitative behaviors to
(a) prune behaviors whose numbers can't work out and (b) produce guaranteed
numeric bounds on trajectories. This is the classic path from "pure symbols"
toward numbers, and the most direct synergy with numeric dynamical systems.

**In qrlib:** `qrlib.semiquant` performs behavior-level interval and time
refinement; `qrlib.tensor.interval` provides batched interval narrowing and
feasibility screening.

## 5. Model learning / QDE induction (GENMODEL, MISQ, QSI lineage)

Sources:
[RichardsKraanKuipers1992](references.md#richardskraankuipers1992) and
[RamachandranMooneyKuipers1994](references.md#ramachandranmooneykuipers1994).

Inducing qualitative models from (numeric or qualitative) trajectory data:
propose constraints consistent with observed qualitative behaviors, search
model space. Attractive here because the upward bridge (trajectory
abstraction) produces exactly the input these methods need.

**In qrlib:** `qrlib.induce` ranks candidate QDE structures using sign
estimation, parsimony, and the data-consistency checker. It is part of this
lineage, not a claim to reproduce every historical induction system.

## 6. Other relatives

- **Comparative analysis:** how does behavior change if a parameter increases?
  See [ChiuKuipers1992](references.md#chiukuipers1992);
  `qrlib.analysis.compare` implements equilibrium comparative statics.
- **Temporal-logic queries over behavior graphs** (model checking behaviors
  against trajectory specifications): `qrlib.guide` provides standalone
  classification and interleaved guided simulation.
- **Order-of-magnitude reasoning** (O(M), ROM): different algebra, same
  "constraint tables over small enums" implementation shape;
  `qrlib.Negligible` implements FOG's `Ne` relation in a sound instantaneous
  form.
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
