# Bridge to numeric dynamical systems

The library is oriented toward integration with numeric dynamical-systems
tooling. This note pins down the tensor-level *interfaces*; the host-facing
requirements and seam contracts they serve are in `docs/host-integration.md`.
Everything here consumes/produces plain tensors and `qrlib` objects — no
dependency on any particular simulation stack.

## The two directions

```
                    abstraction (up)
   numeric world  ───────────────────▶  qualitative world
   x: (B, T, V) trajectories            QState sequences, BehaviorGraph
   sign matrices, landmark      ◀───────  Model (QDE): landmarks, M+/M-,
   values, fitted models         specification (down)  sign structure, regions
```

## Upward: trajectory abstraction (`qrlib.bridge.abstraction`)

Input: numeric trajectories `x ∈ R^(B,T,V)` (+ optional times `(B,T)`,
+ optional integer mode channel `(B,T)` for hybrid executions), plus
quantity spaces whose landmarks carry numeric values. The reference seam
also accepts optional `CrossingEvent` records from an event-aware solver.
Each carries the refined time, crossed variable and landmark, and the
solver's complete state vector at that instant; requiring the complete state
avoids fabricating nonlinear values for the other variables. Output: per
trajectory, its qualitative behavior — the alternating point/interval
`QState` sequence (region-tagged when a mode channel is given) — plus the
mapping back to original sample indices and physical-time bounds.

Pipeline (each stage batched, GPU-friendly):

1. **Quantize magnitudes:** bucketize `x` against landmark values →
   magnitude ranks `(B,T,V)` (searchsorted; tolerance band around landmarks
   absorbs numeric noise).
2. **Estimate directions:** sign of a smoothed derivative (finite
   differences with hysteresis thresholds) → dir codes `(B,T,V)`.
3. **Segment:** collapse runs of identical `(mag, dir, mode)` vectors;
   insert point-states at change indices → ragged behaviors (index tensors
   + a small Python view layer).
4. **Canonicalize:** merge segments shorter than a debounce threshold
   (numeric chatter), producing clean QSIM-style behaviors.

Refined event states are inserted before these stages and protected from
debounce. Multiple landmark records may share one time and state for a
simultaneous crossing. This can recover ordered crossings inside one coarse
sampling interval; without those event states, a jump over multiple landmarks
still raises as undersampling.

The abstraction parameters (landmark tolerance, hysteresis, debounce) form
an explicit config value that travels with every downstream result — they
define what "steady" and "at the landmark" mean, so no claim is
interpretable without them.

## The coverage oracle (`qrlib.bridge.coverage`)

`check(trajectory_or_behavior, graph, *, abstraction_config) → CoverageResult`
— is the observed behavior a path in the predicted behavior graph? The
contract (witness path on success; longest matched prefix, first divergent
segment with source time range, and a variable/constraint-level diagnosis on
failure; embedded abstraction config) is specified in
`host-integration.md`, Surface 3. Batched form: `(B, T, V)` in →
per-trajectory coverage mask + aggregate statistics out; the aggregate
serves as a qualitative-consistency **score** for model selection.

Uses this unlocks:

- **Soundness testing of our own engines** (the phase-3 harness:
  instantiate → integrate → abstract → assert coverage).
- **Validation of learned/fitted models** against trusted qualitative
  structure, with witness-carrying results a host can archive.
- **Behavior mining:** cluster large trajectory batches by qualitative
  behavior — a discrete fingerprint of a phase portrait.

## Structure intake (`qrlib.bridge.signs`, `qrlib.bridge.harvest`)

- **Sign matrices → models:** `S[i][j] ∈ {-1, 0, +1, UNKNOWN}` (sign of
  ∂(dx_i/dt)/∂x_j, from a host's symbolic analysis or domain knowledge)
  compiles to a QDE: `Deriv` + `M±` constraints via named auxiliary
  variables; `UNKNOWN` stays unconstrained (sound). Region-dependent
  structure — a list of (region condition, matrix) — yields an
  operating-region model.
- **Sign estimation from data:** given `(x, dx/dt)` samples or trajectories,
  `estimate_signs_calibrated` fits the affine approximation and bootstraps
  complete sample rows with an explicit resample count and seed. Per-entry
  confidence is the fraction of bootstrap coefficient signs that agree with
  the full fit, bounded in `[0, 1]`. It measures stability under the observed
  sample distribution, not the probability that a physical dependency is
  true. Thresholding maps unstable effects and fitted zeros to `UNKNOWN`.
  The earlier unbounded t-like `estimate_signs` result remains for API
  compatibility.
- **Landmark intake:** `(variable, name, value?, bounds?)` records from any
  source (equilibrium finders, guard thresholds, domain knowledge) are
  deduplicated, ordered, and inserted; conflicts are reported, not guessed
  away. **Landmark proposal from data:** steady-point clustering over
  trajectory batches suggests landmarks (host accepts/rejects).

## Downward: model as specification

- **Consistency checker:** given sampled `(x, dx/dt)`, check
  sign/monotonicity/corresponding-value constraints directly (does
  `outflow` actually increase with `level` in the data?); reports
  per-constraint violation masses, batched.
- **Structure export:** `Model.sign_structure()` → the constraint closure
  (pairwise signs, derivative couplings, pinned zeros/orderings) as plain
  data, for hosts to map onto regression constraints
  (`host-integration.md`, Surface 6).
- **Envelope sampling:** generate concrete instances of a model (monotone
  splines consistent with `M±` + corresponding values) to produce numeric
  ODEs — the randomized-testing rig for soundness, and a Monte Carlo route
  over model uncertainty.
- **(Deferred) soft constraint losses:** differentiable relaxations for
  training-time use; layered strictly above the exact core, not scheduled.

## Interface contracts (minimal, host-agnostic)

The bridge asks the numeric side for at most:

1. **Trajectories:** `(B, T, V)` float (+ optional `(B, T)` times, irregular
   sampling OK; optional `(B, T)` int mode tags), with a name/index mapping
   for `V`.
2. **Optionally, derivative samples or a vector-field callable**
   `f(x) -> dx` (finite differences are the fallback).
3. **Optionally, structure summaries:** sign matrices, landmark records.

Nothing else — no stepping, no solver control, no callbacks, no symbolic
expressions. Conversely the qualitative side exposes stable, serializable
artifacts: `Model` (versioned JSON schema), `SimResult`/`BehaviorGraph`
(neutral array export), `CoverageResult`, `SignStructure`. A host adapter
layer is written entirely against these.

## As built (phase 3)

The reference implementations live in `qrlib.bridge.abstraction`,
`qrlib.bridge.coverage`, and `qrlib.bridge.harvest` — pure Python,
per-trajectory, accepting any array-like (torch/numpy via `tolist()`).
Contract details fixed during implementation:

- **Prefix semantics.** A finite sample window ends mid-behavior;
  asymptotic equilibrium arrival is the t→∞ limit and is never observed.
  Coverage therefore accepts observed behaviors as path *prefixes*;
  quiescent nodes absorb constant continuations; matching into a
  `TRUNCATED` frontier succeeds vacuously (noted in the result).
- **Frame translation.** Observations are quantized against the model's
  root spaces; graph nodes may sit at discovered landmarks. A node covers
  an observed value when its finer magnitude lies inside the observed
  coarser one (names map root→frame; frames only add landmarks).
- **Undersampling raises.** A magnitude jump across more than one landmark
  between samples is an error, not a silently-invented history. A host can
  resolve it by supplying the solver-observed `CrossingEvent` states.
- **Time provenance is explicit.** `AbstractedBehavior.spans` remains in
  original sample coordinates. `time_bounds` contains observed physical-time
  support; an inferred crossing is bracketed by unequal bounds, while a
  solver-refined event has an exact `(time, time)` bound.
- **Endpoint derivatives are second-order.** First-order one-sided
  differences are O(h)-biased exactly at critical points — precisely where
  initial conditions like "released at peak speed" sit.
- **Direction thresholds have a value-scale floor.** A relative-only
  threshold on an (essentially) constant variable is relative to rounding
  noise and hallucinates directions.

## Semantics gotchas to respect

- **QSIM time is event-based, numeric time is sampled.** Abstraction tolerates
  landmark crossings between samples (detect sign changes of `x − landmark`,
  don't require exact hits) — hence point-state *insertion* in stage 3.
  Hosts with event-accurate solvers can additionally pass complete refined
  crossing states; the seam does not require them.
- **Continuity assumptions:** QSIM's guarantees assume C¹ "reasonable"
  functions; discontinuous inputs or hybrid jumps need the operating-region
  machinery — which is why the mode channel exists from day one.
- **Noise vs. chatter:** measured data will chatter in `qdir` near steady
  states; the debounce/hysteresis parameters are semantically meaningful
  and must be explicit, never hidden defaults.
