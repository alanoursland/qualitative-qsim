# Bridge to numeric dynamical systems

The library is oriented toward eventual integration with numeric
dynamical-systems tooling. This note pins down the *interfaces* that make
that possible without referencing (or depending on) any particular stack.
Everything here consumes/produces plain tensors and `qrlib` objects.

## The two directions

```
                    abstraction (up)
   numeric world  ───────────────────▶  qualitative world
   x: (B, T, V) trajectories            QState sequences, BehaviorGraph
   f: vector fields / flows   ◀───────  Model (QDE): landmarks, M+/M-,
                    specification (down)  sign structure, corresponding values
```

### Upward: trajectory abstraction (first priority)

Input: numeric trajectories `x ∈ R^(B,T,V)` (+ optional times `(B,T)`),
plus a `Model` (or just per-variable quantity spaces with numeric landmark
values). Output: for each trajectory, its qualitative behavior — the
alternating point/interval sequence of `QState`s — plus the mapping back to
time indices.

Pipeline (each stage batched, GPU-friendly, in `qrlib.bridge.abstraction`):

1. **Quantize magnitudes:** bucketize `x` against per-variable landmark
   values → magnitude ranks `(B,T,V)` (searchsorted; tolerance band around
   landmarks to absorb numeric noise).
2. **Estimate directions:** sign of a smoothed derivative (finite
   differences with hysteresis thresholds) → dir codes `(B,T,V)`.
3. **Segment:** collapse runs of identical `(mag, dir)` vectors; insert
   point-states at change indices → qualitative behaviors of wildly varying
   length (ragged; returned as index tensors + a small Python view layer).
4. **Canonicalize:** merge segments shorter than a debounce threshold
   (numeric chatter), producing clean QSIM-style behaviors.

Uses this unlocks immediately:

- **Soundness testing of our own engines** (abstract an ODE instance's
  trajectory; assert it's a path in the QSIM-predicted graph).
- **Behavior mining:** cluster/summarize large trajectory batches by
  qualitative behavior — a discrete fingerprint of a phase portrait.
- **Landmark discovery from data:** locations where trajectories
  consistently become steady propose landmarks.
- **Model validation:** flag trajectories of a learned/fitted numeric model
  that fall *outside* a trusted qualitative model's behavior graph.

### Downward: qualitative model as specification (second priority)

A `Model` denotes a *set* of numeric systems. Planned artifacts:

- **Consistency checker** (`qrlib.bridge.consistency`): given sampled
  numeric data `(x, dx/dt)`, check the sign/monotonicity/corresponding-value
  constraints directly (does `outflow` actually increase with `level` in the
  data?). Reports per-constraint violation masses, batched.
- **Envelope sampling:** generate concrete instances of a qualitative model
  (sample monotonic functions consistent with `M+` + corresponding values,
  e.g. monotone splines) to produce numeric ODEs for testing or Monte Carlo
  over model uncertainty. Also the natural randomized-testing rig for QSIM
  soundness.
- **(Later) soft constraint losses:** differentiable relaxations of
  constraint violations, usable as regularizers when fitting numeric models
  so the fitted system respects known qualitative structure. Deliberately
  deferred — see open questions.

## Interface contracts (proposed, minimal)

To stay integration-agnostic, the bridge asks the numeric side for at most:

1. **Trajectories:** `(B, T, V)` float tensor (+ optional `(B, T)` times,
   irregular sampling OK), with a name/index mapping for `V`.
2. **Optionally, derivatives or a vector field callable** `f(x) -> dx` for
   consistency checking (finite differences are the fallback).

Nothing else. No stepping, no solver control, no callbacks. Anything that
can produce a trajectory tensor can integrate.

Conversely the qualitative side exposes stable, serializable artifacts:
`Model` (JSON-able schema), `BehaviorGraph` (nodes = encoded states, edges,
terminal labels), and the abstraction results (segment index tensors +
`QState`s). A future integration layer can be written entirely against these
without touching engine internals.

## Semantics gotchas to respect

- **QSIM time is event-based, numeric time is sampled.** Abstraction must
  tolerate landmark crossings between samples (detect sign changes of
  `x - landmark`, don't require exact hits) — hence the point-state
  *insertion* in stage 3 rather than classification of samples.
- **Continuity assumptions:** QSIM's guarantees assume C¹ "reasonable"
  functions; numeric trajectories with discontinuous inputs or hybrid jumps
  need the operating-region machinery (mode-tagged segments) — design the
  segment representation with an optional mode channel from day one.
- **Noise vs. chatter:** measured data will chatter in `qdir` near steady
  states; the debounce/hysteresis parameters are semantically meaningful
  (they define "steady") and must be explicit, not hidden defaults.
