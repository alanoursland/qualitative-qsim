# Open questions

Things that should be settled by the forthcoming integration details (or an
explicit decision), roughly in order of how much they shape the design.
Provisional defaults are what the skeleton currently assumes.

## 1. Which bridge direction matters most?

The numeric bridge has four distinct products: (a) trajectory abstraction
(numeric → qualitative summaries), (b) validation (numeric model vs.
qualitative spec), (c) guidance (use the behavior graph to steer numeric
search/experiments), (d) model induction (learn QDEs from trajectories).
**Provisional default:** (a) first, since it unlocks the soundness harness
regardless. If the integration mainly needs (b)/(c)/(d), Phase 3 reshapes.

## 2. Scale profile

Typical variable counts, batch sizes of trajectories, and whether *ensembles
of models* is a real workload decide how hard to push the tensor engine and
whether `int8` encodings / `torch.compile` are worth it.
**Provisional default:** design for `V ≤ ~64`, trajectory batches up to ~10⁶
timesteps total, ensembles as a first-class batch axis.

## 3. Time semantics of the numeric side

Continuous-time ODE trajectories, discrete-time maps, or both? Irregular
sampling? Hybrid/mode-switching systems from day one? This shapes the
abstraction pipeline's direction-estimation and the urgency of operating
regions. **Provisional default:** continuous-time, possibly irregularly
sampled, hybrid support designed-for but deferred (mode channel reserved in
the segment representation).

## 4. Differentiability

Is gradient flow through qualitative structure (soft constraint losses,
relaxations) ever wanted, or is QR strictly a symbolic/analysis layer?
**Provisional default:** strictly symbolic core; soft losses deferred to
Phase 6 and layered, never entangled with exact semantics.

## 5. Naming & packaging conventions

Package is currently `qrlib` under `src/`, `pyproject.toml`/hatchling,
Python ≥ 3.10, torch required-but-lazily-imported. If the neighboring
ecosystem has conventions (namespace packages, config style, tensor layout
`(B,T,V)` vs `(T,B,V)`, device handling idioms), we should match them early —
renames get expensive fast. **Provisional default:** batch-first `(B,T,V)`.

## 6. Model interchange format

Should qualitative models be authored/emitted by other tools? If yes, the
JSON schema for `Model` (variables, spaces, constraints, corresponding
values) should be versioned and specified early. **Provisional default:**
schema exists but is marked unstable until Phase 2.

## 7. License and distribution

No license file yet — needs an explicit choice (MIT/BSD-3/Apache-2.0?)
before anything is published. Also: PyPI name (`qualitative-reasoning-lib`?
`qrlib` is likely taken — check before Phase 1 ends).

## 8. Scope of "other stuff too"

The landscape doc lists candidates (envisionment, QPT-style processes,
semi-quantitative Q2, comparative analysis, QDE induction). Current bet on
order: Q2 > operating regions > envisionment > induction > comparative
analysis. Cheap to reorder now, worth confirming against the integration
goals when details arrive.
