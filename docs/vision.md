# Vision

## What this library is

A modern, tensor-friendly implementation of qualitative reasoning (QR) about
continuous dynamical systems. "Qualitative" means the system is described by:

- **orderings and landmarks** instead of real values (`level` is between `0`
  and `FULL`, currently rising),
- **monotonic and algebraic relationships** instead of exact functions
  (`outflow = f(level)` for *some* increasing `f`), and
- **derivative/sign structure** instead of parameterized ODEs.

From such a description the library derives the set of all qualitatively
distinct behaviors the system can exhibit, as a branching graph of qualitative
states. Kuipers' QSIM is the reference algorithm and the first engine.

## What this library wants to become

The long-term orientation is **QR as an abstraction layer over numeric
dynamical systems**. Concretely, the library should eventually support a
two-way bridge:

1. **Downward (specification):** a qualitative model constrains a space of
   numeric systems. Any concrete ODE whose right-hand side satisfies the
   monotonicity/sign structure is an *instance* of the qualitative model, and
   QSIM's soundness guarantee says its trajectories appear (abstracted)
   somewhere in the qualitative behavior graph.
2. **Upward (abstraction):** numeric trajectories — simulated, measured, or
   sampled in large batches — can be quantized against quantity spaces and
   segmented into qualitative behaviors, then compared with or used to prune
   the qualitative behavior graph, discover landmarks, or induce qualitative
   models from data.

We do not name or depend on any particular numeric simulation stack here. The
design just leans that way: model descriptions are declarative data, states
have canonical tensor encodings, trajectory-shaped tensors `(batch, time,
variable)` are an expected input, and everything runs batched.

## Why PyTorch

Three reasons, in decreasing order of importance:

1. **Batched combinatorics.** The inner loops of qualitative simulation are
   large cross-products filtered by small lookup tables (transition tables,
   constraint consistency tables). That is exactly the shape of workload that
   vectorizes well: encode candidates as integer tensors, evaluate constraints
   as gathers/masks, reduce. GPU acceleration matters when frontiers, model
   ensembles, or trajectory batches get large; on small textbook models plain
   Python is fine and remains the reference implementation.
2. **Adjacency to numeric work.** Trajectory abstraction consumes tensors that
   numeric simulators and datasets already produce. Staying in torch avoids
   copies and keeps device placement trivial.
3. **Optionality for learning.** If qualitative structure is ever used inside
   a training loop (e.g., soft/relaxed constraint penalties, or learning
   monotonic function envelopes), being in torch keeps that door open. This is
   explicitly *not* a near-term goal; the core semantics are discrete and
   exact.

## Design principles

- **One model, many engines.** `Model` (variables, quantity spaces,
  constraints, corresponding values) is inert data with a stable schema.
  Engines — pure-Python QSIM, tensorized QSIM, envisioners, abstraction
  pipelines — consume it. No engine-specific state leaks into the model.
- **Reference implementation first, fast implementation second.** Every
  tensorized code path must agree with a slow, readable pure-Python
  implementation, property-tested against it. Soundness bugs in QR are silent
  (a missing behavior looks like a cleaner answer), so the oracle matters.
- **Soundness over precision.** Follow QSIM's contract: the predicted behavior
  set is a superset of the real behaviors of every instance system. Spurious
  behaviors are pruned by explicit, individually-toggleable filters, never by
  ad-hoc shortcuts.
- **States are values.** Qualitative states are immutable and hashable, with a
  canonical integer encoding. Behavior graphs are ordinary graphs over these
  values; nothing about an engine's internals is needed to inspect results.
- **Batch-shaped by default.** Public tensor interfaces take and return
  batched tensors with a documented layout, so scaling from 1 to 10⁶ states
  is a non-event.
- **Small dependency surface.** `torch` plus the standard library for the
  core; visualization and notebook conveniences stay optional extras.

## Non-goals (for now)

- Real-time or embedded use.
- Full Garp3-style graphical model-building environments.
- Fuzzy-quantity or probabilistic-quantity semantics (interesting, but only
  after the exact core is trustworthy).
- Committing to a specific external dynamical-systems API before its details
  arrive; we keep the bridge abstract (see `docs/numeric-bridge.md` and
  `docs/open-questions.md`).
