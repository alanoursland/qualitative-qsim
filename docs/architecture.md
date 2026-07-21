# Architecture

Provisional package layout and core abstractions. Everything here is a
proposal; the skeleton in `src/qrlib/` mirrors it so the proposal stays
honest. Host-facing requirements that shaped this design are collected in
`docs/host-integration.md`.

## Package layout

```
src/qrlib/
├── __init__.py          # public API re-exports
├── quantity.py          # Landmark, QuantitySpace, Qmag ranks, Qdir, QVal
├── state.py             # QState (one qualitative state of a model)
├── constraints.py       # Constraint types: MPlus, MMinus, Add, Mult, Deriv, ...
├── model.py             # Variable, Model, CompiledModel, regions, schema
├── behavior.py          # BehaviorGraph, TerminalClass, SimResult/SimConfig
├── semiquant.py         # Q2-style interval refinement, envelopes, time bounds
├── graph.py             # small graph algorithms: reachability, Tarjan SCC
├── engines/
│   ├── qsim.py          # reference pure-Python QSIM (+ envisionment mode)
│   ├── transitions.py   # P-/I-transition tables
│   ├── filters.py       # predicates, tuple/Waltz filtering, assembly
│   └── landmarks.py     # new-landmark introduction (per-branch frames)
├── tensor/
│   ├── encoding.py      # qcodes, frontier codecs, per-frame constraint tables
│   ├── engine.py        # tensorized prune/assemble, batched frontiers
│   └── abstraction.py   # batched (B,T,V) quantize/directions
├── bridge/
│   ├── abstraction.py   # trajectories (+ mode channel) -> behaviors
│   ├── coverage.py      # coverage oracle: witness / diagnosis / score
│   ├── signs.py         # sign-matrix intake, estimation, consistency check
│   └── harvest.py       # landmark intake/dedup + data-driven proposals
├── analysis/
│   ├── queries.py       # terminal census, quiescence, cycles, state search
│   └── explain.py       # structured + prose behavior narration              [planned]
└── viz/                 # data-first exports; optional plotting extra        [planned]
```

Rules of the layout:

- `quantity/state/constraints/model` are **pure data + trivial logic**, no
  torch imports, importable anywhere, exhaustively unit-tested. No CAS or
  graph-library dependencies anywhere in core (`host-integration.md` rules).
- `engines/` may only *read* models/states. The reference engine is written
  for clarity, not speed.
- `tensor/` is the only place that imports torch in anger (`bridge/` may, for
  batched paths). Tensorized code must be behavior-equivalent to `engines/`
  and is property-tested against it.
- `bridge/` depends on tensor layouts and the schemas in
  `host-integration.md`, never on any external simulator's API.
- `analysis/` and `graph.py` are dependency-free consumers of
  `BehaviorGraph`.

## Core abstractions

### Landmark and QuantitySpace

A `Landmark` is a **name** (canonical identity) plus optional numeric
knowledge: an exact `value` and/or interval bounds `(lo, hi)`. Purely
qualitative reasoning uses only names and order; abstraction, coverage, and
semi-quantitative propagation use the numbers when present
(`host-integration.md`, Surface 1).

A `QuantitySpace` is a totally ordered tuple of landmarks, optionally
open-ended below/above (conceptual `-inf`/`+inf` endpoints). Known landmark
values must respect the declared order (validated). Spaces are immutable
values; QSIM landmark discovery produces *new* spaces, versioned per behavior
branch (`docs/qsim.md` §4).

### Qualitative value: `QVal = (Qmag, Qdir)`

- `Qmag`: either *at* landmark `l_i` or *in* the open interval
  `(l_i, l_{i+1})`. Encoded as a single integer rank `2*i` / `2*i+1`;
  adjacency and ordering become integer arithmetic.
- `Qdir ∈ {DEC, STD, INC}`, encoded `0/1/2` (so `sign = qdir - 1`).

### QState

An immutable mapping `variable -> QVal` plus a time tag (`POINT` or
`INTERVAL` — QSIM alternates) and, once regions land, a region tag.
Hashable, comparable, with a canonical dense encoding `(2*V,)` int
(interleaved mag-rank/dir) given a frozen landmark→rank mapping; frontiers
stack to `(B, 2*V)`. State ids are content-derived hashes so external
references stay valid across runs.

### Constraints

Small frozen dataclasses referencing variables by name: `MPlus(x, y)`,
`MMinus(x, y)`, `Add(x, y, z)`, `Mult(x, y, z)`, `Minus(x, y)`,
`Deriv(x, y)`, `Constant(x)`, optionally carrying **corresponding values**
(tuples of landmark names that co-occur). Each constraint type answers one
question — *is this tuple of QVals consistent with me?* — in two
interchangeable forms: a readable Python predicate (reference) and a dense
boolean lookup table over encoded values (tensor engine), built at model
compilation. Constraints also contribute to the model's exported
`SignStructure` (`host-integration.md`, Surface 6).

### Model and Regions

`Model` = named `Variable`s (each owning a QuantitySpace) + constraints +
**regions**. A region declares its active constraint subset, boundary
conditions as landmark predicates over named landmarks, and a
region-transition map; a model with no declared regions has one implicit
region. `Model.compile()` freezes name→index and landmark→rank mappings and
precomputes constraint tables, yielding a `CompiledModel` that engines
consume. The un-compiled `Model` stays editable and serializable via a
**versioned JSON schema** (variables, spaces with landmark values/bounds,
constraints, corresponding values, regions) — the interchange format hosts
author against.

### BehaviorGraph and results

Directed graph over `QState`s with distinguished initial states; a
*behavior* is a root-to-terminal path. Every terminal carries a
`TerminalClass ∈ {QUIESCENT, CYCLE, DIVERGENT, REGION_EXIT, TRUNCATED}`.
Supports behavior iteration, state dedup (attainable envisionment),
path-predicate filtering, and a **neutral export** (node table + edge list +
labels as plain arrays; dot on top). Engines return a `SimResult`: the
graph, `status ∈ {COMPLETE, TRUNCATED}`, per-filter pruning statistics, and
the `SimConfig` + model hash that produced it — frozen, `to_dict()`-able,
designed to be wrapped into host provenance records.

### Engines and filters

An engine is a function-shaped object: `engine(compiled_model,
initial_states, *, config) -> SimResult`. Filters are named, individually
toggleable objects in three kinds: tuple filters (per-constraint), pairwise
propagation, and **global filters** over candidate successors/paths. Global
filters are a plugin point: built-ins (no-change, quiescence, cycle,
divergence) plus user-supplied path/state predicates — the hook through
which analytic knowledge (e.g. a declared energy-like variable that must not
increase) prunes spurious behaviors without touching core semantics.

## Cross-cutting decisions (proposed)

- **Immutability at the boundary:** models compile to frozen artifacts;
  states and results are values. Engines may mutate private scratch only.
- **Determinism:** given a model + config, output graphs are identical
  across runs and devices (documented ordering; no set-iteration
  nondeterminism in results).
- **Tensor layout conventions:** batch-first everywhere; qualitative
  frontier `(B, 2*V)` int; numeric trajectories `(B, T, V)` float with
  optional `(B, T)` times and optional `(B, T)` integer mode channel.
  Device/dtype follow inputs.
- **Naming:** QR-literature names in docs (`M+`, `deriv`, `landmark`,
  `envisionment`), ASCII-safe in code (`MPlus`, `Deriv`).
- **Python ≥ 3.10**, `torch` required but imported lazily outside
  `tensor/`/`bridge/` so the symbolic core works in torch-free environments.

## Testing strategy

- Golden-model tests: bathtub, U-tube, spring/mass (± friction), two-tank
  cascade — QSIM's published behavior sets are the expected outputs.
- Property tests: tensor engine ≡ reference engine on randomized small
  models; constraint tables ≡ constraint predicates on exhaustive small
  domains.
- **Soundness harness (the invariant of the whole library):** sample
  concrete instances of a model (monotone functions consistent with its
  M-constraints), integrate numerically, abstract the trajectories, and
  assert coverage in the predicted behavior graph. Runs as a randomized
  property test from phase 3 onward; doubles as the first end-to-end test
  of the bridge surfaces.
- Schema tests: JSON round-trips for models and results; export/import
  stability under the versioned schema.
