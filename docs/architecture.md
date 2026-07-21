# Architecture

Provisional package layout and core abstractions. Everything here is a
proposal; the skeleton in `src/qrlib/` mirrors it so the proposal stays
honest.

## Package layout

```
src/qrlib/
├── __init__.py          # public API re-exports
├── quantity.py          # QuantitySpace, Qmag, Qdir, QVal
├── state.py             # QState (one qualitative state of a model)
├── constraints.py       # Constraint types: MPlus, MMinus, Add, Mult, Deriv, ...
├── model.py             # Variable, Model (declarative QDE description)
├── behavior.py          # BehaviorGraph / behavior trees over QStates      [planned]
├── engines/
│   ├── __init__.py
│   ├── qsim.py          # reference pure-Python QSIM                       [stub]
│   ├── transitions.py   # P-/I-transition tables                           [stub]
│   └── filters.py       # constraint, Waltz, and global filters            [stub]
├── tensor/
│   ├── __init__.py
│   ├── encoding.py      # QState/frontier <-> integer tensor codecs        [stub]
│   └── engine.py        # batched/tensorized QSIM                          [planned]
├── bridge/
│   ├── __init__.py      # numeric <-> qualitative bridge                   [planned]
│   ├── abstraction.py   # trajectory batches -> qualitative behaviors      [planned]
│   └── consistency.py   # check numeric runs against behavior graphs      [planned]
└── viz/                 # behavior-graph rendering                         [planned]
```

Rules of the layout:

- `quantity/state/constraints/model` are **pure data + trivial logic**, no
  torch imports, importable anywhere, exhaustively unit-tested.
- `engines/` may only *read* models/states. The reference engine is written
  for clarity, not speed.
- `tensor/` is the only place that imports torch in anger. It must be
  behavior-equivalent to `engines/` and is tested against it.
- `bridge/` depends on tensor layouts, not on any external simulator's API.

## Core abstractions

### QuantitySpace

A totally ordered tuple of named landmarks, optionally open-ended at either
end (conceptually `-inf` / `+inf` endpoints). Example: `(0, FULL, +inf)`.
Landmark identity is per-variable; new landmarks can be inserted between
existing neighbors during simulation (QSIM landmark discovery), so positions
are identities, not indices — but every space can render itself to a dense
index mapping for tensor encoding.

### Qualitative value: `QVal = (Qmag, Qdir)`

- `Qmag`: either *at* landmark `l_i` or *in* the open interval `(l_i, l_i+1)`.
  Encoded as a single integer `2*i` (at landmark i) / `2*i+1` (in interval
  above landmark i) — the "rank" encoding; adjacency and ordering become
  integer arithmetic.
- `Qdir ∈ {DEC, STD, INC}`, encoded `0/1/2` (so `sign = qdir - 1`).

### QState

An immutable mapping `variable -> QVal` plus a time tag (`POINT` or
`INTERVAL` — QSIM alternates). Hashable, comparable, with a canonical dense
encoding: an int tensor of shape `(2*V,)` (interleaved mag-rank and dir per
variable) given a frozen landmark→index mapping. Frontiers stack to `(B, 2*V)`.

### Constraints

Small dataclasses referencing variables by name: `MPlus(x, y)`,
`MMinus(x, y)`, `Add(x, y, z)`, `Mult(x, y, z)`, `Minus(x, y)`,
`Deriv(x, y)`, `Constant(x)`, each optionally carrying **corresponding
values** (tuples of landmarks that co-occur). Each constraint type knows how
to answer one question — *is this tuple of QVals consistent with me?* — in two
interchangeable forms:

1. a readable Python predicate (reference), and
2. a dense boolean lookup table over encoded values (for the tensor engine),
   built once per model compilation.

### Model

`Model` = named `Variable`s (each owning a QuantitySpace) + constraints +
(later) operating regions with guard conditions and region-transition rules.
`Model.compile()` freezes name→index and landmark→rank mappings and
precomputes constraint tables, yielding a `CompiledModel` that engines
consume. The un-compiled `Model` stays editable and serializable (plain-dict
schema, so models can be stored as JSON/YAML and generated
programmatically — important for future interop where another system emits
qualitative model descriptions).

### BehaviorGraph

Directed graph over `QState`s with distinguished initial states; a *behavior*
is a root-to-leaf/-cycle path. Supports: iteration over behaviors, state
dedup (turning the tree into an attainable envisionment), predicates over
paths (for filtering/queries), and export to dot/graphviz. This is the main
user-facing result object.

### Engines

An engine is a function-shaped object: `engine(compiled_model,
initial_states, *, limits, filters) -> BehaviorGraph`. QSIM (reference and
tensorized) both fit; so will envisioners. Filters are named and
individually toggleable so soundness/spuriousness tradeoffs are explicit and
testable.

## Cross-cutting decisions (proposed)

- **Immutability at the boundary:** models compile to frozen artifacts;
  states are values. Engines may mutate private scratch, nothing else.
- **Determinism:** given a model + config, output graphs are identical across
  runs and devices (ordering conventions documented; no set-iteration
  nondeterminism in results).
- **Tensor layout conventions:** batch-first everywhere; qualitative frontier
  `(B, 2*V)` int; numeric trajectories `(B, T, V)` float with an optional
  companion time tensor `(B, T)`. Device/dtype follow inputs.
- **Naming:** stick to QR-literature names (`M+`, `deriv`, `landmark`,
  `envisionment`) in docs, ASCII-safe versions in code (`MPlus`, `Deriv`).
- **Python ≥ 3.10**, `torch` as a required dependency but imported lazily
  outside `tensor/` so the symbolic core works in torch-free environments
  (keeps CI cheap and the reference engine honest).

## Testing strategy

- Golden-model tests: bathtub, U-tube, spring/mass (± friction), two-tank
  cascade — QSIM's published behavior sets are the expected outputs.
- Property tests: tensor engine ≡ reference engine on randomized small
  models; constraint tables ≡ constraint predicates on exhaustive small
  domains.
- Soundness spot-checks: integrate concrete ODE instances of a model
  numerically, abstract the trajectory, assert it is a path in the predicted
  behavior graph (this doubles as the first bridge test).
