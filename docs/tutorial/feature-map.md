# Tutorial feature map

This page accounts for qrlib's public feature surface. It distinguishes what
the tutorial teaches in depth from what it introduces briefly and what remains
an implementation detail.

| Public area | What it provides | Tutorial coverage |
|---|---|---|
| `Landmark`, `QuantitySpace`, `QVal`, `QState` | Qualitative magnitudes, directions, and states | Lessons 2–3 |
| Built-in constraints | `M+`, `M-`, `Add`, `Mult`, `Minus`, `Deriv`, `Constant`, `At`, `Negligible` | Lesson 3; `At` in 9/12; `Negligible` in 13 |
| `constraint_syntax` | Safe compact parsing and canonical formatting | Lesson 10 |
| `Model` and `CompiledModel` | Authoring, regions, validation, schemas, stable identity | Lessons 3, 9, and 10 |
| `qsim`, `SimConfig`, `SimResult` | Reference/tensor-dispatched qualitative simulation | Lessons 4–6 and 14 |
| `BehaviorGraph` and terminal classes | Branching results, cycles, queries, export | Lessons 4–5 and 10 |
| Landmark discovery | Per-branch introduction of newly important values | Lesson 6 |
| Dynamic chatter abstraction | Structural merging of unobservable direction distinctions | Lesson 6 |
| `EnergyFilter` | Declarative conserved/nonincreasing amplitude pruning | Lesson 6 |
| Phase-pair filtering | Sound phase-space non-intersection pruning | Lesson 13 |
| Attainable envisionment | Reachable graph merging from an initial state | Lesson 13 |
| `envision` | Total consistent-state portrait for a region | Lesson 13 |
| `bridge.abstraction` | Sampled trajectory abstraction, modes, events, batching | Lessons 7 and 11 |
| `bridge.coverage` | Witnesses, localized refutations, and aggregate score | Lessons 7, 11, and 15 |
| `bridge.harvest` | Landmark intake and proposal from steady data | Lesson 12 |
| `bridge.signs` | Sign intake, estimation, calibration, and consistency | Lesson 12 |
| `semiquant` | Q2-style numeric bounds, envelopes, time bounds, pruning | Lesson 8 |
| `tensor.interval` | Batched algebraic interval narrowing and feasibility | Lesson 14 |
| `analysis.queries` | Terminal census, equilibria, cycles, and state search | Lessons 4–5 |
| `analysis.explain` | Structured events and prose behavior narration | Lesson 4 |
| `analysis.causal` | Structural causal ordering and singularity reporting | Lessons 3 and 9 |
| `analysis.compare` | Equilibrium comparative statics | Lesson 9 |
| `analysis.monotonicity` | Signed-graph orthant consistency certificates | Lesson 13 |
| `guide` | LTL classification, guided simulation, exogenous inputs | Lessons 9 and 13 |
| `diagnosis` | Minimal consistency-based fault explanations | Lessons 9 and 12 |
| `induce` | Ranked QDE structure candidates learned from trajectories | Lesson 12 |
| `decompose` | DecSIM-style partition, component guidance, and joining | Lesson 14 |
| `frontends.qpt` | Process-centric model authoring | Lesson 14 |
| `frontends.devices` | Reusable component and netlist authoring | Lesson 14 |
| `tensor.encoding` and `tensor.engine` | Tensor state encoding and batched filtering | Lesson 14 |
| `tensor.abstraction` | Device-resident batched trajectory front end | Lessons 11 and 14 |
| `tensor.losses` | Differentiable numeric constraint losses | Lesson 14 |
| `viz` | Data-first timeline/tree layouts and dependency-free SVG | Lessons 4–5 and 8 |
| Model/result provenance | Versioned plain-data exports and replay metadata | Lessons 10 and 15 |

The transition tables, constraint tables, landmark-frame mechanics, graph
algorithms, and individual engine filters are implementation details. Their
semantics are taught through the public APIs rather than by asking users to
call internal functions.

Features explicitly outside the open library—full numeric reachability,
general hybrid-system conformance, spatial reasoning, and a GUI model
builder—are discussed in the design documents, not presented as tutorial
capabilities.

← [Back to the tutorial index](README.md)
