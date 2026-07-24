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
| `LyapunovCertificate` | Conditional strict scalar decrease and recurrence pruning | Lesson 6 |
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

## Research lineage for tutorial features

The table below distinguishes research-derived functionality from original
library engineering. Complete annotated records are in
[`docs/references.md`](../references.md).

| Tutorial capability | Direct lineage |
|---|---|
| QSIM states, transitions, landmark discovery, and coverage semantics | [Kuipers1986](../references.md#kuipers1986), [Kuipers1994](../references.md#kuipers1994) |
| Semi-quantitative refinement | [KuipersBerleant1988](../references.md#kuipersberleant1988) |
| Energy and Lyapunov pruning | [FoucheKuipers1992](../references.md#fouchekuipers1992) |
| Dynamic chatter abstraction | [ClancyKuipers1997Chatter](../references.md#clancykuipers1997chatter) |
| Phase-space non-intersection | [LeeKuipers1988](../references.md#leekuipers1988) |
| Guided simulation and temporal classification | [BrajnikClancy1998](../references.md#brajnikclancy1998), [ShultsKuipers1997](../references.md#shultskuipers1997) |
| Diagnosis with behavioral modes | [SubramanianMooney1996](../references.md#subramanianmooney1996) |
| Decomposed simulation | [ClancyKuipers1997](../references.md#clancykuipers1997) |
| Process-centered authoring | [Forbus1984](../references.md#forbus1984) |
| Device composition and total envisionment | [deKleerBrown1984](../references.md#dekleerbrown1984) |
| Causal ordering | [IwasakiSimon1986](../references.md#iwasakisimon1986), [deKleerBrown1986](../references.md#dekleerbrown1986) |
| Comparative analysis | [ChiuKuipers1992](../references.md#chiukuipers1992) |
| Negligibility / order of magnitude | [Raiman1986](../references.md#raiman1986) |
| Signed-graph consistency | [Harary1953](../references.md#harary1953) |
| QDE induction | [RichardsKraanKuipers1992](../references.md#richardskraankuipers1992) |

Tensor encoding, backend selection, numeric-bridge schemas, provenance
records, compact constraint syntax, SVG rendering, and the exact public APIs
are qrlib engineering contributions. PyTorch is the tensor execution
dependency ([PaszkeEtAl2019](../references.md#paszkeetal2019)).

← [Back to the tutorial index](README.md)
