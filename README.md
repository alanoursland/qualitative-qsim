# qualitative-qsim

A Python/PyTorch library for **qualitative reasoning (QR)** about dynamical
systems: simulating and analyzing the behavior of continuous systems when the
governing equations are only partially known — described by monotonic
relationships, signs, orderings, and landmark values rather than exact
functions and parameters.

The initial centerpiece is an implementation of **QSIM** (Kuipers-style
qualitative simulation), with the architecture deliberately laid out so other
QR formalisms (envisionment, process-based modeling, semi-quantitative
refinement) and tensorized/GPU-accelerated execution can slot in alongside it.

> **Status: functional, pre-release.** The practical QSIM path, core model
> representation, behavior graphs, serialization, and numeric coverage bridge
> are the supported pre-release core. The broader research surface is
> deliberately exposed at lower maturity levels rather than implied to have
> uniform validation. The project is available under the
> [MIT License](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/LICENSE);
> APIs are stable-ish but unversioned.

| Maturity | Surface | What the label means |
|---|---|---|
| **Core alpha** | Model/QSIM practical profile, behavior graphs, schemas, trajectory abstraction and coverage | Supported pre-release path; golden, adversarial, integration, and concrete-trajectory soundness checks |
| **Experimental** | Classic landmark discovery, chatter controls, energy/Lyapunov and phase filters, Q2 refinement, tensor execution and differentiable losses | Semantics and limitations are tested, but configuration and performance contracts may change |
| **Research preview** | Diagnosis, decomposition, induction, envisionment/guidance, process/device front ends, monotonicity and comparative/causal analysis | Usable and evidence-backed, but narrower scenario coverage; independently audit before consequential use |

The
[maturity and qualification record](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/maturity.md)
lists the evidence and known limits behind each label. The
[roadmap](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/roadmap.md)
records implementation scope, not a claim that every module has equal
validation depth.

## Why qualitative reasoning?

A numeric simulator answers *"what does **this** system with **these**
parameters do from **this** initial condition?"* A qualitative simulator
answers a complementary question: *"what are **all** the behaviors any system
consistent with this qualitative description can exhibit?"*

That makes QR useful for:

- **Incomplete models** — reasoning when you know `flow increases with level`
  but not the pipe's discharge coefficient.
- **Behavior enumeration** — producing the complete branching tree of
  qualitatively distinct outcomes (overflows / reaches equilibrium / oscillates),
  with guarantees that no real behavior is missed.
- **Abstraction of numeric systems** — compressing families of numeric
  trajectories into a small graph of qualitative states, giving a discrete,
  symbolic summary of a continuous system's phase portrait.
- **Verification and explanation** — checking that a fitted or learned numeric
  model only does things the qualitative physics allows, and explaining
  behaviors in human terms ("the level rises, decelerating, toward equilibrium").

## Shape of the library

```
Model description        Reasoning engines           Analysis / bridge
─────────────────        ─────────────────           ──────────────────
QuantitySpace            QSIM simulation             Behavior graphs + queries
Variables (mag, dir)     Attainable envisionment     Trajectory abstraction
Constraints (M+, ADD,    Batched/tensorized          Coverage oracle
  DERIV, MULT, ...)        filtering (torch)         Sign-structure intake
Corresponding values     Semi-quantitative (Q2)      Explanation + viz
Operating regions        Landmark discovery          Landmark harvest/proposal
```

- **`docs/`** — design notes:
  [vision](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/vision.md),
  the hands-on
  [tutorial](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/tutorial/README.md),
  [QR landscape survey](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/landscape.md),
  [research references and implementation lineage](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/references.md),
  [architecture](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/architecture.md),
  [QSIM deep-dive](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/qsim.md),
  [compact constraint syntax](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/constraint-syntax.md),
  [signed-graph consistency](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/monotonicity.md),
  [tensorization & GPU strategy](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/gpu-tensorization.md),
  [production-shaped scale profiles](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/scale-profiles.md),
  [numeric bridge](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/numeric-bridge.md),
  [host integration](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/host-integration.md),
  [piecewise-affine analysis](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/piecewise-affine.md),
  [literature survey](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/literature-survey.md),
  [roadmap](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/roadmap.md),
  and
  [open questions](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/open-questions.md).
- **`src/qrlib/`** — the library: core representations, the reference and
  tensorized engines, the numeric bridge, semi-quantitative refinement,
  analysis, and visualization (the
  [architecture guide](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/architecture.md)
  maps the layout).
- **`tests/`** — golden models, equivalence properties, and the soundness
  harness (concrete ODE instances integrated, abstracted, and verified
  against predicted behavior graphs). **`benchmarks/`** — measured
  reference-vs-tensor comparisons.

## Installation

The supported install includes PyTorch:

```console
pip install qualitative-qsim
```

PyTorch is imported lazily: ordinary model construction, reference
simulation, analysis, and the pure-Python bridge do not import it, while
`SimConfig(backend="auto")` uses tensor filtering only when the measured
workload shape benefits; explicit `"reference"` / `"tensor"` modes and
`qrlib.tensor` use the installed dependency directly.

## Quick taste

```python
import qrlib as qr
from qrlib import Qdir
from qrlib.analysis import explain

m = qr.Model("bathtub")
m.variable("amount", landmarks=("0", "FULL"))
m.variable("level", landmarks=("0", "TOP"))
m.variable("outflow", landmarks=("0", "OMAX"))
m.variable("inflow", landmarks=("0", "IF*"))
m.variable("netflow", landmarks=("0",), unbounded=True)

m.constrain(qr.MPlus("amount", "level", cvals=(("0", "0"), ("FULL", "TOP"))))
m.constrain(qr.MPlus("level", "outflow", cvals=(("0", "0"), ("TOP", "OMAX"))))
m.constrain(qr.Add("netflow", "outflow", "inflow"))  # netflow + outflow = inflow
m.constrain(qr.Deriv("amount", "netflow"))           # d(amount)/dt = netflow
m.constrain(qr.Constant("inflow"))

initial = m.state(
    amount=("0", Qdir.INC), level=("0", Qdir.INC), outflow=("0", Qdir.INC),
    inflow=("IF*", Qdir.STD), netflow=(("0", "+inf"), Qdir.DEC),
)
result = qr.qsim(m, initial)
for b in result.behaviors():
    print(explain.narrate(result.graph, b))
```

yields exactly the three textbook outcomes — equilibrium below FULL,
equilibrium exactly at FULL, and overflow. The practical default keeps an
intermediate equilibrium unnamed:

```
Behavior of 'bathtub': 3 states, ending in quiescent.
  0. Initially, amount at 0, rising, level at 0, rising, ...
  1. Then, over an interval, amount rises into (0, FULL); ...
  2. At the next instant, amount levels off in (0, FULL); ...
     — the system is in equilibrium ...
```

`SimConfig()` is the sound, bounded practical profile: landmark discovery is
off and structurally unobservable direction chatter is merged automatically.
Use `SimConfig.classic()` when named intermediate extrema and full textbook
QSIM discovery are required; classic runs may not terminate and report
actionable diagnostics when a resource limit is reached.

Constraints may equivalently use the optional compact authoring syntax, for
example `m.constrain("Deriv(amount, netflow)")`; models still store and
serialize ordinary constraint objects.

From here: `qrlib.bridge.coverage` checks numeric trajectories against the
graph (witness paths / localized refutations), `qrlib.semiquant` turns
landmark bounds and monotone-function envelopes into guaranteed value and
transition-time bounds (and prunes numerically impossible behaviors), and
`qrlib.viz` renders timelines and behavior trees as plain data or SVG.
Additional modules provide total envisionment, temporal-logic guidance,
model induction and diagnosis, comparative/causal analysis, decomposition,
differentiable constraint losses, and process/device modeling front ends.
`qrlib.analysis.monotonicity.check_signed_graph` additionally certifies
whether all declared `M+`/`M-`/`Minus` relationships admit one consistent
orthant ordering, returning a contradictory signed cycle when they do not.

## Design commitments (early)

1. **Model description is decoupled from every engine.** A `Model` is pure
   data; QSIM, envisioners, tensor engines, and abstraction tools all consume
   the same description.
2. **PyTorch-native state encoding.** Qualitative states have a canonical
   integer-tensor encoding so that large frontiers, ensembles of models, and
   batches of numeric trajectories can be processed with batched tensor ops
   (GPU when it helps; the semantics never require it).
3. **Numeric systems are first-class neighbors.** Interfaces are shaped so a
   numeric dynamical system (a vector field / trajectory source) can be
   abstracted into, or checked against, a qualitative model — see
   [the numeric bridge](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/numeric-bridge.md).
4. **Embeddable by design.** Larger dynamical-systems toolkits should be able
   to build thin adapter modules on top of qrlib — names as canonical
   identity, no CAS/graph-library dependencies in core, tensors as the only
   numeric interchange, serializable witness-carrying results — see
   [host integration](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/host-integration.md).
5. **Soundness is sacred, spuriousness is managed.** Like QSIM itself: never
   drop a real behavior; add filters to prune impossible ones.

## Research lineage and citation

The core semantics follow
[Kuipers1986](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/references.md#kuipers1986)
and the authoritative
[Kuipers1994](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/references.md#kuipers1994)
presentation. Process-centered authoring draws from
[Forbus1984](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/references.md#forbus1984),
device composition and envisionment from
[deKleerBrown1984](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/references.md#dekleerbrown1984),
and
semi-quantitative refinement from
[KuipersBerleant1988](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/references.md#kuipersberleant1988).

The
[canonical annotated bibliography](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/docs/references.md)
maps every research-derived feature to its source. Machine-readable records
are in
[`paper.bib`](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/paper.bib),
software citation metadata is in
[`CITATION.cff`](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/CITATION.cff),
and the draft JOSS article is
[`paper.md`](https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/paper.md).
