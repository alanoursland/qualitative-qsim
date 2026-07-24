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

> **Status: functional, pre-release.** Phases 0-7 of the
> [roadmap](docs/roadmap.md) are implemented and covered by the test suite:
> full QSIM (landmark discovery, chatter abstraction, analytic filters,
> envisionment, operating regions), the numeric bridge (trajectory
> abstraction, the coverage oracle, sign-structure intake/estimation),
> Q2-style semi-quantitative bounds, a torch-backed tensor layer proven
> equivalent to the reference engine, model induction and diagnosis,
> temporal-logic-guided simulation, decomposition, comparative and causal
> analysis, process/device front ends, explanation, and visualization.
> No license has been chosen yet; APIs are stable-ish but unversioned.

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

- **`docs/`** — design notes: [vision](docs/vision.md),
  the hands-on [tutorial](docs/tutorial/README.md),
  [QR landscape survey](docs/landscape.md),
  [architecture](docs/architecture.md), [QSIM deep-dive](docs/qsim.md),
  [compact constraint syntax](docs/constraint-syntax.md),
  [signed-graph consistency](docs/monotonicity.md),
  [tensorization & GPU strategy](docs/gpu-tensorization.md),
  [production-shaped scale profiles](docs/scale-profiles.md),
  [bridge to numeric dynamical systems](docs/numeric-bridge.md),
  [embedding in a host toolkit](docs/host-integration.md),
  [piecewise-affine qualitative analysis](docs/piecewise-affine.md),
  [QR literature survey](docs/literature-survey.md),
  [roadmap](docs/roadmap.md), [open questions](docs/open-questions.md).
- **`src/qrlib/`** — the library: core representations, the reference and
  tensorized engines, the numeric bridge, semi-quantitative refinement,
  analysis, and visualization (`docs/architecture.md` maps the layout).
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

yields exactly the three textbook outcomes — equilibrium below FULL (at a
*discovered* landmark `amount*0`), equilibrium exactly at FULL, and
overflow:

```
Behavior of 'bathtub': 3 states, ending in quiescent.
  0. Initially, amount at 0, rising, level at 0, rising, ...
  1. Then, over an interval, amount rises into (0, FULL); ...
  2. At the next instant, amount becomes steady at amount*0 (a newly
     identified value); ... — the system is in equilibrium ...
```

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
   [`docs/numeric-bridge.md`](docs/numeric-bridge.md).
4. **Embeddable by design.** Larger dynamical-systems toolkits should be able
   to build thin adapter modules on top of qrlib — names as canonical
   identity, no CAS/graph-library dependencies in core, tensors as the only
   numeric interchange, serializable witness-carrying results — see
   [`docs/host-integration.md`](docs/host-integration.md).
5. **Soundness is sacred, spuriousness is managed.** Like QSIM itself: never
   drop a real behavior; add filters to prune impossible ones.

## References (orientation)

- B. Kuipers, *Qualitative Simulation*, Artificial Intelligence 29 (1986).
- B. Kuipers, *Qualitative Reasoning: Modeling and Simulation with Incomplete
  Knowledge*, MIT Press (1994).
- K. Forbus, *Qualitative Process Theory*, Artificial Intelligence 24 (1984).
- J. de Kleer & J. S. Brown, *A Qualitative Physics Based on Confluences*,
  Artificial Intelligence 24 (1984).
- Kuipers & Berleant, *Using Incomplete Quantitative Knowledge in Qualitative
  Reasoning* (Q2), AAAI (1988).
