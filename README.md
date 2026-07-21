# qualitative_reasoning_lib

A Python/PyTorch library for **qualitative reasoning (QR)** about dynamical
systems: simulating and analyzing the behavior of continuous systems when the
governing equations are only partially known — described by monotonic
relationships, signs, orderings, and landmark values rather than exact
functions and parameters.

The initial centerpiece is an implementation of **QSIM** (Kuipers-style
qualitative simulation), with the architecture deliberately laid out so other
QR formalisms (envisionment, process-based modeling, semi-quantitative
refinement) and tensorized/GPU-accelerated execution can slot in alongside it.

> **Status: exploratory / pre-alpha.** The repository currently contains
> design documents and a provisional package skeleton. APIs here are sketches
> meant to be argued with, not depended on. See [`docs/roadmap.md`](docs/roadmap.md).

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

## Planned shape of the library

```
Model description        Reasoning engines           Analysis / bridge
─────────────────        ─────────────────           ──────────────────
QuantitySpace            QSIM simulation             Behavior graphs
Variables (mag, dir)     Attainable envisionment     Trajectory abstraction
Constraints (M+, ADD,    Batched/tensorized          (numeric → qualitative)
  DERIV, MULT, ...)        filtering on GPU          Consistency checking
Corresponding values     Semi-quantitative (Q2-ish)  Visualization
```

- **`docs/`** — design notes: [vision](docs/vision.md),
  [QR landscape survey](docs/landscape.md),
  [architecture](docs/architecture.md), [QSIM deep-dive](docs/qsim.md),
  [tensorization & GPU strategy](docs/gpu-tensorization.md),
  [bridge to numeric dynamical systems](docs/numeric-bridge.md),
  [roadmap](docs/roadmap.md), [open questions](docs/open-questions.md).
- **`src/qrlib/`** — provisional package skeleton. Core representations
  (quantity spaces, qualitative values, constraints, models) are small real
  implementations; engines are stubs.
- **`tests/`** — seed tests for the core representations.

## Quick taste (target API, subject to change)

```python
import qrlib as qr

m = qr.Model("bathtub")
amount  = m.variable("amount",  landmarks=("0", "FULL"), upper_unbounded=True)
level   = m.variable("level",   landmarks=("0",), upper_unbounded=True)
outflow = m.variable("outflow", landmarks=("0",), upper_unbounded=True)
inflow  = m.variable("inflow",  landmarks=("0", "IF*"), upper_unbounded=True)
netflow = m.variable("netflow", landmarks=("0",), unbounded=True)

m.constrain(qr.MPlus(amount, level))            # level rises with amount
m.constrain(qr.MPlus(level, outflow))           # outflow rises with level
m.constrain(qr.Add(netflow, outflow, inflow))   # netflow + outflow = inflow
m.constrain(qr.Deriv(amount, netflow))          # d(amount)/dt = netflow
m.constrain(qr.Constant(inflow))

behaviors = qr.qsim(m, initial=..., max_states=500)
behaviors.plot()   # branching behavior tree: equilibrium vs. overflow, etc.
```

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
4. **Soundness is sacred, spuriousness is managed.** Like QSIM itself: never
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
