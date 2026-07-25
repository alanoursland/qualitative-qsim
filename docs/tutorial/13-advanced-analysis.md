# 13. Reasoning beyond one simulation

> **In this lesson:** distinguish reachable and total envisionments, classify
> temporal properties, inspect signed-graph consistency, and add stronger
> qualitative knowledge without overstating what it proves.

## Reachable versus total state spaces

Build the spring from Lesson 5:

```python
import qrlib as qr
from qrlib import Qdir, TimeTag

spring = qr.Model("spring")
for name in ("x", "v", "a"):
    spring.variable(name, landmarks=("0",), unbounded=True)
spring.constrain(qr.Deriv("x", "v"))
spring.constrain(qr.Deriv("v", "a"))
spring.constrain(qr.Minus("x", "a"))

spring_initial = spring.state(
    time=TimeTag.POINT,
    x=("0", Qdir.INC),
    v=(("0", "+inf"), Qdir.STD),
    a=("0", Qdir.DEC),
)
```

Ordinary simulation explores what is reachable from that initial state.
Attainable-envisionment mode merges equal reachable states into a graph:

```python
attainable = qr.qsim(
    spring,
    spring_initial,
    config=qr.SimConfig(envisionment=True, discover_landmarks=False),
)
```

Total envisionment starts without an initial state and enumerates every
constraint-consistent state in one region:

```python
total = qr.envision(spring)
print(total.stats["states"], len(total.cycles()))
```

The reachable graph must embed in the total portrait, but the total portrait
may contain equilibria and transient states unreachable from your chosen
initial condition. A hard `max_states` limit raises an error: a silently
partial “total” envisionment would be a contradiction in terms.

## Temporal logic: classify or guide

Classify the existing reachable graph when the question is “does this model
have behaviors satisfying the property?”:

```python
from qrlib import guide
from qrlib.guide import F, G, mag

cycle_result = qr.qsim(spring, spring_initial)
returns_to_zero = G(F(mag("x", "==", "0")))
classification = guide.classify(cycle_result, returns_to_zero)
print(classification.universal)               # True
```

Use `guide.guided` when the temporal formula is part of the problem
definition—an observation, boundary condition, or exogenous input—and you
want it to prune generation. The generated graph then covers real behaviors
*conditioned on that formula*.

## Signed-graph consistency

The basic monotonicity checker asks whether all `M+`, `M-`, and `Minus`
relationships admit one consistent assignment of orthant polarities:

```python
from qrlib.analysis import monotonicity

sign_model = qr.Model("signed cycle")
for name in ("a", "b", "c"):
    sign_model.variable(name, landmarks=("0",), unbounded=True)
sign_model.constrain(qr.MPlus("a", "b"))
sign_model.constrain(qr.MPlus("b", "c"))
sign_model.constrain(qr.MMinus("c", "a"))

certificate = monotonicity.check_signed_graph(sign_model)
print(certificate.is_consistent)              # False
print([edge.kind for edge in certificate.conflict_cycle])
```

The returned negative cycle is a concrete structural conflict witness. A
positive certificate is not a proof that a nonlinear vector field is a
monotone dynamical system; it is only an orthant-consistency check over the
declared signed relationships.

For multi-region models, check regions separately. Mutually exclusive
positive and negative relationships may each be consistent in their own
region even though their union is not.

## Order-of-magnitude knowledge

`Negligible(small, large)` declares `|small| < |large|` throughout every
region where it is active:

```python
ordered = qr.Model("dominant sum")
for name in ("small", "large", "total"):
    ordered.variable(name, landmarks=("0",), unbounded=True)
ordered.constrain(qr.Add("small", "large", "total"))
ordered.constrain(qr.Negligible("small", "large"))
ordered.compile()
```

That inequality lets sign algebra resolve sums whose dominant operand is
known. It is a global model assertion, not a floating-point “small enough”
tolerance. Cyclic declarations such as `Negligible(a, b)` and
`Negligible(b, a)` are contradictions.

## Phase-space non-intersection

For an autonomous phase pair connected by `Deriv(x, v)`, the optional phase
filter rejects provable non-monotone returns across the same directed
transversal:

```python
phase_checked = qr.qsim(
    spring,
    spring_initial,
    config=qr.SimConfig(
        discover_landmarks=False,
        phase_pairs=(("x", "v"),),
    ),
)
print(phase_checked.stats["phase_filtered"])  # 0 for this already-clean cycle
```

The filter acts only when landmark order proves a crossing conflict. Unknown
or unseparated crossings survive. It therefore strengthens QSIM without
turning the library into a numeric reachability engine.

## Choosing the right question

- Use `qsim` for futures reachable from an initial state.
- Use `envision` for the complete qualitative portrait of one region.
- Use `guide.classify` for temporal model checking of an existing graph.
- Use `guide.guided` to condition generation on a temporal specification.
- Use monotonicity checking for signed-graph contradictions.
- Use `Negligible`, energy, or phase filtering only when their premises are
  valid physical knowledge.

## Exercises

1. Replace the final `MMinus` in the signed cycle with `MPlus` and inspect the
   polarity assignment.
2. Compare `len(attainable.graph.nodes)` with the number of nodes in `total`.
3. Turn landmark discovery on for the phase-checked spring and compare
   behavior counts with and without `phase_pairs`.

---

Next: [**14. Composition, tensors, and scale →**](14-composition-and-scale.md)
