# 15. End-to-end host integration

> **In this lesson:** put the public seams together as a numeric dynamical
> systems host would: author, identify, simulate, abstract, check, refine,
> classify, and serialize.

The host owns numeric integration, event detection, parameter fitting, and
domain objects. qrlib owns the qualitative model and the reasoning performed
over it.

## 1. Author a portable hybrid model

```python
import json
import qrlib as qr
from qrlib import Landmark, Qdir, TimeTag
from qrlib.bridge import abstraction, coverage
from qrlib.bridge.abstraction import AbstractionConfig

model = qr.Model("fill then hold")
model.variable(
    "amount",
    landmarks=(Landmark("0", 0.0), Landmark("FULL", 1.0)),
)
model.variable(
    "rate",
    landmarks=(Landmark("0", 0.0),),
    unbounded=True,
)

deriv = model.constrain("Deriv(amount, rate)")
rate_constant = model.constrain("Constant(rate)")
amount_constant = model.constrain("Constant(amount)")
model.region("filling", constraints=(deriv, rate_constant))
model.region("holding", constraints=(amount_constant, rate_constant))
model.transition("filling", "holding", when=(("amount", "==", "FULL"),))
model.initial_region = "filling"

model_payload = model.to_dict()
model_id = model.content_hash()
assert qr.Model.from_dict(model_payload).content_hash() == model_id
```

The host can persist `model_payload` and use `model_id` as the stable identity
connecting later results to this exact qualitative specification.

## 2. Simulate the qualitative futures

```python
initial = model.state(
    time=TimeTag.POINT,
    amount=("0", Qdir.INC),
    rate=(("0", "+inf"), Qdir.STD),
)
simulation = qr.qsim(model, initial)
assert simulation.model_hash == model_id
print(simulation.status.value, len(simulation.behaviors()))  # complete 1
```

This model has one qualitative future: fill to `FULL`, cross into `holding`,
then remain quiescent.

## 3. Abstract a host trajectory

The host supplies values in model variable order, timestamps, and its mode
channel:

```python
rows = [
    [0.0, 1.0], [0.4, 1.0], [0.8, 1.0], [1.0, 1.0],
    [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
]
times = [0.0, 0.4, 0.8, 1.0, 1.1, 1.2, 1.3]
modes = ["filling"] * 4 + ["holding"] * 3
abstraction_config = AbstractionConfig(debounce=1)

observed = abstraction.abstract_trajectory(
    rows,
    model,
    times=times,
    modes=modes,
    config=abstraction_config,
)
```

An event-aware solver would additionally pass `CrossingEvent` objects for
roots that occur between samples.

## 4. Check coverage

```python
coverage_result = coverage.check(observed, simulation.graph)
assert coverage_result.covered
print(coverage_result.witness)
```

Persist the witness when covered. When refuted, persist the divergence index,
diagnosis, abstraction configuration, sample spans, and time bounds before
deciding whether the model or the abstraction boundary is responsible.

## 5. Add numeric refinement

```python
(behavior,) = simulation.behaviors()
refinement = qr.semiquant.refine(simulation.graph, behavior)
print(refinement.feasible)                    # True
```

Numeric landmark values narrow the behavior. Richer monotone envelopes can
add transition-time bounds without changing the qualitative graph.

## 6. Classify a temporal requirement

```python
from qrlib import guide
from qrlib.guide import F, mag

eventually_full = F(mag("amount", "==", "FULL"))
property_result = guide.classify(simulation, eventually_full)
print(property_result.universal)              # True
```

This is a universal statement over the simulated graph. Its proof strength
still follows the QSIM asymmetry: universal properties over a sound
over-approximation are meaningful; a satisfying path alone would not prove
physical existence.

## 7. Emit a plain-data record

```python
record = {
    "model": model_payload,
    "simulation": simulation.to_dict(),
    "observation": {
        "variable_order": list(observed.var_order),
        "regions": list(observed.regions),
        "spans": [list(span) for span in observed.spans],
        "time_bounds": [list(bound) for bound in observed.time_bounds],
        "abstraction_config": {
            "landmark_atol": observed.config.landmark_atol,
            "direction_eps": observed.config.direction_eps,
            "debounce": observed.config.debounce,
        },
    },
    "coverage": {
        "covered": coverage_result.covered,
        "witness": list(coverage_result.witness),
        "diagnosis": coverage_result.diagnosis,
    },
    "refinement": refinement.to_dict(),
    "property": property_result.to_dict(),
}
json.dumps(record)
```

No simulator-specific object is required in this record. A larger host can
attach its own solver version, parameter identity, units, tolerances, and
artifact provenance alongside qrlib's data.

## The integration boundary

Keep these responsibilities separate:

| Numeric host owns | qrlib owns |
|---|---|
| ODE/DAE integration and root finding | Qualitative model semantics |
| Parameters, units, and fitted functions | QSIM and envisionment |
| Exact hybrid event state | Trajectory abstraction |
| Numeric reachability, if available | Qualitative coverage witnesses |
| Application storage and UI | Plain-data model/result exports |
| Domain-specific fault priors | Consistency-based diagnosis |

That boundary lets the qualitative layer be rewritten or embedded without
forcing qrlib to become a numerical solver.

## Exercises

1. Change one mode label to `holding` before the crossing and preserve the
   resulting refutation in `record`.
2. Change `rate` from constant to unconstrained and inspect how the qualitative
   behavior graph expands.
3. Add a host-side solver identifier and parameter hash to the record without
   changing qrlib's schemas.

---

You have completed both tutorial tracks. Return to the
[tutorial index](README.md) or use the [feature map](feature-map.md) to find a
specific public surface.
