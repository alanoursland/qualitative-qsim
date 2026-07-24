# 11. Trustworthy hybrid trajectory abstraction

> **In this lesson:** preserve exact solver crossings, carry hybrid modes
> through abstraction, score batches, and interpret coverage failures without
> blaming the wrong layer.

Lesson 7 used regularly sampled rows. Production solvers usually know more:
irregular timestamps, exact event roots, and the active hybrid mode.

## Inferred and exact crossings

Without an event record, a crossing time is only bracketed by adjacent
samples:

```python
import qrlib as qr
from qrlib import Landmark
from qrlib.bridge import abstraction, coverage
from qrlib.bridge.abstraction import AbstractionConfig, CrossingEvent

crossing_model = qr.Model("crossing")
crossing_model.variable(
    "x",
    landmarks=(Landmark("0", 0.0),),
    unbounded=True,
)

inferred = abstraction.abstract_trajectory(
    [[-1.0], [1.0]],
    crossing_model,
    times=[0.0, 1.0],
)
print(inferred.time_bounds[1])               # (0.0, 1.0)
```

An event-aware solver can provide the root time and its complete state:

```python
exact = abstraction.abstract_trajectory(
    [[-1.0], [1.0]],
    crossing_model,
    times=[0.0, 1.0],
    crossings=[CrossingEvent(0.25, "x", "0", {"x": 0.0})],
)
print(exact.time_bounds[1])                  # (0.25, 0.25)
```

Events must lie strictly between their bracketing samples. Multiple
simultaneous variable crossings share a complete solver state so the emitted
qualitative point is physically coherent.

## A small hybrid model

This system fills at a positive constant rate, then switches to a holding
mode at `FULL`. Magnitudes are continuous across a region transition;
directions are re-derived under the target constraints.

```python
from qrlib import Qdir, TimeTag

hybrid = qr.Model("fill then hold")
hybrid.variable(
    "amount",
    landmarks=(Landmark("0", 0.0), Landmark("FULL", 1.0)),
)
hybrid.variable(
    "rate",
    landmarks=(Landmark("0", 0.0),),
    unbounded=True,
)

deriv = hybrid.constrain(qr.Deriv("amount", "rate"))
rate_constant = hybrid.constrain(qr.Constant("rate"))
amount_constant = hybrid.constrain(qr.Constant("amount"))
hybrid.region("filling", constraints=(deriv, rate_constant))
hybrid.region("holding", constraints=(amount_constant, rate_constant))
hybrid.transition("filling", "holding", when=(("amount", "==", "FULL"),))
hybrid.initial_region = "filling"

initial = hybrid.state(
    time=TimeTag.POINT,
    amount=("0", Qdir.INC),
    rate=(("0", "+inf"), Qdir.STD),
)
predicted = qr.qsim(hybrid, initial)
```

Now preserve the numeric host's mode channel:

```python
rows = [
    [0.0, 1.0], [0.4, 1.0], [0.8, 1.0], [1.0, 1.0],
    [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
]
times = [0.0, 0.4, 0.8, 1.0, 1.1, 1.2, 1.3]
modes = ["filling"] * 4 + ["holding"] * 3
cfg = AbstractionConfig(debounce=1)

observed = abstraction.abstract_trajectory(
    rows,
    hybrid,
    times=times,
    modes=modes,
    config=cfg,
)
checked = coverage.check(observed, predicted.graph)
print(checked.covered, observed.regions[-1])  # True holding
```

A wrong mode is a real mismatch:

```python
wrong_mode = abstraction.abstract_trajectory(
    rows,
    hybrid,
    times=times,
    modes=["holding"] * len(rows),
    config=cfg,
)
assert not coverage.check(wrong_mode, predicted.graph).covered
```

## Batches and scores

The reference bridge accepts a batch of trajectories and the coverage oracle
reports both the aggregate fraction and every individual witness/refutation:

```python
batch = abstraction.abstract_batch(
    [rows, rows],
    hybrid,
    times=[times, times],
    modes=[modes, modes],
    config=cfg,
)
fraction, details = coverage.score(batch, predicted.graph)
print(fraction, len(details))                # 1.0 2
```

For an actual tensor shaped `(B, T, V)`,
`qrlib.tensor.abstraction.abstract_batch_tensor` performs quantization,
direction estimation, and run detection on its current device before
materializing only the ragged run tail. That path is covered in Lesson 14.

## A coverage-failure checklist

Before concluding that the qualitative model is wrong, check:

- variable order and units;
- strictly increasing timestamps;
- landmark numeric values and tolerances;
- sampling density and multi-landmark jumps;
- exact event records and their full states;
- mode labels at boundaries;
- direction thresholds and debounce;
- whether the observation is a finite prefix.

Once those are correct, an uncovered observation is a sound refutation of the
qualitative model for that trajectory.

## Exercises

1. Remove the mode channel. Inspect how much of the hybrid distinction remains.
2. Label the first sample `holding` and read the resulting coverage diagnosis.
3. Replace the exact crossing time with an inferred crossing and compare
   `time_bounds`.

---

Next: [**12. Learning and checking structure from data →**](12-learning-and-diagnosis.md)
