# 12. Learning and checking structure from data

> **In this lesson:** turn observations into candidate landmarks and signed
> structure, check declared constraints against data, induce a QDE, and use
> the same consistency machinery for diagnosis.

These tools do not turn finite data into certain physics. They produce
candidate structure, calibrated stability information, and sound
refutations. That distinction is the theme of the lesson.

## Harvesting landmarks

A host may already know a meaningful operating point:

```python
import qrlib as qr
from qrlib import Landmark, Qdir
from qrlib.bridge import harvest, signs
from qrlib.bridge.abstraction import AbstractionConfig, abstract_trajectory

landmark_model = qr.Model("measured quantity")
landmark_model.variable(
    "x",
    landmarks=(Landmark("0", 0.0), Landmark("TOP", 10.0)),
    upper_unbounded=True,
)

grown = harvest.harvest_into_model(
    landmark_model,
    [harvest.LandmarkRecord("x", "EQUILIBRIUM", 4.0)],
)
print(grown.variables["x"].space.names)       # ('0', 'EQUILIBRIUM', 'TOP')
```

The original model is unchanged. Conflicting names, duplicate numeric values,
and values outside a bounded quantity space are reported rather than silently
merged.

Steady stretches can propose candidates:

```python
plateau = [[min(3.0, 0.1 * i)] for i in range(120)]
proposals = harvest.propose_landmarks(
    plateau,
    landmark_model,
    config=AbstractionConfig(direction_eps=1e-3),
    min_dwell=10,
)
print(proposals[0].variable, round(proposals[0].value, 2))  # x 3.0
```

A proposal is evidence for review, not an automatic declaration that every
model instance contains that landmark.

## Estimating signed influence

Suppose samples suggest `dx/dt = -x`. Bootstrap sign agreement reports how
stable the fitted sign is under resampling:

```python
xs = [[i / 20] for i in range(1, 41)]
dxs = [[-row[0]] for row in xs]

estimate = signs.estimate_signs_calibrated(
    xs,
    dxs,
    resamples=20,
    seed=7,
)
matrix = estimate.threshold(0.9)
print(matrix, estimate.confidence)            # [[-1]] ((1.0,),)
```

Confidence is stability under the observed sample distribution—not the
probability that the physical law is true. Weak, unstable, and fitted-zero
effects become `UNKNOWN` when thresholded. Fitted-zero classification uses
both an absolute coefficient floor and a relative contribution floor, so
deterministic floating-point residue does not receive perfect bootstrap
confidence merely because every resample reproduces it.

The matrix can seed a QDE:

```python
candidate_model = signs.model_from_signs(["x"], matrix)
print(candidate_model.sign_structure().monotone)
```

`UNKNOWN` entries remain unconstrained. The conversion never invents a
confident zero or a corresponding landmark value that the sign matrix did not
contain. An all-zero row means the rate has no state dependence; it compiles
to `Constant(d_x)` plus `Deriv(x, d_x)`, allowing `x` to move at a nonzero
constant rate.

## Checking a declared relation

The downward consistency checker localizes which constraints disagree with
numeric rows:

```python
relation = qr.Model("relation")
relation.variable("x", landmarks=(Landmark("0", 0.0),), unbounded=True)
relation.variable("y", landmarks=(Landmark("0", 0.0),), unbounded=True)
relation.constrain(qr.MPlus("x", "y"))

consistent_rows = [[x, 2.0 * x] for x in range(20)]
records = signs.check_consistency(consistent_rows, relation)
print(records[0].kind, records[0].violation)   # mplus 0.0

corrupted_rows = [[x, -2.0 * x] for x in range(20)]
broken = signs.check_consistency(corrupted_rows, relation)
assert broken[0].violation > 0.9
```

This checks the supplied data against the relation. It does not prove that
the relation holds outside the observed domain.

## Inducing a ranked QDE candidate

`qrlib.induce` fits signed rate influences, evaluates the resulting QDEs with
the consistency checker, and ranks consistent candidates by parsimony:

```python
from qrlib import induce

decay = []
x = 1.0
for _ in range(100):
    decay.append([x])
    x *= 0.95

induced = induce.induce(decay, ["x"])
print(induced.best.influences)                # (('x', 'x', -1),)
print(induced.best.consistent)                # True
```

The result is a ranked set, not a claim of unique identification. A candidate
can be consistent because the observations do not distinguish it from another
model.

## Diagnosis is structured consistency search

A component defines a normal mode and one or more fault modes. Diagnosis
tries mode combinations in increasing fault cardinality and uses
simulate-and-cover as the consistency test:

```python
from qrlib import diagnosis

sensor_model = qr.Model("sensor")
sensor_model.variable(
    "reading",
    landmarks=(Landmark("0", 0.0), Landmark("HIGH", 1.0)),
)
sensor = diagnosis.Component("sensor", modes={
    "ok": (qr.Constant("reading"), qr.At("reading", "HIGH")),
    "stuck_low": (qr.Constant("reading"), qr.At("reading", "0")),
})

observed = abstract_trajectory([[0.0]] * 20, sensor_model)
observed_initial = sensor_model.state(reading=("0", Qdir.STD))
answer = diagnosis.diagnose(
    sensor_model,
    [sensor],
    observed,
    initial=observed_initial,
)

print(answer.fault_detected)                  # True
print(dict(answer.diagnoses[0].modes))        # {'sensor': 'stuck_low'}
```

The normal mode is refuted; the fault mode is consistent with the observation.
“Consistent” still does not prove the sensor really failed that way. It means
the candidate survived every supplied observation and model constraint.

## Exercises

1. Add noise to the decay derivative samples and increase the confidence
   threshold until the self-influence becomes `UNKNOWN`.
2. Give the sensor a second observation at `HIGH`. Explain why no single
   stuck mode can cover both observations.
3. Add a second variable to the decay data and verify that induction does not
   invent a cross-influence when the variables evolve independently.

---

Next: [**13. Reasoning beyond one simulation →**](13-advanced-analysis.md)
