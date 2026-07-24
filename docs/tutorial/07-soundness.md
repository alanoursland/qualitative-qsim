# 7. The soundness guarantee

> **In this lesson:** the promise at the heart of the library — *every real
> behavior appears in the graph* — and the tool that lets you check it: the
> **coverage oracle**. This is the most important lesson in the tutorial.

## What "sound" actually means

We keep saying the library's output has **guaranteed coverage**: every
behavior the real system can exhibit shows up (abstracted) somewhere in the
predicted graph. It's worth making that concrete, because it's what separates
this from guessing.

Take the bathtub. Its model stands for a whole *family* of real bathtubs —
any tub whose level rises with amount, whose drain speeds up with depth, and
so on, with *any* actual numbers. The guarantee says: **pick any specific tub
from that family, run it for real, and its trajectory will match one of the
paths in the qualitative graph.** No real tub in the family can surprise you.

We can test this directly. Here's the recipe:

1. Pick concrete numbers (a specific tub) consistent with the model.
2. Simulate it numerically — an ordinary, quantitative trajectory.
3. **Abstract** that trajectory into a qualitative behavior.
4. **Check** that it's covered by the qualitative graph.

## Abstraction: from a trajectory to a qualitative behavior

The bridge from numbers back to qualitative land is `abstract_trajectory`. It
takes a table of numeric samples (one row per time step, one column per
variable) and quantizes them against the model's landmark values, producing
the same kind of point/interval behavior `qsim` produces.

```python
import qrlib as qr
from qrlib import Landmark, Qdir, TimeTag
from qrlib.bridge import abstraction, coverage
from qrlib.bridge.abstraction import AbstractionConfig

# a bathtub whose landmarks carry real numbers (a concrete family member)
m = qr.Model("bathtub")
m.variable("amount",  landmarks=(Landmark("0", 0.0), Landmark("FULL", 10.0)))
m.variable("level",   landmarks=(Landmark("0", 0.0), Landmark("TOP", 5.0)))
m.variable("outflow", landmarks=(Landmark("0", 0.0), Landmark("OMAX", 3.0)))
m.variable("inflow",  landmarks=(Landmark("0", 0.0), Landmark("IF", 2.0)))
m.variable("netflow", landmarks=(Landmark("0", 0.0),), unbounded=True)
m.constrain(qr.MPlus("amount", "level",  cvals=(("0", "0"), ("FULL", "TOP"))))
m.constrain(qr.MPlus("level",  "outflow", cvals=(("0", "0"), ("TOP", "OMAX"))))
m.constrain(qr.Add("netflow", "outflow", "inflow"))
m.constrain(qr.Deriv("amount", "netflow"))
m.constrain(qr.Constant("inflow"))
initial = m.state(time=TimeTag.POINT,
                  amount=("0", Qdir.INC), level=("0", Qdir.INC),
                  outflow=("0", Qdir.INC), inflow=("IF", Qdir.STD),
                  netflow=(("0", "+inf"), Qdir.DEC))

graph = qr.qsim(m, initial).graph

# a concrete tub: level = 0.5·amount, outflow = 0.6·level, inflow = 2.
# Its drain can keep up before the brim, so it settles below FULL.
rows, dt, a = [], 0.05, 0.0
for _ in range(400):
    level = 0.5 * a
    outflow = 0.6 * level
    inflow = 2.0
    net = inflow - outflow
    rows.append([a, level, outflow, inflow, net])
    a += dt * net           # simple Euler step

cfg = AbstractionConfig(landmark_atol=1e-9, direction_eps=1e-4, debounce=3)
observed = abstraction.abstract_trajectory(rows, m, config=cfg)
```

## The coverage oracle

Now the payoff — ask whether that real trajectory is a path in the
qualitative graph:

```python
result = coverage.check(observed, graph)
print(result.covered)        # True
print(result.matched, "/", result.total)   # 2 / 2  — every observed state matched
print(result.witness)        # the node-id path it matched along
```

`covered = True` means this specific tub behaved exactly as one of the
qualitative behaviors predicted. Try it with *any* consistent set of numbers —
a faster drain, a slower tap — and it will still be covered. That's the
guarantee, made checkable. (The library's own test suite runs this over
hundreds of randomized instances; see `tests/test_soundness.py`.)

If a trajectory is **not** covered, the oracle tells you where the abstracted
observation diverged:

```python
if not result.covered:
    print(result.divergence_index)   # first observed state that couldn't be matched
    print(result.diagnosis)          # a localized explanation
```

Interpreting that refutation requires checking the abstraction boundary.
Failure can mean the numeric dynamics violate the qualitative model, but it
can also mean the samples were too sparse, the variable order or mode channel
was wrong, measurement noise crossed a tolerance, or an important landmark
crossing was only poorly bracketed. The result carries its
`AbstractionConfig`, spans, and time bounds so those assumptions are
inspectable. Event-aware solvers can supply exact `CrossingEvent` records;
Lesson 11 develops that production path.

## The catch, precisely

Coverage runs **one direction only**. The oracle confirms that a real
trajectory *is* among the predicted behaviors. It does **not** confirm that
every predicted behavior is *real* — some may be spurious (Lesson 6). This is
the asymmetry from Lesson 1, now with a name:

- **You can refute, conditional on the abstraction.** If a correctly
  abstracted trajectory is not covered, the qualitative model cannot produce
  it.
- **You cannot certify existence.** A behavior appearing in the graph is
  *possible*, not *proven* — it might be spurious.

So universal statements over the graph ("in *all* behaviors, the level never
goes negative") are **provable**; existential ones ("there *exists* a behavior
that overflows") are **suggested**. Every analysis in the next lesson — and
in the whole library — respects this line. It is the difference between a tool
you can trust and one you merely hope about.

## Exercises

1. Change the concrete tub so its drain is *weaker* (say `outflow = 0.1·level`)
   and re-run. It should overflow; confirm the abstracted trajectory is still
   `covered` (now matching the overflow behavior).
2. Deliberately break the numbers — make `inflow` *decrease* over time even
   though the model declares it `Constant`. Run the oracle and read
   `divergence_index` and `diagnosis`. What does it tell you?
3. In your own words: why is "I could not find a covering behavior" a *sound*
   conclusion, while "I found a covering behavior" is only a *suggestive* one?

---

Next: [**8. Adding numbers back →**](08-semi-quantitative.md)
