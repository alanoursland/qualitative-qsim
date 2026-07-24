# Reasoning About Change Without Numbers

A hands-on tutorial for **qrlib**, a library for *qualitative reasoning* about
dynamical systems. You'll learn to describe how a system works using only
signs and orderings — no exact numbers — and have the computer work out
**every** way it can behave over time.

## Who this is for

Undergraduates who can read a little Python and have met the idea of a
*derivative* (the rate at which something changes). You do **not** need any
background in differential equations, control theory, or artificial
intelligence. Each lesson is short, runnable, and ends with exercises.

## What you'll be able to do by the end

The first nine lessons are the **core track**. By the end of it you can:

- Describe a system — a draining tank, a swinging spring — as a **model** of
  qualitative constraints.
- **Simulate** it and read the tree of possible behaviors.
- Understand the library's central promise: it never misses a real behavior
  (and why that guarantee runs one direction only).
- Add numbers back in to get **time bounds** ("it fills between t=1 and t=2").
- Reach for the analysis tools: guided simulation, fault diagnosis, causal
  ordering, and comparative "what-if" analysis.

The advanced track then shows how to use the rest of the public library:
portable models and results, hybrid numeric trajectories, structure learning,
total envisionment, alternative authoring front ends, decomposition, tensor
execution, and differentiable losses.

See the [feature map](feature-map.md) for an explicit accounting of every
public package area.

## Core track

| # | Lesson | You'll learn |
|---|--------|--------------|
| 1 | [What is qualitative reasoning?](01-what-is-qr.md) | Why sign-and-order reasoning is powerful, and the one big promise |
| 2 | [Quantities and values](02-quantities-and-values.md) | Landmarks, quantity spaces, magnitude + direction |
| 3 | [Constraints and models](03-constraints-and-models.md) | Building a model out of `M+`, `Add`, `Deriv`, … |
| 4 | [Your first simulation](04-simulating.md) | `qsim`, behaviors, terminal classes |
| 5 | [Cycles and the spring](05-cycles-and-the-spring.md) | Oscillations, point vs. interval time, timelines |
| 6 | [Taming spurious behaviors](06-spurious-behaviors.md) | Landmark discovery, chatter, and the `EnergyFilter` |
| 7 | [The soundness guarantee](07-soundness.md) | The coverage oracle — the heart of the library |
| 8 | [Adding numbers back](08-semi-quantitative.md) | Semi-quantitative refinement and time bounds |
| 9 | [Regions and a tour of the reasoning layer](09-regions-and-reasoning.md) | Piecewise systems; guided sim, diagnosis, causes, what-ifs |

## Advanced and applied track

| # | Lesson | You'll learn |
|---|--------|--------------|
| 10 | [Portable models and reproducible results](10-portable-models.md) | Compact syntax, schemas, model identity, and result provenance |
| 11 | [Trustworthy hybrid trajectory abstraction](11-hybrid-abstraction.md) | Crossing events, mode channels, batching, and honest coverage failures |
| 12 | [Learning and checking structure from data](12-learning-and-diagnosis.md) | Landmark harvest, sign estimation, induction, consistency, and diagnosis |
| 13 | [Reasoning beyond one simulation](13-advanced-analysis.md) | Total envisionment, temporal logic, monotonicity, and stronger pruning |
| 14 | [Composition, tensors, and scale](14-composition-and-scale.md) | QPT/devices, DecSIM, backend selection, interval batches, and losses |
| 15 | [End-to-end host integration](15-host-integration-capstone.md) | A complete model → simulate → abstract → check → serialize workflow |

## Running the examples

Examples use only a normal `qrlib` install unless an optional dependency is
called out. Blocks build cumulatively within a lesson; output and deliberately
incomplete sketches are marked as text rather than executable Python. From
the repository root:

```bash
pip install -e .        # install qrlib in editable mode
python -c "import qrlib; print(qrlib.__version__)"
```

Copy a lesson's code into a file (or a REPL) and run it. Where a lesson shows
a picture, that picture was produced by the library itself — regenerate them
all with:

```bash
python docs/tutorial/make_figures.py
```

The script in [`make_figures.py`](make_figures.py) is also worth reading: it
is a compact tour of the same API the lessons introduce.

## A note on honesty

Qualitative reasoning trades precision for coverage. This library is careful
about what it can and cannot promise, and so is this tutorial — Lesson 7 is
entirely about that boundary. Keep it in mind: the goal is not to predict
*the* future, but to enumerate *every possible* future soundly.
