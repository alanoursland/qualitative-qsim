# 1. What is qualitative reasoning?

> **In this lesson:** why you can predict a system's behavior without knowing
> its numbers, and the single promise that makes this library trustworthy.

## The bathtub you can reason about in your head

Picture a bathtub. The tap is running at a steady rate, and the drain is
open. Will the water overflow, settle at some level, or drain away?

You already know the answer depends on a race: *how fast water comes in*
versus *how fast it drains out*. And you know one more thing without being
told a single number — **the drain gets faster as the water gets deeper**.
So as the tub fills, the outflow climbs to meet the inflow, and the level
either steadies out (if the drain can keep up before the tub is full) or the
tub overflows (if it can't).

Notice what you just did. You reasoned about:

- **directions** — the water level is *rising*; the outflow is *increasing*;
- **orderings** — is the inflow *more than*, *equal to*, or *less than* the
  maximum the drain can manage?

You never used a number. You used *signs* (rising / falling / steady) and
*order* (bigger / smaller / equal). That is **qualitative reasoning**, and
this library automates it.

## Why bother, when we have real simulators?

Ordinary (quantitative) simulation needs exact numbers: the tap rate, the
drain coefficient, the tub's shape. Give it those, and it predicts *one*
future — the trajectory for *those* numbers.

Qualitative reasoning answers a different, often more useful question:

> **Given only how the parts relate, what are _all_ the qualitatively
> distinct things that can happen?**

That's valuable when you don't have exact numbers, when you want to be sure
you haven't missed a failure mode, or when you want to *explain* why a system
behaves as it does rather than just watch it. For the bathtub, the answer is
a set of three outcomes — settle below the brim, settle exactly at the brim,
or overflow — and the library derives all three from the structure alone.

## The one promise (and its one catch)

Here is the guarantee that makes the whole approach trustworthy, stated
plainly:

> **Every behavior the real system can actually exhibit will appear in the
> library's output.**

This is called *guaranteed coverage*. If you build an honest model, the true
behavior is guaranteed to be somewhere in the tree the library gives you. You
will never be blindsided by a behavior it failed to predict.

The catch — and it is important — runs the other way:

> **The output may also contain _extra_ behaviors that no real system
> exhibits.** These are called *spurious* behaviors.

So the library is like a careful witness who promises to name everyone who
*could* have been at the scene, knowing the list might include a few
innocents. It never misses the culprit; it may over-include. Lessons 6 and 7
are about living with this honestly — pruning spurious behaviors where we
soundly can, and never claiming more than the guarantee allows.

This asymmetry has a precise consequence you'll use throughout:

- A claim about **all** behaviors ("the tub *never* runs dry") can be
  **proven** — if it holds for every behavior in the (over-complete) output,
  it certainly holds for the real ones.
- A claim about **some** behavior ("there *exists* a way for it to overflow")
  is only **suggested** — the witnessing behavior might be spurious.

## What a "behavior" looks like

Before the next lesson, here's the shape of what you're working toward. The
library turns a model into a **behavior graph**: a tree (sometimes with
loops) whose paths are the possible behaviors. Here is the real one for the
bathtub, produced by the library:

![The bathtub's three behaviors](figures/bathtub-tree.svg)

Read it top to bottom: the system starts at the root, moves to an
intermediate state, then splits into three endings — two `quiescent`
(settled) equilibria and one `region_exit` (the overflow, leaving the
model's valid range). You'll learn to build this graph yourself in Lesson 4.

## Exercises

1. Think of another everyday system (a cup of coffee cooling, a savings
   account earning interest, a car braking to a stop). List the **directions**
   and **orderings** you'd reason about to predict its behavior — without
   numbers.
2. For the bathtub, which of these is a *universal* claim (provable) and which
   is *existential* (only suggested)?
   (a) "The water level never decreases."
   (b) "There is a scenario where the tub overflows."
3. In one sentence, explain to a friend why a tool that sometimes reports
   *extra* behaviors can still be *more* trustworthy than one that reports a
   single exact trajectory.

---

Next: [**2. Quantities and values →**](02-quantities-and-values.md)
