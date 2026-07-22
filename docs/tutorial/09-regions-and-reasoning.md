# 9. Regions and a tour of the reasoning layer

> **In this lesson:** systems that switch behavior at a threshold, and a quick
> tour of the analysis tools built on top of the simulator. This is your map
> to the rest of the library.

## Part A — Regions: piecewise systems

Real systems change their rules at thresholds. A tank *overflows* once it's
full — beyond the brim, the physics is different (excess spills; the level
holds at the top). The library models this with **operating regions**: named
subsets of constraints that are active in different parts of state space, with
guarded **transitions** between them.

Here's the bathtub upgraded so that reaching the brim *while still filling*
crosses into a steady-overflow region:

```python
import qrlib as qr
from qrlib import Qdir, TimeTag

m = qr.Model("overflow bathtub")
m.variable("amount",  landmarks=("0", "FULL"))
m.variable("level",   landmarks=("0", "TOP"))
m.variable("outflow", landmarks=("0", "OMAX"))
m.variable("inflow",  landmarks=("0", "IF"))
m.variable("netflow", landmarks=("0",), unbounded=True)

c_al  = m.constrain(qr.MPlus("amount", "level",  cvals=(("0", "0"), ("FULL", "TOP"))))
c_lo  = m.constrain(qr.MPlus("level",  "outflow", cvals=(("0", "0"), ("TOP", "OMAX"))))
c_add = m.constrain(qr.Add("netflow", "outflow", "inflow"))
c_der = m.constrain(qr.Deriv("amount", "netflow"))
c_inf = m.constrain(qr.Constant("inflow"))
c_amt = m.constrain(qr.Constant("amount"))          # amount held while overflowing

# two regions: while filling, and while overflowing
m.region("filling",     constraints=(c_al, c_lo, c_add, c_der, c_inf))
m.region("overflowing", constraints=(c_al, c_lo, c_add, c_inf, c_amt))

# cross when the tub is full *and* still gaining water
m.transition("filling", "overflowing",
             when=(("amount", "==", "FULL"), ("netflow", ">", "0")))

initial = m.state(time=TimeTag.POINT,
                  amount=("0", Qdir.INC), level=("0", Qdir.INC),
                  outflow=("0", Qdir.INC), inflow=("IF", Qdir.STD),
                  netflow=(("0", "+inf"), Qdir.DEC))

result = qr.qsim(m, initial)
print(result.stats["region_crossings"])   # 1
```

![Behavior graph with a region crossing](figures/regions-tree.svg)

One of the behaviors now crosses regions — you can see it in the results:

```python
for b in result.behaviors():
    print(b.terminal.value, b.regions)
```

```
quiescent   ('filling', 'filling', 'filling')                  # settles below the brim
quiescent   ('filling', 'filling', 'filling', 'overflowing')   # fills, then overflows steadily
quiescent   ('filling', 'filling', 'filling')                  # settles exactly at the brim
```

The middle behavior fills, hits `FULL` while still gaining, **crosses** into
the overflow region, and there reaches a new steady state (spilling the excess
at the brim). At the crossing, magnitudes carry over continuously and only the
*directions* re-derive under the new region's rules — that's how the tub goes
from "rising" to "held at the top."

## Part B — A tour of the reasoning layer

Everything so far produces or refines a behavior graph. The library also
*reasons about* those graphs. Here are the main tools, each a few lines, with
pointers to go deeper. Use the plain (non-region) bathtub `m`/`initial` from
Lessons 3–4 for these.

### What causes what — `analysis.causal`

Derives the causal structure from the constraints alone (no simulation):

```python
from qrlib.analysis import causal
order = causal.causal_order(m)
print(order.exogenous)        # ('inflow',)   — the driving input
print(order.state_variables)  # ('amount',)   — the system's memory
print(causal.narrate_causes(order))
```

### What-if questions — `analysis.compare`

*Comparative analysis*: perturb a parameter and get the sign of change of
every quantity at equilibrium.

```python
from qrlib.analysis import compare
change = compare.compare(m, {"inflow": +1})   # turn the tap up
print({k: v.symbol for k, v in change.changes.items()})
# {'amount': '↑', 'level': '↑', 'outflow': '↑', 'inflow': '↑', 'netflow': '·'}
```

Read it: raise the inflow and the equilibrium amount, level, and outflow all
rise, while the net flow stays zero (it's an equilibrium). Derived purely by
sign propagation — no numbers.

### Focusing on behaviors of interest — `qrlib.guide`

*Guided simulation*: state a temporal-logic property and keep only behaviors
consistent with it — and get a sound verdict.

```python
from qrlib import guide
from qrlib.guide import G, mag

# "the amount stays below FULL forever" (a safety property)
g = guide.guided(m, initial, G(mag("amount", "<", "FULL")))
print(len(g.satisfied), g.universal)   # 1  True
```

`universal=True` is a *sound proof* (recall Lesson 7): among all behaviors
that keep the amount below full, the property holds — and this is the class of
behaviors where the drain keeps up.

### Which part is broken — `qrlib.diagnosis`

*Model-based diagnosis*: give components fault modes and observations, and get
the minimal explanations. In outline:

```python
from qrlib import diagnosis
drain = diagnosis.Component("drain", modes={
    "ok":    (qr.MPlus("level", "outflow", cvals=(("0", "0"), ("TOP", "OMAX"))),),
    "stuck": (qr.At("outflow", "0"), qr.Constant("outflow")),   # drain jammed shut
})
# diagnosis.diagnose(base_model, [drain], observations)  -> the consistent fault sets
```

Given a trajectory where the tub overflows even though the tap is modest,
diagnosis reports that the `drain` being `stuck` is the minimal explanation —
it uses the *coverage oracle* from Lesson 7 as its consistency check.

## Where to go next

You now have the whole mental model. The remaining modules extend it; each has
a thorough module docstring:

| Want to… | Reach for |
|---|---|
| Author models as **processes** or by **wiring components** | `qrlib.frontends.qpt`, `qrlib.frontends.devices` |
| **Learn a model** from trajectory data | `qrlib.induce` |
| Scale to **larger systems** by decomposition | `qrlib.decompose` |
| Enumerate **every** state, not just reachable ones | `qrlib.envision` |
| Use a model as a **training signal** (gradients) | `qrlib.tensor.losses` |
| Prune spurious oscillations by **phase geometry** | `SimConfig(phase_pairs=...)` |
| Order-of-magnitude reasoning (`x ≪ y`) | `qrlib.Negligible` |

For the design behind it all, see the top-level `docs/`: `architecture.md`
(how the pieces fit), `qsim.md` (the engine), `host-integration.md` (using the
library from a larger system), and `literature-survey.md` (where these ideas
come from).

## Exercises

1. In the region model, change the crossing guard to fire on `amount == FULL`
   *alone* (drop the `netflow > 0` condition). Does the behavior that settles
   *exactly at the brim* now wrongly cross into overflow? Why is the extra
   guard important?
2. Use `compare.compare` to predict what happens to the equilibrium level if
   you made the drain more efficient (imagine an `outflow` that rises *faster*
   with level). Sketch how you'd encode that perturbation.
3. Pick one module from the "where to go next" table, open its docstring, and
   run its first example. Write two sentences on what new question it lets you
   answer that the plain simulator could not.

---

**That's the tutorial.** You can now describe a system qualitatively, simulate
every behavior it admits, tame spurious clutter soundly, add numbers for
tighter bounds, and reason about causes, faults, and what-ifs — all on the
firm footing of guaranteed coverage. Welcome to qualitative reasoning.

← [Back to the index](README.md)
