# 14. Composition, tensors, and scale

> **In this lesson:** author models through processes and reusable devices,
> decompose independent subsystems, understand backend dispatch, screen
> interval batches, and use qualitative constraints as differentiable losses.

The same `Model` sits beneath every interface in this lesson. Front ends build
models; execution strategies consume them.

## Process-centric authoring

QPT-style authoring describes quantities, proportionalities, and direct
process influences:

```python
import qrlib as qr
from qrlib import Qdir, TimeTag
from qrlib.frontends import devices, qpt

system = qpt.System("equalization")
system.quantity("a", landmarks=("0",), upper_unbounded=True)
system.quantity("b", landmarks=("0",), upper_unbounded=True)
system.quantity("difference", landmarks=("0",), unbounded=True)
system.quantity("flow", landmarks=("0",), unbounded=True)
system.constrain(qr.Add("difference", "b", "a"))  # difference + b = a
system.proportional("flow", "difference", +1, cvals=(("0", "0"),))

transfer = system.process("transfer")
transfer.influence("a", -1, "flow")
transfer.influence("b", +1, "flow")
process_model = system.build()
print(type(process_model).__name__)           # Model
```

Conditional processes compile to operating regions and guarded transitions.
When no process influences a quantity in a region, the sole-mechanism
assumption pins its generated rate to zero.

## Reusable devices

The device front end wires reusable component types into a single ordinary
model:

```python
library = devices.Library()
library.component(
    "storage",
    terminals={"in": devices.VarSpec(("0", "MAX"))},
    internals={
        "amount": devices.VarSpec(("0", "CAP")),
        "rate": devices.VarSpec.unbounded(),
    },
    law=lambda v: [
        qr.Deriv(v("amount"), v("rate")),
        qr.MPlus(v("in"), v("rate"), cvals=(("0", "0"),)),
    ],
)

plant = devices.Device("plant", library)
plant.add("left", "storage")
plant.add("right", "storage")
device_model = plant.build()
print(tuple(device_model.variables))
```

Connected terminals become shared variables; internals remain instance
qualified. The engine has no special process or device logic after `build()`.

## Decomposition

DecSIM-style execution avoids representing every interleaving of independent
subsystems:

```python
from qrlib import decompose

independent = qr.Model("independent")
independent.variable("a", landmarks=("0",), unbounded=True)
independent.variable("b", landmarks=("0",), unbounded=True)
independent.constrain(qr.Constant("a"))
independent.constrain(qr.Constant("b"))
independent_initial = independent.state(
    time=TimeTag.POINT,
    a=("0", Qdir.STD),
    b=("0", Qdir.STD),
)

decomposed = decompose.decsim(independent, independent_initial)
print(decomposed.stats["component_nodes"])    # {'c0': 1, 'c1': 1}
print(len(decomposed.joint_behaviors()))      # 1
```

For coupled components, upstream qualitative episode words guide downstream
simulation and compatible component behaviors are joined afterward. Cyclic
coupling uses a more conservative chatter-and-join fallback and may still
truncate; the result reports that honestly.

## Backend selection

`SimConfig(backend="auto")` chooses per filtering workload:

```python
automatic = qr.qsim(independent, independent_initial)
print(automatic.stats["backend"]["requested"])       # auto
print(automatic.stats["backend"]["selection_reasons"])
```

Small or unconstrained products stay on the reference path. Sufficiently
large constrained interpretation products use tensor filtering. Explicit
`"reference"` and `"tensor"` modes are available for qualification and
debugging, but CUDA is not automatically best: transfer and Python output
costs dominate small or high-run-density workloads.

## Batched interval feasibility

The tensor interval layer screens many qualitative states against numeric
`Add`, `Minus`, `Mult`, and `At` ranges:

```python
import torch
from qrlib.tensor import interval as tensor_interval

pinned = qr.Model("pinned")
pinned.variable(
    "x",
    landmarks=(qr.Landmark("0", 0.0), qr.Landmark("SET", 2.0)),
)
pinned.constrain(qr.At("x", "SET"))
pinned_frame = pinned.compile()
pinned_states = [
    pinned.state(x=("SET", Qdir.STD)),
    pinned.state(x=("0", Qdir.STD)),
]
mask = tensor_interval.feasible_mask(pinned_frame, pinned_states)
print(mask.tolist())                           # [True, False]
```

This vectorizes within-state algebraic narrowing. Monotone envelopes and
cross-state time coupling remain in `qrlib.semiquant.refine`.

## Differentiable qualitative losses

The exact engine is Boolean. Above it, `qrlib.tensor.losses` renders
constraints as differentiable penalties over numeric trajectories:

```python
from qrlib.tensor.losses import constraint_loss

balance = qr.Model("balance")
for name in ("netflow", "outflow", "inflow"):
    balance.variable(
        name,
        landmarks=(qr.Landmark("0", 0.0),),
        unbounded=True,
    )
balance.constrain(qr.Add("netflow", "outflow", "inflow"))

good = torch.tensor(
    [[0.0, 1.0, 1.0], [0.0, 2.0, 2.0]],
    dtype=torch.float64,
    requires_grad=True,
)
loss = constraint_loss(good, balance)
loss.backward()
print(float(loss.detach()))                    # 0.0
```

Gradients can regularize a differentiable simulator or fit parameters toward
qualitative structure. They never feed back into the exact QSIM predicate
engine.

## Production scale

The measured workload coordinates are:

- variable count and total scalar volume;
- run density after qualitative segmentation;
- whether tensors already reside on the target device;
- output volume, which ultimately returns as Python objects.

Homogeneous jobs sharing one compiled frame can batch together. The observed
heterogeneous ensemble shape is small enough that independently scheduling
models is cleaner than padding them into a synthetic model axis. See
[`docs/scale-profiles.md`](../scale-profiles.md) for qualified workloads and
measured results.

## Exercises

1. Make the QPT transfer process active only while `difference > 0` and inspect
   the generated regions.
2. Force `backend="tensor"` on the independent model and compare graph exports
   with the reference result.
3. Change one row of `good` so `netflow + outflow != inflow`; verify the loss
   becomes positive and gradients are populated.

---

Next: [**15. End-to-end host integration →**](15-host-integration-capstone.md)
