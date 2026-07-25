# 10. Portable models and reproducible results

> **In this lesson:** use compact constraint syntax, serialize models, compare
> stable model identities, and inspect exactly what a simulation result says
> about replayability.

The core lessons built models directly from Python objects. That is ideal
inside an application, but hosts also need models that can cross process,
language, and storage boundaries.

## Compact authoring is only syntax

`Model.constrain` accepts built-in constraint objects or safe compact strings:

```python
import json
import qrlib as qr
from qrlib import Landmark, Qdir, TimeTag

m = qr.Model("portable constant")
m.variable(
    "x",
    landmarks=(Landmark("0", 0.0), Landmark("HIGH", 1.0)),
)
m.constrain("Constant(x)")
m.constrain('At(x, "0")')

round_trip = qr.parse_constraint("Negligible(error, signal)")
assert qr.format_constraint(round_trip) == "Negligible(error, signal)"
```

The parser accepts only known constraint calls and literal arguments. It does
not use `eval`, import names, or preserve strings in the model. The stored
constraints are the same frozen objects you would have constructed directly.

## The model schema

`Model.to_dict()` returns the versioned `qrlib.model/v1` representation:

```python
model_data = m.to_dict()
print(model_data["schema"])                 # qrlib.model/v1

encoded = json.dumps(model_data)
rebuilt = qr.Model.from_dict(json.loads(encoded))
assert rebuilt.to_dict() == model_data
```

The schema is the interchange format. Compact constraint strings are an
authoring convenience, not a second schema.

## Stable semantic identity

Models have a canonical SHA-256 identity:

```python
print(m.content_hash())                     # sha256:...
assert rebuilt.content_hash() == m.content_hash()
assert rebuilt.compile().model_hash == m.content_hash()
```

The hash is based on semantic model content rather than incidental dictionary
or authoring order. Changing a quantity space, constraint, region, guard, or
model name changes the identity.

## Result provenance

Run the model and inspect its plain-data result:

```python
initial = m.state(time=TimeTag.POINT, x=("0", Qdir.STD))
result = qr.qsim(m, initial)
result_data = result.to_dict()

print(result_data["schema"])                # qrlib.result/v3
assert result_data["model_hash"] == m.content_hash()
assert result_data["config"]["backend"] == "auto"
assert result_data["config"]["profile"] == "practical"
json.dumps(result_data)                     # entirely plain JSON data
```

The result records the input model identity even if a classic or custom
configuration later grows per-branch quantity spaces through landmark
discovery. It also records the configuration and terminal/statistical
information needed to interpret the graph.

Built-in successor filters describe themselves in replayable form:

```python
filtered = qr.qsim(
    m,
    initial,
    config=qr.SimConfig(successor_filters=(qr.EnergyFilter(("x",)),)),
)
descriptor = filtered.to_dict()["config"]["successor_filters"][0]
print(descriptor["kind"], descriptor["replayable"])  # energy True
```

An arbitrary Python callable cannot be reconstructed safely. Its descriptor
records its module and qualified name with `replayable=False`; it does not
pretend that a name is executable provenance.

## What to persist

For a reproducible record, persist:

1. the model's `to_dict()` payload;
2. the result's `to_dict()` payload;
3. the numeric observations and their variable order, if coverage was used;
4. the environment/package versions used for execution;
5. any opaque user filter source or separately versioned configuration.

The model hash connects the model and result, but it is an identity—not a
substitute for storing the model itself.

## Exercises

1. Reorder the calls that add constraints and confirm the rebuilt semantic
   model receives the same content hash.
2. Change the value of `HIGH` and confirm the hash changes.
3. Add a small named Python filter, export the result, and inspect its opaque
   descriptor.

---

Next: [**11. Trustworthy hybrid trajectory abstraction →**](11-hybrid-abstraction.md)
