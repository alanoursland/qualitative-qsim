# Signed-graph consistency

`qrlib.analysis.monotonicity` checks whether a model's pairwise monotone
relationships admit one consistent orthant ordering. It is a lightweight
structural check over the qualitative model; it does not require numeric
equations, Jacobians, or a particular array representation.

```python
import qrlib as qr
from qrlib.analysis import monotonicity

m = qr.Model("example")
for name in ("temperature", "pressure", "volume"):
    m.variable(name, unbounded=True)

m.constrain(qr.MPlus("temperature", "pressure"))
m.constrain(qr.MMinus("pressure", "volume"))
m.constrain(qr.MMinus("volume", "temperature"))

certificate = monotonicity.check_signed_graph(m)
assert certificate.is_consistent
assert certificate.polarities == {
    "temperature": 1,
    "pressure": 1,
    "volume": -1,
}
```

## Meaning of the certificate

For every `M+(x, y)`, the check requires `x` and `y` to have the same
polarity. For every `M-(x, y)` or exact `Minus(x, y)`, it requires opposite
polarities. A graph is consistent exactly when polarities
`p(variable) in {-1, +1}` exist such that

```text
p(x) * p(y) == declared_sign(x, y)
```

Equivalently, the product of signs around every cycle must be positive.
This is the classical balance characterization for signed graphs
([Harary1953](references.md#harary1953)).
Disconnected components are independent; qrlib chooses polarity `+1` for
the first declared variable in each component to make the returned
assignment deterministic. Isolated variables receive `+1`.

When no assignment exists, `conflict_cycle` gives a concrete negative-sign
cycle:

```python
m = qr.Model("conflict")
for name in ("a", "b", "c"):
    m.variable(name, unbounded=True)
m.constrain(qr.MPlus("a", "b"))
m.constrain(qr.MPlus("b", "c"))
m.constrain(qr.MMinus("c", "a"))

certificate = monotonicity.check_signed_graph(m)
assert not certificate.is_consistent
assert len(certificate.conflict_cycle) == 3
```

Each `SignedRelation` records the pair, sign, constraint kind, and original
constraint index. `MonotonicityCertificate.to_dict()` produces a
JSON-compatible diagnostic.

## Operating regions

Pass `region="name"` to check only constraints active in one operating
region. With no region, qrlib checks the union of all model constraints.
Consequently, opposite relationships used in mutually exclusive regions
conflict in the whole-model result even when each region is individually
consistent. This is intentional: there is no single whole-model orthant in
that case.

The authored `Model` and `CompiledModel` forms have the same behavior,
including the implicit `"default"` region.

## Scope

The check includes only `M+`, `M-`, and `Minus`. It does not infer
unconditional pairwise signs from `Add` or `Mult`, because those signs
depend on operand values and causal interpretation. Derivative constraints
also do not assert the Jacobian signs required by monotone-systems theory.

Therefore a successful certificate means:

> The declared pairwise qualitative relationships are compatible with one
> orthant ordering.

It does **not** by itself prove that an external ODE or hybrid vector field
is a monotone dynamical system. A host can use the returned polarities as a
candidate order and separately verify its numeric vector field.
