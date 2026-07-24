# Compact constraint syntax

qrlib models may be authored with constraint objects or optional compact
strings:

```python
m.constrain(qr.MPlus("amount", "level", cvals=(("0", "0"),)))
m.constrain('M+(amount, level, cvals=[("0", "0")])')
```

These statements are equivalent. `Model.constrain` immediately parses a
string into the existing frozen constraint dataclass, applies the same model
validation, and stores only that object. Model serialization remains
`qrlib.model/v1`; the string form is authoring sugar, not another interchange
schema.

## Grammar

```text
constraint := KIND "(" argument ("," argument)* ["," cvals] ")"
cvals      := "cvals" "=" ("[" | "(") landmark_tuple* ("]" | ")")
argument   := bare_name | dotted_name | quoted_string
```

Supported kinds and positional arguments:

| Syntax | Object |
|---|---|
| `M+(x, y)` or `MPlus(x, y)` | `MPlus(x, y)` |
| `M-(x, y)` or `MMinus(x, y)` | `MMinus(x, y)` |
| `Add(x, y, z)` | `Add(x, y, z)` |
| `Mult(x, y, z)` | `Mult(x, y, z)` |
| `Minus(x, y)` | `Minus(x, y)` |
| `Deriv(x, rate)` | `Deriv(x, rate)` |
| `Constant(x)` | `Constant(x)` |
| `At(x, landmark)` | `At(x, landmark)` |
| `Negligible(small, large)` | `Negligible(small, large)` |

Kind names other than `M+` and `M-` are case-insensitive. `cvals` is accepted
only by `M+`, `M-`, `Add`, `Mult`, and `Minus`, and every landmark tuple must
match the constraint's positional arity:

```python
m.constrain(
    'M+(amount, level, '
    'cvals=[("0", "0"), (FULL, TOP)])'
)
m.constrain('Add(netflow, outflow, inflow, cvals=[("0", "0", "0")])')
```

Ordinary identifiers and dotted names may be bare. Quote names containing
punctuation, whitespace, Python keywords, numeric landmark names, or infinity
markers:

```python
m.constrain('Deriv(A.amount, "amount-rate")')
m.constrain('At(amount, "0")')
m.constrain('M+("input flow", output)')
```

Numeric literals are accepted in landmark positions and converted to their
string spelling, so `At(x, 0)` is equivalent to `At(x, "0")`.

## Public helpers

- `parse_constraint(source)` returns a built-in `Constraint`.
- `format_constraint(constraint)` returns canonical compact syntax.
- `ConstraintSyntaxError` is a `ValueError` subclass for syntax, kind, arity,
  keyword, and literal-shape failures.

Formatting and parsing round-trip every built-in constraint:

```python
constraint = qr.MPlus("x", "y", cvals=(("0", "0"),))
assert qr.parse_constraint(qr.format_constraint(constraint)) == constraint
```

## Safety boundary

The parser never calls `eval` and never imports or resolves names. It accepts
only a direct recognized constraint call whose arguments are names, dotted
names, quoted strings, numeric landmark literals, and literal list/tuple
structures for `cvals`. Function calls, attribute calls such as
`qr.Add(...)`, comprehensions, operators, star expansion, and arbitrary
keywords are rejected (except a unary sign on a numeric landmark literal).
