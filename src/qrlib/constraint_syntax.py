"""Safe compact syntax for authoring qrlib constraints.

The syntax is deliberately a thin spelling of existing constraint objects:

``M+(amount, level, cvals=[("0", "0"), (FULL, TOP)])``

It is parsed as a restricted expression tree and never evaluated. Models
continue to store and serialize ordinary :mod:`qrlib.constraints` objects.
"""

from __future__ import annotations

import ast
import json
import keyword
import re

from .constraints import (
    Add,
    At,
    Constant,
    Constraint,
    Deriv,
    Minus,
    MMinus,
    MPlus,
    Mult,
    Negligible,
)

__all__ = [
    "ConstraintSyntaxError",
    "parse_constraint",
    "format_constraint",
]


class ConstraintSyntaxError(ValueError):
    """A malformed or unsupported compact constraint expression."""


_SPECS = {
    "mplus": (MPlus, 2, True),
    "mminus": (MMinus, 2, True),
    "add": (Add, 3, True),
    "mult": (Mult, 3, True),
    "minus": (Minus, 2, True),
    "deriv": (Deriv, 2, False),
    "constant": (Constant, 1, False),
    "at": (At, 2, False),
    "negligible": (Negligible, 2, False),
}
_DISPLAY = {
    MPlus: "M+",
    MMinus: "M-",
    Add: "Add",
    Mult: "Mult",
    Minus: "Minus",
    Deriv: "Deriv",
    Constant: "Constant",
    At: "At",
    Negligible: "Negligible",
}
_HEAD_ALIAS = re.compile(r"^\s*(M[+-])\s*(?=\()")


def _normalize_head(source: str) -> str:
    match = _HEAD_ALIAS.match(source)
    if match is None:
        return source
    replacement = "MPlus" if match.group(1) == "M+" else "MMinus"
    return source[: match.start(1)] + replacement + source[match.end(1) :]


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix is not None else None
    return None


def _atom(node: ast.AST, *, numeric: bool = False) -> str:
    dotted = _dotted_name(node)
    if dotted is not None:
        return dotted
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if numeric and isinstance(node.value, (int, float)) and not isinstance(
            node.value, bool
        ):
            return str(node.value)
    if (
        numeric
        and isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        value = node.operand.value
        return str(-value if isinstance(node.op, ast.USub) else value)
    expected = "a name, dotted name, or quoted string"
    if numeric:
        expected += " (numeric landmark literals are also allowed)"
    raise ConstraintSyntaxError(
        f"constraint arguments must be {expected}; got "
        f"{ast.dump(node, include_attributes=False)}"
    )


def _corresponding_values(node: ast.AST, arity: int):
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise ConstraintSyntaxError(
            "cvals must be a list or tuple of landmark tuples"
        )
    result = []
    for index, item in enumerate(node.elts):
        if not isinstance(item, (ast.List, ast.Tuple)):
            raise ConstraintSyntaxError(
                f"cvals[{index}] must be a list or tuple"
            )
        values = tuple(_atom(value, numeric=True) for value in item.elts)
        if len(values) != arity:
            raise ConstraintSyntaxError(
                f"cvals[{index}] has arity {len(values)}, expected {arity}"
            )
        result.append(values)
    return tuple(result)


def parse_constraint(source: str) -> Constraint:
    """Parse one compact constraint expression into an existing constraint.

    Accepted kinds are ``M+``/``MPlus``, ``M-``/``MMinus``, ``Add``,
    ``Mult``, ``Minus``, ``Deriv``, ``Constant``, ``At``, and
    ``Negligible`` (kind names other than ``M+``/``M-`` are
    case-insensitive). Corresponding values are an optional ``cvals`` keyword
    on monotone/algebraic constraints.
    """
    if not isinstance(source, str):
        raise TypeError("constraint source must be a string")
    if not source.strip():
        raise ConstraintSyntaxError("constraint source is empty")
    normalized = _normalize_head(source)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        location = f" at column {exc.offset}" if exc.offset is not None else ""
        raise ConstraintSyntaxError(
            f"invalid constraint syntax{location}: {exc.msg}"
        ) from None
    if not isinstance(tree.body, ast.Call) or not isinstance(
        tree.body.func, ast.Name
    ):
        raise ConstraintSyntaxError(
            "constraint must be a direct call such as M+(x, y)"
        )
    kind = tree.body.func.id.lower()
    if kind not in _SPECS:
        names = "M+, M-, Add, Mult, Minus, Deriv, Constant, At, Negligible"
        raise ConstraintSyntaxError(
            f"unknown constraint kind {tree.body.func.id!r}; expected one of {names}"
        )
    cls, arity, supports_cvals = _SPECS[kind]
    if len(tree.body.args) != arity:
        raise ConstraintSyntaxError(
            f"{_DISPLAY[cls]} expects {arity} positional arguments, "
            f"got {len(tree.body.args)}"
        )
    args = []
    for index, node in enumerate(tree.body.args):
        numeric = cls is At and index == 1
        args.append(_atom(node, numeric=numeric))

    kwargs = {}
    seen = set()
    for item in tree.body.keywords:
        if item.arg is None:
            raise ConstraintSyntaxError("**keyword expansion is not allowed")
        if item.arg in seen:
            raise ConstraintSyntaxError(f"duplicate keyword {item.arg!r}")
        seen.add(item.arg)
        if item.arg != "cvals":
            raise ConstraintSyntaxError(f"unknown keyword {item.arg!r}")
        if not supports_cvals:
            raise ConstraintSyntaxError(
                f"{_DISPLAY[cls]} does not accept corresponding values"
            )
        kwargs["cvals"] = _corresponding_values(item.value, arity)
    return cls(*args, **kwargs)


def _format_atom(value: str) -> str:
    parts = value.split(".")
    if all(part.isidentifier() and not keyword.iskeyword(part) for part in parts):
        return value
    return json.dumps(value, ensure_ascii=False)


def format_constraint(constraint: Constraint) -> str:
    """Return the canonical compact syntax for a built-in constraint."""
    cls = type(constraint)
    if cls not in _DISPLAY:
        raise TypeError(
            f"unsupported constraint type {cls.__module__}.{cls.__qualname__}"
        )
    if isinstance(constraint, At):
        args = (constraint.x, constraint.landmark)
    else:
        args = constraint.variables
    parts = [_format_atom(value) for value in args]
    if constraint.corresponding_values and not isinstance(constraint, At):
        rows = []
        for row in constraint.corresponding_values:
            values = ", ".join(_format_atom(value) for value in row)
            if len(row) == 1:
                values += ","
            rows.append(f"({values})")
        parts.append(f"cvals=[{', '.join(rows)}]")
    return f"{_DISPLAY[cls]}({', '.join(parts)})"
