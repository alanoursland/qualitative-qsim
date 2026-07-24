"""Compact constraint authoring syntax."""

import pytest

import qrlib as qr


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            'M+(amount, level, cvals=[("0", "0"), (FULL, TOP)])',
            qr.MPlus(
                "amount",
                "level",
                cvals=(("0", "0"), ("FULL", "TOP")),
            ),
        ),
        ("MMinus(x, y)", qr.MMinus("x", "y")),
        ("ADD(x, y, total)", qr.Add("x", "y", "total")),
        ("Mult(x, y, product)", qr.Mult("x", "y", "product")),
        ("Minus(flow, reverse)", qr.Minus("flow", "reverse")),
        ("Deriv(amount, netflow)", qr.Deriv("amount", "netflow")),
        ("Constant(inflow)", qr.Constant("inflow")),
        ("At(amount, 0)", qr.At("amount", "0")),
        ("At(amount, -1)", qr.At("amount", "-1")),
        ("Negligible(leak, flow)", qr.Negligible("leak", "flow")),
    ],
)
def test_parse_every_builtin_constraint(source, expected):
    assert qr.parse_constraint(source) == expected


def test_aliases_dotted_names_and_quoted_names():
    assert qr.parse_constraint("M-(A.amount, A.outflow)") == qr.MMinus(
        "A.amount", "A.outflow"
    )
    assert qr.parse_constraint('Deriv("amount-rate", "a\'")') == qr.Deriv(
        "amount-rate", "a'"
    )


@pytest.mark.parametrize(
    "constraint",
    [
        qr.MPlus("x", "y", cvals=(("0", "0"), ("HIGH", "TOP"))),
        qr.MMinus("x", "y"),
        qr.Add("x", "y", "z", cvals=(("0", "0", "0"),)),
        qr.Mult("x", "y", "z"),
        qr.Minus("x", "y"),
        qr.Deriv("A.amount", "amount-rate"),
        qr.Constant("class"),
        qr.At("x", "0"),
        qr.Negligible("small", "large"),
    ],
)
def test_canonical_format_round_trips(constraint):
    rendered = qr.format_constraint(constraint)
    assert qr.parse_constraint(rendered) == constraint


def test_model_constrain_accepts_string_and_keeps_schema_objects():
    model = qr.Model("string-authored")
    model.variable("x", landmarks=("0", "HIGH"), unbounded=True)
    model.variable("y", landmarks=("0", "TOP"), unbounded=True)
    added = model.constrain(
        'M+(x, y, cvals=[("0", "0"), (HIGH, TOP)])'
    )
    assert isinstance(added, qr.MPlus)
    assert model.constraints == [added]
    assert model.to_dict()["constraints"] == [
        {
            "kind": "mplus",
            "args": ["x", "y"],
            "cvals": [["0", "0"], ["HIGH", "TOP"]],
        }
    ]
    assert qr.Model.from_dict(model.to_dict()).to_dict() == model.to_dict()


@pytest.mark.parametrize(
    "source, match",
    [
        ("", "empty"),
        ("Unknown(x)", "unknown constraint kind"),
        ("Add(x, y)", "expects 3"),
        ("Add(x, y, z, nope=1)", "unknown keyword"),
        ("Deriv(x, y, cvals=[])", "does not accept"),
        ("M+(x, y, cvals=zero)", "list or tuple"),
        ("M+(x, y, cvals=[(0,)])", "arity 1"),
        ("qr.Add(x, y, z)", "direct call"),
        ('Add(__import__("os"), y, z)', "arguments must be"),
    ],
)
def test_syntax_errors_are_specific_and_safe(source, match):
    with pytest.raises(qr.ConstraintSyntaxError, match=match):
        qr.parse_constraint(source)


def test_model_validation_still_applies_after_parsing():
    model = qr.Model()
    model.variable("x")
    with pytest.raises(ValueError, match="undeclared variable"):
        model.constrain("Deriv(x, missing)")
    with pytest.raises(TypeError, match="Constraint or string"):
        model.constrain(42)
