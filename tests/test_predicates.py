"""Unit tests for the reference constraint predicates."""

import pytest

import qrlib as qr
from qrlib.engines import filters
from qrlib.quantity import Qdir, QVal


def compiled_pair(constraint, spaces):
    m = qr.Model()
    for name, kw in spaces.items():
        m.variable(name, **kw)
    m.constrain(constraint)
    cm = m.compile()
    return cm, cm.constraints[0]


UNBOUNDED = dict(landmarks=("0",), unbounded=True)
# unbounded space ranks: 0=-inf, 1=(-inf,0), 2=at 0, 3=(0,inf), 4=+inf
NEG, AT0, POS, PINF, NINF = 1, 2, 3, 4, 0


def test_deriv_ties_direction_to_sign():
    cm, cc = compiled_pair(qr.Deriv("x", "y"), {"x": UNBOUNDED, "y": UNBOUNDED})
    assert filters.check(cc, (QVal(AT0, Qdir.INC), QVal(POS, Qdir.DEC)))
    assert filters.check(cc, (QVal(POS, Qdir.STD), QVal(AT0, Qdir.INC)))
    assert filters.check(cc, (QVal(POS, Qdir.DEC), QVal(NEG, Qdir.STD)))
    assert not filters.check(cc, (QVal(AT0, Qdir.INC), QVal(NEG, Qdir.STD)))
    assert not filters.check(cc, (QVal(AT0, Qdir.STD), QVal(POS, Qdir.STD)))


def test_deriv_requires_zero_landmark_on_derivative():
    m = qr.Model()
    m.variable("x", landmarks=("0",), unbounded=True)
    m.variable("y", landmarks=("a",), unbounded=True)  # no 0
    m.constrain(qr.Deriv("x", "y"))
    with pytest.raises(ValueError):
        m.compile()


def test_mplus_matches_directions_and_corresponding_values():
    cm, cc = compiled_pair(
        qr.MPlus("x", "y", cvals=(("0", "0"),)), {"x": UNBOUNDED, "y": UNBOUNDED}
    )
    assert filters.check(cc, (QVal(POS, Qdir.INC), QVal(POS, Qdir.INC)))
    assert filters.check(cc, (QVal(AT0, Qdir.DEC), QVal(AT0, Qdir.DEC)))
    # directions must match
    assert not filters.check(cc, (QVal(POS, Qdir.INC), QVal(POS, Qdir.DEC)))
    # x above its cval landmark forces y above its partner
    assert not filters.check(cc, (QVal(POS, Qdir.STD), QVal(NEG, Qdir.STD)))
    assert not filters.check(cc, (QVal(AT0, Qdir.STD), QVal(POS, Qdir.STD)))


def test_minus_opposes_directions_and_signs():
    cm, cc = compiled_pair(qr.Minus("x", "y"), {"x": UNBOUNDED, "y": UNBOUNDED})
    assert filters.check(cc, (QVal(POS, Qdir.INC), QVal(NEG, Qdir.DEC)))
    assert filters.check(cc, (QVal(AT0, Qdir.STD), QVal(AT0, Qdir.STD)))
    assert not filters.check(cc, (QVal(POS, Qdir.INC), QVal(NEG, Qdir.INC)))
    assert not filters.check(cc, (QVal(POS, Qdir.STD), QVal(POS, Qdir.STD)))


def add_cc():
    return compiled_pair(
        qr.Add("x", "y", "z"),
        {"x": UNBOUNDED, "y": UNBOUNDED, "z": UNBOUNDED},
    )


def test_add_sign_algebra():
    cm, cc = add_cc()
    # + + + is fine; + + cannot sum to 0 or -
    assert filters.check(cc, (QVal(POS, Qdir.STD), QVal(POS, Qdir.STD), QVal(POS, Qdir.STD)))
    assert not filters.check(cc, (QVal(POS, Qdir.STD), QVal(POS, Qdir.STD), QVal(AT0, Qdir.STD)))
    assert not filters.check(cc, (QVal(POS, Qdir.STD), QVal(POS, Qdir.STD), QVal(NEG, Qdir.STD)))
    # opposing signs leave the sum unconstrained
    assert filters.check(cc, (QVal(POS, Qdir.STD), QVal(NEG, Qdir.STD), QVal(AT0, Qdir.STD)))


def test_add_direction_algebra():
    cm, cc = add_cc()
    # inc + inc = must be inc
    assert not filters.check(cc, (QVal(POS, Qdir.INC), QVal(POS, Qdir.INC), QVal(POS, Qdir.STD)))
    assert filters.check(cc, (QVal(POS, Qdir.INC), QVal(POS, Qdir.INC), QVal(POS, Qdir.INC)))
    # inc + dec = anything
    assert filters.check(cc, (QVal(POS, Qdir.INC), QVal(POS, Qdir.DEC), QVal(POS, Qdir.STD)))


def test_add_infinity_algebra():
    cm, cc = add_cc()
    # finite + (+inf) must be +inf
    assert not filters.check(cc, (QVal(AT0, Qdir.STD), QVal(PINF, Qdir.STD), QVal(POS, Qdir.STD)))
    assert filters.check(cc, (QVal(AT0, Qdir.STD), QVal(PINF, Qdir.STD), QVal(PINF, Qdir.STD)))
    # (+inf) + (-inf) is indeterminate
    assert filters.check(cc, (QVal(PINF, Qdir.STD), QVal(NINF, Qdir.STD), QVal(AT0, Qdir.STD)))
    # finite + finite cannot be infinite
    assert not filters.check(cc, (QVal(POS, Qdir.STD), QVal(POS, Qdir.STD), QVal(PINF, Qdir.STD)))


def test_mult_sign_rule():
    cm, cc = compiled_pair(
        qr.Mult("x", "y", "z"),
        {"x": UNBOUNDED, "y": UNBOUNDED, "z": UNBOUNDED},
    )
    assert filters.check(cc, (QVal(POS, Qdir.STD), QVal(NEG, Qdir.STD), QVal(NEG, Qdir.STD)))
    assert not filters.check(cc, (QVal(POS, Qdir.STD), QVal(NEG, Qdir.STD), QVal(POS, Qdir.STD)))
    assert filters.check(cc, (QVal(AT0, Qdir.INC), QVal(POS, Qdir.STD), QVal(AT0, Qdir.INC)))


def test_constant():
    cm, cc = compiled_pair(qr.Constant("x"), {"x": UNBOUNDED})
    assert filters.check(cc, (QVal(POS, Qdir.STD),))
    assert not filters.check(cc, (QVal(POS, Qdir.INC),))


def test_infinity_admissibility():
    m = qr.Model()
    m.variable("x", **UNBOUNDED)
    m.variable("y", **UNBOUNDED)
    cm = m.compile()
    # x at +inf with y still moving: excluded (infinity needs t = infinity)
    assert not filters.admissible_at_infinity(
        cm, (QVal(PINF, Qdir.INC), QVal(POS, Qdir.DEC))
    )
    # x at +inf with y steady: fine
    assert filters.admissible_at_infinity(
        cm, (QVal(PINF, Qdir.INC), QVal(POS, Qdir.STD))
    )
    # no variable at infinity: rule does not apply
    assert filters.admissible_at_infinity(
        cm, (QVal(POS, Qdir.INC), QVal(POS, Qdir.DEC))
    )
