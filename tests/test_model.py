import pytest

import qrlib as qr
from qrlib import Qdir, TimeTag


def bathtub() -> qr.Model:
    m = qr.Model("bathtub")
    m.variable("amount", landmarks=("0", "FULL"), upper_unbounded=True)
    m.variable("level", landmarks=("0",), upper_unbounded=True)
    m.variable("outflow", landmarks=("0",), upper_unbounded=True)
    m.variable("inflow", landmarks=("0", "IF*"), upper_unbounded=True)
    m.variable("netflow", landmarks=("0",), unbounded=True)

    m.constrain(qr.MPlus("amount", "level", cvals=(("0", "0"),)))
    m.constrain(qr.MPlus("level", "outflow", cvals=(("0", "0"),)))
    m.constrain(qr.Add("netflow", "outflow", "inflow"))
    m.constrain(qr.Deriv("amount", "netflow"))
    m.constrain(qr.Constant("inflow"))
    return m


def test_bathtub_builds():
    m = bathtub()
    assert set(m.variables) == {"amount", "level", "outflow", "inflow", "netflow"}
    assert len(m.constraints) == 5


def test_duplicate_variable_rejected():
    m = qr.Model()
    m.variable("x")
    with pytest.raises(ValueError):
        m.variable("x")


def test_constraint_referencing_unknown_variable_rejected():
    m = qr.Model()
    m.variable("x")
    with pytest.raises(ValueError):
        m.constrain(qr.Deriv("x", "nope"))


def test_corresponding_values_validated():
    m = qr.Model()
    m.variable("x", landmarks=("0",))
    m.variable("y", landmarks=("0",))
    with pytest.raises(ValueError):
        m.constrain(qr.MPlus("x", "y", cvals=(("0", "NOT_A_LANDMARK"),)))
    with pytest.raises(ValueError):  # wrong arity
        qr.MPlus("x", "y", cvals=(("0",),))


def test_initial_state_construction_and_encoding():
    m = bathtub()
    s = m.state(
        time=TimeTag.POINT,
        amount=("0", Qdir.INC),
        level=("0", Qdir.INC),
        outflow=("0", Qdir.INC),
        inflow=("IF*", Qdir.STD),
        netflow=qr.QVal(
            m.variables["netflow"].space.rank_between("0", "+inf"), Qdir.DEC
        ),
    )
    assert s["inflow"].dir == Qdir.STD
    order = tuple(sorted(m.variables))
    enc = s.encode(order)
    assert len(enc) == 2 * len(order)
    # states are values: identical construction is equal and hashable
    s2 = m.state(time=TimeTag.POINT, **s.as_dict())
    assert s == s2 and hash(s) == hash(s2)


def test_state_requires_all_variables():
    m = bathtub()
    with pytest.raises(ValueError):
        m.state(amount=("0", Qdir.STD))


def test_compile_freezes_order_and_resolves_cvals():
    m = bathtub()
    cm = m.compile()
    assert cm.var_order == tuple(m.variables)
    # MPlus(amount, level) resolved its (0, 0) corresponding value to ranks
    mplus = cm.constraints[0]
    assert mplus.kind == "mplus"
    assert (0, 0) in mplus.cvals
    # Add gained the implicit (0, 0, 0) corresponding value
    add = next(cc for cc in cm.constraints if cc.kind == "add")
    zeros = tuple(add.zeros)
    assert zeros in add.cvals
