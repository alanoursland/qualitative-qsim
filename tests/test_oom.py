"""Order-of-magnitude reasoning (FOG's Ne): Negligible declarations."""

import json

import pytest

import qrlib as qr
from qrlib import Landmark, Qdir, TerminalClass, TimeTag
from qrlib.analysis import causal
from qrlib.bridge import abstraction, coverage
from qrlib.engines import filters

from test_soundness import CFG


def perturbed(ne=True, *, values=False):
    """A small bounded quantity rides on a large constant: z = x + y."""
    m = qr.Model("perturbed-sum")
    zero = Landmark("0", value=0.0) if values else "0"
    m.variable("u", landmarks=(zero,), lower_unbounded=True)
    m.variable(
        "x",
        landmarks=(
            (Landmark("XMIN", value=-2.0), zero, Landmark("XMAX", value=3.0))
            if values
            else ("XMIN", "0", "XMAX")
        ),
    )
    m.variable(
        "y",
        landmarks=(zero, Landmark("Y*", value=10.0)) if values else ("0", "Y*"),
    )
    m.variable("z", landmarks=(zero,), unbounded=True)
    m.constrain(qr.Constant("u"))
    m.constrain(qr.Deriv("x", "u"))  # x declines at the constant rate u < 0
    m.constrain(qr.Constant("y"))
    m.constrain(qr.Add("x", "y", "z"))
    if ne:
        m.constrain(qr.Negligible("x", "y"))
    initial = m.state(
        time=TimeTag.POINT,
        u=(("-inf", "0"), Qdir.STD),
        x=("XMAX", Qdir.DEC),
        y=("Y*", Qdir.STD),
        z=(("0", "+inf"), Qdir.DEC),
    )
    return m, initial


# --- declaration and compilation -------------------------------------------


def test_declaration_validation():
    with pytest.raises(ValueError, match="two distinct"):
        qr.Negligible("x", "x")
    m = qr.Model("no-zero")
    m.variable("a", landmarks=("A0",))
    m.variable("b", landmarks=("0",), unbounded=True)
    m.constrain(qr.Negligible("a", "b"))
    with pytest.raises(ValueError, match="'0' landmark"):
        m.compile()


def test_closure_is_transitive_and_detects_contradictions():
    m = qr.Model("trans")
    for v in ("a", "b", "c", "s"):
        m.variable(v, landmarks=("0",), unbounded=True)
    m.constrain(qr.Negligible("a", "b"))
    m.constrain(qr.Negligible("b", "c"))
    m.constrain(qr.Add("a", "c", "s"))  # ordered only through the closure
    (add,) = [cc for cc in m.compile().constraints if cc.kind == "add"]
    assert add.dominant == 1

    m2 = qr.Model("contra")
    m2.variable("a", landmarks=("0",), unbounded=True)
    m2.variable("b", landmarks=("0",), unbounded=True)
    m2.constrain(qr.Negligible("a", "b"))
    m2.constrain(qr.Negligible("b", "a"))
    with pytest.raises(ValueError, match="contradictory Negligible cycle"):
        m2.compile()


def test_regions_must_all_support_the_ordering():
    m = qr.Model("regioned")
    for v in ("a", "b", "s"):
        m.variable(v, landmarks=("0",), unbounded=True)
    ne = m.constrain(qr.Negligible("a", "b"))
    add = m.constrain(qr.Add("a", "b", "s"))
    m.region("with-ne", constraints=(ne, add))
    m.region("without", constraints=(add,))
    (cc,) = [c for c in m.compile().constraints if c.kind == "add"]
    assert cc.dominant is None  # region "without" cannot justify it


# --- the disambiguation ----------------------------------------------------


def test_add_sign_fork_is_resolved():
    # without the declaration, sign algebra forks when x goes negative:
    # z = x + y could cross zero, touch it, or stay positive
    plain_m, plain_init = perturbed(ne=False)
    plain = qr.qsim(plain_m, plain_init)
    assert len(plain.behaviors()) == 3

    ne_m, ne_init = perturbed(ne=True)
    ne = qr.qsim(ne_m, ne_init)
    assert len(ne.behaviors()) == 1
    (b,) = ne.behaviors()
    assert b.terminal is TerminalClass.REGION_EXIT  # x runs into XMIN
    zero = ne_m.variables["z"].space.rank_of("0")
    assert all(st["z"].mag > zero for st in b.states)  # z never reaches 0


def test_negligible_is_itself_checked():
    m = qr.Model("pair")
    m.variable("x", landmarks=("0",), unbounded=True)
    m.variable("y", landmarks=("0",), unbounded=True)
    m.constrain(qr.Negligible("x", "y"))
    frame = m.compile()
    (ne,) = frame.constraints

    def state(xv, yv):
        return m.state(time=TimeTag.POINT, x=xv, y=yv)

    # |x| < |y| = 0 is impossible
    bad = state((("0", "+inf"), Qdir.STD), ("0", Qdir.STD))
    assert filters.check_state(frame, bad) is ne
    # an infinite small quantity cannot be dominated by a finite large one
    worse = state(("+inf", Qdir.STD), (("0", "+inf"), Qdir.STD))
    assert filters.check_state(frame, worse) is ne
    ok = state((("0", "+inf"), Qdir.STD), (("0", "+inf"), Qdir.STD))
    assert filters.check_state(frame, ok) is None


# --- integration -----------------------------------------------------------


def test_tensor_path_agrees():
    m, init = perturbed(ne=True)
    ref = qr.qsim(m, init)
    ten = qr.qsim(m, init, config=qr.SimConfig(use_tensor=True))
    assert ten.graph.export() == ref.graph.export()


def test_causal_ordering_skips_the_annotation():
    # Negligible is an inequality, not an equation: the model stays
    # exactly determined and z is still caused by the Add
    m, _ = perturbed(ne=True)
    order = causal.causal_order(m)
    assert not order.singular
    (z_link,) = [l for l in order.links if l.variable == "z"]
    assert z_link.relation == "+"


def test_serialization_and_sign_structure():
    m, _ = perturbed(ne=True)
    data = m.to_dict()
    json.dumps(data)
    assert {"kind": "negligible", "args": ["x", "y"]} in data["constraints"]
    assert qr.Model.from_dict(data).to_dict() == data
    assert m.sign_structure().negligible == (("x", "y"),)


def test_real_trajectories_stay_covered():
    # y = 10, u = -1, x: 3 -> -2 (|x| < y throughout, as declared);
    # z = x + 10 stays positive and the single behavior covers it
    m, init = perturbed(ne=True, values=True)
    steps = 200
    rows = []
    for k in range(steps + 1):
        t = 5.0 * k / steps
        x = 3.0 - t
        rows.append([-1.0, x, 10.0, x + 10.0])
    result = qr.qsim(m, init)
    observed = abstraction.abstract_trajectory(rows, m, config=CFG)
    res = coverage.check(observed, result.graph)
    assert res.covered, res.diagnosis
