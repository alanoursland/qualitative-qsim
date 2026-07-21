"""Golden-model tests for the reference QSIM engine.

Expected behavior sets follow the literature (Kuipers 1986/1994):

- Bathtub with constant inflow: three behaviors — equilibrium below FULL,
  equilibrium exactly at FULL, overflow (region exit past FULL).
- U-tube: a single behavior — flow decays to zero, both levels become
  steady (equal-pressure equilibrium).
- Frictionless spring: a single behavior — the sustained oscillation,
  closed as a cycle back to the initial state after 8 transitions.
"""

import pytest

import qrlib as qr
from qrlib import Qdir, SimStatus, TerminalClass, TimeTag
from qrlib.analysis import queries


# --- bathtub ---------------------------------------------------------------


def bathtub():
    m = qr.Model("bathtub")
    m.variable("amount", landmarks=("0", "FULL"))
    m.variable("level", landmarks=("0", "TOP"))
    m.variable("outflow", landmarks=("0", "OMAX"))
    m.variable("inflow", landmarks=("0", "IF*"))
    m.variable("netflow", landmarks=("0",), unbounded=True)
    m.constrain(qr.MPlus("amount", "level", cvals=(("0", "0"), ("FULL", "TOP"))))
    m.constrain(qr.MPlus("level", "outflow", cvals=(("0", "0"), ("TOP", "OMAX"))))
    m.constrain(qr.Add("netflow", "outflow", "inflow"))
    m.constrain(qr.Deriv("amount", "netflow"))
    m.constrain(qr.Constant("inflow"))
    initial = m.state(
        time=TimeTag.POINT,
        amount=("0", Qdir.INC),
        level=("0", Qdir.INC),
        outflow=("0", Qdir.INC),
        inflow=("IF*", Qdir.STD),
        netflow=(("0", "+inf"), Qdir.DEC),
    )
    return m, initial


def test_bathtub_three_outcomes():
    m, initial = bathtub()
    result = qr.qsim(m, initial)
    assert result.status is SimStatus.COMPLETE

    behaviors = result.behaviors()
    assert len(behaviors) == 3
    census = queries.terminal_census(result.graph)
    assert census == {TerminalClass.QUIESCENT: 2, TerminalClass.REGION_EXIT: 1}

    space = m.variables["amount"].space
    quiescent_amounts = {
        result.graph.nodes[i].state["amount"].mag
        for i in queries.quiescent_states(result.graph)
    }
    # equilibrium strictly below FULL, and equilibrium exactly at FULL
    assert quiescent_amounts == {
        space.rank_between("0", "FULL"),
        space.rank_of("FULL"),
    }

    (overflow,) = [
        b for b in behaviors if b.terminal is TerminalClass.REGION_EXIT
    ]
    final = overflow.states[-1]
    assert final["amount"].mag == space.rank_of("FULL")
    assert final["amount"].dir is Qdir.INC  # still rising: leaves the model


def test_bathtub_tree_shape():
    m, initial = bathtub()
    result = qr.qsim(m, initial)
    # root point -> single interval state -> three terminal points
    root = result.graph.nodes[result.graph.root]
    assert len(root.children) == 1
    (interval_id,) = root.children
    assert len(result.graph.nodes[interval_id].children) == 3


# --- U-tube ----------------------------------------------------------------


def utube():
    m = qr.Model("u-tube")
    m.variable("a", landmarks=("0",), upper_unbounded=True)
    m.variable("b", landmarks=("0",), upper_unbounded=True)
    m.variable("diff", landmarks=("0",), unbounded=True)
    m.variable("flow", landmarks=("0",), unbounded=True)
    m.variable("mflow", landmarks=("0",), unbounded=True)
    m.constrain(qr.Add("diff", "b", "a"))  # diff + b = a
    m.constrain(qr.MPlus("diff", "flow", cvals=(("0", "0"),)))
    m.constrain(qr.Minus("flow", "mflow"))
    m.constrain(qr.Deriv("b", "flow"))
    m.constrain(qr.Deriv("a", "mflow"))
    initial = m.state(
        time=TimeTag.POINT,
        a=(("0", "+inf"), Qdir.DEC),
        b=(("0", "+inf"), Qdir.INC),
        diff=(("0", "+inf"), Qdir.DEC),
        flow=(("0", "+inf"), Qdir.DEC),
        mflow=(("-inf", "0"), Qdir.INC),
    )
    return m, initial


def test_utube_single_equilibrium_behavior():
    m, initial = utube()
    result = qr.qsim(m, initial)
    assert result.status is SimStatus.COMPLETE

    behaviors = result.behaviors()
    assert len(behaviors) == 1
    (b,) = behaviors
    assert b.terminal is TerminalClass.QUIESCENT
    assert len(b.states) == 3  # point -> interval -> quiescent point

    final = b.states[-1]
    diff_space = m.variables["diff"].space
    a_space = m.variables["a"].space
    assert final["diff"].mag == diff_space.rank_of("0")
    assert final["flow"].mag == diff_space.rank_of("0")
    # levels equilibrate at unnamed values inside their intervals
    assert final["a"].mag == a_space.rank_between("0", "+inf")
    assert all(final[v].dir is Qdir.STD for v in ("a", "b", "diff", "flow", "mflow"))


# --- frictionless spring ---------------------------------------------------


def spring():
    m = qr.Model("spring")
    m.variable("x", landmarks=("0",), unbounded=True)
    m.variable("v", landmarks=("0",), unbounded=True)
    m.variable("a", landmarks=("0",), unbounded=True)
    m.constrain(qr.Deriv("x", "v"))
    m.constrain(qr.Deriv("v", "a"))
    m.constrain(qr.Minus("x", "a"))
    initial = m.state(
        time=TimeTag.POINT,
        x=("0", Qdir.INC),
        v=(("0", "+inf"), Qdir.STD),  # released through center at peak speed
        a=("0", Qdir.DEC),
    )
    return m, initial


def test_spring_sustained_oscillation():
    m, initial = spring()
    result = qr.qsim(m, initial)
    assert result.status is SimStatus.COMPLETE

    behaviors = result.behaviors()
    assert len(behaviors) == 1
    (b,) = behaviors
    assert b.terminal is TerminalClass.CYCLE
    assert b.cycle_target == result.graph.root
    # 8 transitions through the full cycle: 4 quarter-swings, each an
    # interval plus its closing point.
    assert len(b.states) == 9

    (loop,) = queries.cycles(result.graph)
    assert loop[0] == result.graph.root
    assert len(loop) == 9


def test_initial_state_must_be_consistent():
    m, _ = bathtub()
    bad = m.state(
        time=TimeTag.POINT,
        amount=("0", Qdir.DEC),  # falling amount contradicts positive netflow
        level=("0", Qdir.INC),
        outflow=("0", Qdir.INC),
        inflow=("IF*", Qdir.STD),
        netflow=(("0", "+inf"), Qdir.DEC),
    )
    with pytest.raises(ValueError):
        qr.qsim(m, bad)


def test_truncation_is_reported():
    m, initial = spring()
    result = qr.qsim(m, initial, max_depth=3)
    assert result.status is SimStatus.TRUNCATED
    census = queries.terminal_census(result.graph)
    assert TerminalClass.TRUNCATED in census
