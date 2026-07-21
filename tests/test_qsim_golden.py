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

    # equilibrium below FULL gets a *discovered* landmark; the other sits at FULL
    quiescent_amounts = set()
    for i in queries.quiescent_states(result.graph):
        node = result.graph.nodes[i]
        amount_space = node.model.spaces[node.model.index("amount")]
        quiescent_amounts.add(amount_space.describe(node.state["amount"].mag))
    assert quiescent_amounts == {"amount*0", "FULL"}

    (overflow,) = [
        b for b in behaviors if b.terminal is TerminalClass.REGION_EXIT
    ]
    final = overflow.states[-1]
    space = m.variables["amount"].space
    assert final["amount"].mag == space.rank_of("FULL")
    assert final["amount"].dir is Qdir.INC  # still rising: leaves the model


def test_bathtub_discovery_records_corresponding_values():
    m, initial = bathtub()
    result = qr.qsim(m, initial)
    def amount_desc(node):
        space = node.model.spaces[node.model.index("amount")]
        return space.describe(node.state["amount"].mag)

    (below_full,) = [
        node
        for node in (result.graph.nodes[i] for i in queries.quiescent_states(result.graph))
        if amount_desc(node) == "amount*0"
    ]
    frame = below_full.model
    # amount, level, outflow all became steady at unnamed values -> all minted
    for var in ("amount", "level", "outflow"):
        space = frame.spaces[frame.index(var)]
        assert f"{var}*0" in space.names
    # M+(amount, level) learned the equilibrium corresponding pair
    mplus = frame.constraints[0]
    a_space = frame.spaces[frame.index("amount")]
    l_space = frame.spaces[frame.index("level")]
    assert (a_space.rank_of("amount*0"), l_space.rank_of("level*0")) in mplus.cvals


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
    final_node = result.graph.nodes[b.node_ids[-1]]
    frame = final_node.model
    diff_space = m.variables["diff"].space
    assert final["diff"].mag == diff_space.rank_of("0")
    assert final["flow"].mag == diff_space.rank_of("0")
    # levels equilibrate at *discovered* landmarks
    a_space = frame.spaces[frame.index("a")]
    b_space = frame.spaces[frame.index("b")]
    assert a_space.describe(final["a"].mag) == "a*0"
    assert b_space.describe(final["b"].mag) == "b*0"
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
    # Without landmark discovery the oscillation closes in one period
    # (with discovery, peaks mint landmarks first — see test_qsim_phase2).
    m, initial = spring()
    result = qr.qsim(m, initial, config=qr.SimConfig(discover_landmarks=False))
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
