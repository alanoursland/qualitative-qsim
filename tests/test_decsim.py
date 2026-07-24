"""DecSIM-style decomposed simulation: partition, guide, join."""

import json
from collections import Counter
from itertools import product

import pytest

import qrlib as qr
from qrlib import Qdir, SimStatus, TerminalClass, TimeTag
from qrlib import decompose as dc

from test_qsim_golden import bathtub, utube


def twin_tubs():
    """Two structurally independent bathtubs in one model."""
    m = qr.Model("twin-tubs")
    for tag in ("A", "B"):
        m.variable(f"amount{tag}", landmarks=("0", "FULL"))
        m.variable(f"level{tag}", landmarks=("0", "TOP"))
        m.variable(f"outflow{tag}", landmarks=("0", "OMAX"))
        m.variable(f"inflow{tag}", landmarks=("0", "IF*"))
        m.variable(f"netflow{tag}", landmarks=("0",), unbounded=True)
        m.constrain(
            qr.MPlus(f"amount{tag}", f"level{tag}", cvals=(("0", "0"), ("FULL", "TOP")))
        )
        m.constrain(
            qr.MPlus(f"level{tag}", f"outflow{tag}", cvals=(("0", "0"), ("TOP", "OMAX")))
        )
        m.constrain(qr.Add(f"netflow{tag}", f"outflow{tag}", f"inflow{tag}"))
        m.constrain(qr.Deriv(f"amount{tag}", f"netflow{tag}"))
        m.constrain(qr.Constant(f"inflow{tag}"))
    values = {}
    for tag in ("A", "B"):
        values[f"amount{tag}"] = ("0", Qdir.INC)
        values[f"level{tag}"] = ("0", Qdir.INC)
        values[f"outflow{tag}"] = ("0", Qdir.INC)
        values[f"inflow{tag}"] = ("IF*", Qdir.STD)
        values[f"netflow{tag}"] = (("0", "+inf"), Qdir.DEC)
    return m, m.state(time=TimeTag.POINT, **values)


def cascade():
    """Tank A drains through outA into tank B, which drains through outB."""
    m = qr.Model("cascade")
    m.variable("amtA", landmarks=("0", "A0"))
    m.variable("outA", landmarks=("0", "OA0"))
    m.variable("netA", landmarks=("0",), unbounded=True)
    m.variable("amtB", landmarks=("0", "B0"))
    m.variable("outB", landmarks=("0", "OB0"))
    m.variable("netB", landmarks=("0",), unbounded=True)
    m.constrain(qr.MPlus("amtA", "outA", cvals=(("0", "0"), ("A0", "OA0"))))
    m.constrain(qr.Minus("outA", "netA"))  # netA = -outA
    m.constrain(qr.Deriv("amtA", "netA"))
    m.constrain(qr.MPlus("amtB", "outB", cvals=(("0", "0"), ("B0", "OB0"))))
    m.constrain(qr.Add("netB", "outB", "outA"))  # netB = outA - outB
    m.constrain(qr.Deriv("amtB", "netB"))
    initial = m.state(
        time=TimeTag.POINT,
        amtA=("A0", Qdir.DEC),
        outA=("OA0", Qdir.DEC),
        netA=(("-inf", "0"), Qdir.INC),
        amtB=("0", Qdir.INC),
        outB=("0", Qdir.INC),
        netB=(("0", "+inf"), Qdir.DEC),
    )
    return m, initial


CASCADE_PARTS = {"tankA": ("amtA", "outA", "netA"), "tankB": ("amtB", "outB", "netB")}
# the net-flow variables chatter (their directions wiggle without bound);
# abstracting them is what makes even the monolithic run complete
CASCADE_CFG = qr.SimConfig(ignore_qdir=("netA", "netB"))


def assert_covers(dec_result, mono_result, model):
    """Every monolithic behavior must project into each component's
    behavior set and appear as a linked tuple (the coverage direction of
    the DecSIM equivalence: real behaviors survive decomposition)."""
    graph = mono_result.graph
    joint = set(dec_result.joint_behaviors())
    for beh in mono_result.behaviors():
        choices = []
        for run in dec_result.runs:
            svars = tuple(run.part.model.variables)
            declared = {v: model.variables[v].space for v in svars}
            ms = dc.episode_stream(graph, beh, svars, declared)
            cs = {
                k: dc.episode_stream(run.result.graph, run.behaviors[k], svars, declared)
                for k in run.live
            }
            choices.append([k for k, s in cs.items() if dc.compatible(ms, s)])
        assert all(choices), "a component has no matching behavior"
        assert any(t in joint for t in product(*choices)), "no linked tuple"


# --- partitioning ----------------------------------------------------------


def test_suggest_partition_splits_independent_couples_coupled():
    m, _ = twin_tubs()
    parts = dc.suggest_partition(m)
    assert [c.variables for c in parts] == [
        ("amountA", "levelA", "outflowA", "inflowA", "netflowA"),
        ("amountB", "levelB", "outflowB", "inflowB", "netflowB"),
    ]
    m2, _ = cascade()
    assert len(dc.suggest_partition(m2)) == 1  # coupled: one component


def test_constraint_assignment_and_interface():
    m, _ = cascade()
    dec = dc.decompose(m, CASCADE_PARTS)
    a, b = dec.parts
    assert a.interface == ()  # tank A is self-contained
    assert b.interface == ("outA",)  # Add(netB, outB, outA) drags outA in
    assert len(a.model.constraints) == 3 and len(b.model.constraints) == 3
    assert dec.shared(0, 1) == ("outA",)


def test_decompose_validation():
    m, _ = cascade()
    with pytest.raises(ValueError, match="unknown variables"):
        dc.decompose(m, {"x": ("nope",), "y": tuple(m.variables)})
    with pytest.raises(ValueError, match="owned by both"):
        dc.decompose(m, {"x": ("amtA",), "y": tuple(m.variables)})
    with pytest.raises(ValueError, match="not owned"):
        dc.decompose(m, {"x": ("amtA", "outA", "netA")})
    with pytest.raises(ValueError, match="duplicate component names"):
        dc.decompose(m, (dc.Component("x", ("amtA",)), dc.Component("x", ("outA",))))
    with pytest.raises(TypeError, match="uncompiled Model"):
        dc.decompose(m.compile(), CASCADE_PARTS)
    mr, _ = bathtub()
    mr.region("full-range")
    with pytest.raises(ValueError, match="operating regions"):
        dc.decompose(mr, {"all": tuple(mr.variables)})


def test_decsim_rejects_user_guide():
    from qrlib.guide import G, mag

    m, init = twin_tubs()
    cfg = qr.SimConfig(guide=G(mag("amountA", ">=", "0")))
    with pytest.raises(ValueError, match="guided decomposed simulation"):
        dc.decsim(m, init, config=cfg)


# --- independent subsystems: the interleaving blowup disappears ------------


def test_independent_subsystems():
    m, init = twin_tubs()
    r = dc.decsim(m, init)  # auto-partition
    assert r.status is SimStatus.COMPLETE
    assert r.stats["component_nodes"] == {"c0": 5, "c1": 5}
    assert r.stats["component_behaviors"] == {"c0": 3, "c1": 3}
    assert r.links == {}  # no shared variables: nothing to reconcile
    assert len(r.joint_behaviors()) == 9

    mono = qr.qsim(m, init)
    # the monolithic graph pays for interleaving the two tubs' events;
    # the decomposition never even represents the orderings
    assert mono.stats["nodes"] == 33
    assert len(mono.behaviors()) == 23
    assert r.stats["total_nodes"] == 10 < mono.stats["nodes"]
    assert_covers(r, mono, m)


def test_component_graph_matches_single_tub_golden():
    m, init = twin_tubs()
    r = dc.decsim(m, init)
    single = qr.qsim(*bathtub())
    for run in r.runs:
        assert Counter(b.terminal for b in run.behaviors) == Counter(
            b.terminal for b in single.behaviors()
        )


# --- coupled cascade: upstream words guide downstream ----------------------


def test_cascade_guided_by_upstream_word():
    m, init = cascade()
    r = dc.decsim(m, init, CASCADE_PARTS, config=CASCADE_CFG)
    assert r.status is SimStatus.COMPLETE
    tank_a, tank_b = r.runs
    assert tank_a.guided == () and tank_b.guided == ("outA",)
    # the guide (tank A's story for outA, as a temporal word) keeps the
    # downstream run finite: without it, outA's magnitude wanders freely
    # and the run blows the state budget
    assert tank_b.result.stats["nodes"] == 15
    assert tank_b.result.config.guide is not None
    assert Counter(tank_b.behaviors[k].terminal for k in tank_b.live) == {
        TerminalClass.QUIESCENT: 4,
        TerminalClass.DOMAIN_EXIT: 1,
    }

    mono = qr.qsim(m, init, config=CASCADE_CFG)
    assert len(mono.behaviors()) == 5
    assert len(r.joint_behaviors()) == 5
    assert_covers(r, mono, m)


def test_cascade_unguided_downstream_blows_up():
    # the contrast that motivates guiding: simulate tank B's sub-model
    # alone, interface chatter-abstracted but unconstrained in magnitude
    m, init = cascade()
    dec = dc.decompose(m, CASCADE_PARTS)
    part = dec.parts[1]
    sub_init = qr.QState.from_dict(
        {v: init[v] for v in part.model.variables}, init.time
    )
    free = qr.qsim(
        part.model, sub_init, config=qr.SimConfig(ignore_qdir=("netB", "outA"))
    )
    assert free.status is SimStatus.TRUNCATED  # blows the 500-state budget


# --- cyclic coupling falls back to chatter + post-hoc join -----------------


def test_cyclic_coupling_falls_back():
    m, init = utube()  # a and b mutually coupled through diff/flow
    comps = {"left": ("a", "mflow"), "right": ("b", "diff", "flow")}
    r = dc.decsim(m, init, comps, max_states=60)
    left, right = r.runs
    # the cycle is broken at 'left': its interface var is chatter-abstracted,
    # and 'right' is still guided by left's words for a
    assert left.guided == () and "flow" in left.result.config.ignore_qdir
    assert right.guided == ("a",)
    assert r.status is SimStatus.TRUNCATED  # honest: a bad cut stays reported
    mono = qr.qsim(m, init)
    assert_covers(r, mono, m)


# --- stream algebra --------------------------------------------------------


def test_stream_compatibility_algebra():
    S = dc.Stream
    x, y, z = (0,), (1,), (2,)
    # prefix agreement: a truncated word never conflicts with an extension
    assert dc.compatible(S((x, y)), S((x, y, z)))
    assert not dc.compatible(S((x, z)), S((x, y, z)))
    # a forever-pinned word conflicts with a partner that changes again
    assert not dc.compatible(S((x, y), forever=True), S((x, y, z)))
    assert dc.compatible(S((x, y), forever=True), S((x, y)))  # partner ends
    assert dc.compatible(S((x, y), forever=True), S((x, y), forever=True))
    # collapse: repeated letters are one episode
    assert dc.compatible(S((x, x, y, y)), S((x, y)))
    # cycles: an eventually-constant loop equals quiescence
    assert dc.compatible(S((x,), loop=(y,)), S((x, y), forever=True))
    # a genuinely periodic loop conflicts with quiescence...
    assert not dc.compatible(S((), loop=(x, y)), S((x, y), forever=True))
    # ...and matches itself, including phase-equivalent unrollings
    assert dc.compatible(S((), loop=(x, y)), S((x,), loop=(y, x)))
    assert not dc.compatible(S((), loop=(x, y)), S((), loop=(x, z)))


def test_declared_rank_dissolves_discovered_landmarks():
    m, init = bathtub()
    result = qr.qsim(m, init)
    graph = result.graph
    declared = {"amount": m.variables["amount"].space}
    quiescent = [
        b for b in result.behaviors() if b.terminal is TerminalClass.QUIESCENT
    ]
    streams = [dc.episode_stream(graph, b, ("amount",), declared) for b in quiescent]
    # below-FULL equilibrium sits at discovered amount*0: in declared terms
    # it stays inside (0, FULL); the at-FULL equilibrium reaches rank 2
    words = {s.letters for s in streams}
    assert words == {((0,), (1,), (1,)), ((0,), (1,), (2,))}
    assert all(s.forever for s in streams)


# --- exports ---------------------------------------------------------------


def test_to_dict_exports_plain_data():
    m, init = cascade()
    r = dc.decsim(m, init, CASCADE_PARTS, config=CASCADE_CFG)
    data = r.to_dict()
    json.dumps(data)
    assert data["schema"] == "qrlib.decsim/v1"
    assert data["status"] == "complete"
    assert [c["name"] for c in data["components"]] == ["tankA", "tankB"]
    assert data["components"][1]["guided"] == ["outA"]
    assert data["joint_behaviors"] == 5
    assert data["links"] == [[0, 1, [list(p) for p in r.links[(0, 1)]]]]
