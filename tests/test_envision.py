"""Total envisionment: all consistent states, connected."""

import json

import pytest

import qrlib as qr
from qrlib import TimeTag

from test_decsim import cascade
from test_qsim_golden import bathtub, spring


def test_bathtub_full_phase_portrait():
    m, _ = bathtub()
    env = qr.envision(m)
    assert env.stats == {
        "assignments": 17,
        "states": 17,
        "point_nodes": 17,
        "interval_nodes": 10,
        "edges": 19,
    }
    # every equilibrium of the model, not just the reachable ones: empty,
    # full, and mid-level balances for inflow at IF* AND below it — the
    # attainable envisionment from the standard fill-up initial state
    # only ever sees the IF* ones
    described = {env.describe(n) for n in env.quiescent()}
    assert (
        "[•] amount=0· level=0· outflow=0· inflow=0· netflow=0·" in described
    )
    assert any("inflow=(0, IF*)" in d for d in described)
    assert len(env.quiescent()) == 5


def test_attainable_is_a_subgraph_of_total():
    m, init = bathtub()
    env = qr.envision(m)
    att = qr.qsim(
        m, init, config=qr.SimConfig(envisionment=True, discover_landmarks=False)
    )
    ids = {n.state: n.id for n in env.nodes}
    for node in att.graph.nodes.values():
        assert node.state in ids
    edge_set = set(env.edges)
    for node in att.graph.nodes.values():
        for child in node.children:
            edge = (ids[node.state], ids[att.graph.nodes[child].state])
            assert edge in edge_set


def test_spring_portrait_contains_the_oscillation():
    m, init = spring()
    env = qr.envision(m)
    # the four phase quadrants persist as interval states (+ the origin)
    assert env.stats["interval_nodes"] == 5
    result = qr.qsim(m, init, config=qr.SimConfig(discover_landmarks=False))
    (cyc,) = result.behaviors()
    ids = {n.state: n.id for n in env.nodes}
    edge_set = set(env.edges)
    for a, b in zip(cyc.node_ids, cyc.node_ids[1:]):
        edge = (
            ids[result.graph.nodes[a].state],
            ids[result.graph.nodes[b].state],
        )
        assert edge in edge_set
    # the oscillation is the model's one recurrent set
    (scc,) = env.cycles()
    assert {ids[st] for st in cyc.states} <= set(scc)
    # divergent limit states are terminal
    for n in env.nodes:
        if n.kind == "divergent":
            assert env.successors(n.id) == ()


def test_regions_are_envisioned_separately():
    from qrlib.frontends import qpt

    s = qpt.System("drip")
    s.quantity("x", landmarks=("0", "MAX"))
    s.quantity("drain", landmarks=("0", "DMAX"))
    s.proportional("drain", "x", +1, cvals=(("0", "0"), ("MAX", "DMAX")))
    s.process("drip", when=(("x", ">", "0"),)).influence("x", -1, "drain")
    m = s.build()
    with pytest.raises(ValueError, match="multi-region"):
        qr.envision(m)
    on = qr.envision(m, region="drip")
    off = qr.envision(m, region="!drip")
    # with the process off every rate is pinned: the region's portrait has
    # no transient states at all — everything consistent is a fixed point
    # (or a divergent limit)
    assert all(
        n.kind != "transient"
        for n in off.nodes
        if n.state.time is TimeTag.POINT
    )
    assert any(n.kind == "transient" for n in on.nodes)


def test_chatter_abstraction_shrinks_the_portrait():
    m, _ = cascade()
    plain = qr.envision(m)
    abstracted = qr.envision(m, ignore_qdir=("netA", "netB"))
    assert abstracted.stats["states"] < plain.stats["states"]


def test_cap_is_enforced():
    m, _ = bathtub()
    with pytest.raises(ValueError, match="max_states"):
        qr.envision(m, max_states=5)


def test_validation_and_export():
    m, _ = bathtub()
    with pytest.raises(ValueError, match="unknown variables"):
        qr.envision(m, ignore_qdir=("nope",))
    data = qr.envision(m).to_dict()
    json.dumps(data)
    assert data["schema"] == "qrlib.envisionment/v1"
    assert data["region"] == "default"
    assert len(data["nodes"]) == data["stats"]["point_nodes"] + data["stats"]["interval_nodes"]
