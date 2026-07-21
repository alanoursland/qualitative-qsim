"""Phase-4: operating regions — engine semantics, mode channel end-to-end,
and the versioned model schema."""

import json

import pytest

import qrlib as qr
from qrlib import Landmark, Qdir, SimStatus, TerminalClass, TimeTag
from qrlib.bridge import abstraction, coverage
from qrlib.bridge.abstraction import AbstractionConfig


def two_region_bathtub(*, values=False):
    """Bathtub that crosses into a steady-overflow region at FULL."""
    full, inflow = 1.0, 1.0
    f1 = lambda A: 1.2 * A
    f2 = lambda L: 0.4 * L  # OMAX < inflow: overflow genuinely happens

    def lm(name, v):
        return Landmark(name, value=v) if values else name

    m = qr.Model("bathtub2")
    m.variable("amount", landmarks=(lm("0", 0.0), lm("FULL", full)))
    m.variable("level", landmarks=(lm("0", 0.0), lm("TOP", f1(full))))
    m.variable("outflow", landmarks=(lm("0", 0.0), lm("OMAX", f2(f1(full)))))
    m.variable("inflow", landmarks=(lm("0", 0.0), lm("IF*", inflow)))
    m.variable("netflow", landmarks=(lm("0", 0.0),), unbounded=True)
    c_al = m.constrain(qr.MPlus("amount", "level", cvals=(("0", "0"), ("FULL", "TOP"))))
    c_lo = m.constrain(qr.MPlus("level", "outflow", cvals=(("0", "0"), ("TOP", "OMAX"))))
    c_add = m.constrain(qr.Add("netflow", "outflow", "inflow"))
    c_der = m.constrain(qr.Deriv("amount", "netflow"))
    c_inf = m.constrain(qr.Constant("inflow"))
    c_amt = m.constrain(qr.Constant("amount"))  # active only while overflowing
    m.region("filling", constraints=(c_al, c_lo, c_add, c_der, c_inf))
    m.region("overflowing", constraints=(c_al, c_lo, c_add, c_inf, c_amt))
    m.transition(
        "filling",
        "overflowing",
        when=(("amount", "==", "FULL"), ("netflow", ">", "0")),
    )
    initial = m.state(
        time=TimeTag.POINT,
        amount=("0", Qdir.INC),
        level=("0", Qdir.INC),
        outflow=("0", Qdir.INC),
        inflow=("IF*", Qdir.STD),
        netflow=(("0", "+inf"), Qdir.DEC),
    )
    params = (full, inflow, f1, f2)
    return m, initial, params


def test_two_region_bathtub_crosses_into_steady_overflow():
    m, initial, _ = two_region_bathtub()
    result = qr.qsim(m, initial)
    assert result.status is SimStatus.COMPLETE
    assert result.stats["region_crossings"] == 1

    behaviors = result.behaviors()
    assert len(behaviors) == 3
    # the former REGION_EXIT branch now continues into the overflow region
    crossing = [b for b in behaviors if "overflowing" in b.regions]
    assert len(crossing) == 1
    (b,) = crossing
    assert b.terminal is TerminalClass.QUIESCENT
    assert b.regions == ("filling", "filling", "filling", "overflowing")

    # entry state: magnitudes carried over, directions re-derived to steady
    entry = result.graph.nodes[b.node_ids[-1]]
    assert entry.region == "overflowing"
    assert entry.state.time is TimeTag.POINT  # instantaneous point->point edge
    space = m.variables["amount"].space
    assert entry.state["amount"].mag == space.rank_of("FULL")
    assert all(entry.state[v].dir is Qdir.STD for v in m.variables)

    # the transition node kept its filling-region identity
    transition_node = result.graph.nodes[entry.parent]
    assert transition_node.region == "filling"
    assert transition_node.state["amount"].dir is Qdir.INC


def test_guard_conjunction_blocks_equilibrium_crossing():
    # the at-FULL equilibrium has netflow == 0: the (netflow > 0) guard atom
    # must keep it from crossing into the overflow region
    m, initial, _ = two_region_bathtub()
    result = qr.qsim(m, initial)
    at_full_quiescent = [
        n
        for n in result.graph.nodes.values()
        if n.terminal is TerminalClass.QUIESCENT
        and n.region == "filling"
        and n.model.spaces[n.model.index("amount")].describe(
            n.state["amount"].mag
        )
        == "FULL"
    ]
    (node,) = at_full_quiescent
    assert node.children == []


def test_boundary_without_transition_still_region_exits():
    m, initial, _ = two_region_bathtub()
    m.region_transitions.clear()
    result = qr.qsim(m, initial)
    census = {n.terminal for n in result.graph.nodes.values() if n.terminal}
    assert TerminalClass.REGION_EXIT in census


def test_mode_channel_end_to_end():
    m, initial, (full, inflow, f1, f2) = two_region_bathtub(values=True)
    graph = qr.qsim(m, initial).graph

    dt = 0.005
    A, t = 0.0, 0.0
    rows, times, modes = [], [], []
    while A < full:
        rows.append([A, f1(A), f2(f1(A)), inflow, inflow - f2(f1(A))])
        times.append(t)
        modes.append("filling")
        prev = A
        A += dt * (inflow - f2(f1(A)))
        t += dt
        if A >= full:
            frac = (full - prev) / (A - prev)
            rows.append([full, f1(full), f2(f1(full)), inflow, inflow - f2(f1(full))])
            times.append(t - dt + frac * dt)
            modes.append("filling")
            break
    for _ in range(60):  # steady spill
        rows.append([full, f1(full), f2(f1(full)), inflow, inflow - f2(f1(full))])
        times.append(times[-1] + dt)
        modes.append("overflowing")

    cfg = AbstractionConfig(landmark_atol=1e-9, direction_eps=1e-4, debounce=3)
    obs = abstraction.abstract_trajectory(rows, m, times=times, modes=modes, config=cfg)
    assert obs.regions[-1] == "overflowing"
    assert obs.regions[0] == "filling"

    res = coverage.check(obs, graph)
    assert res.covered, res.diagnosis
    # the witness ends at the overflowing entry node
    final = graph.nodes[res.witness[-1]]
    assert final.region == "overflowing"
    assert final.terminal is TerminalClass.QUIESCENT

    # a wrong mode channel is refuted
    bad = abstraction.abstract_trajectory(
        rows, m, times=times, modes=["overflowing"] * len(rows), config=cfg
    )
    assert not coverage.check(bad, graph).covered


def test_region_validation():
    m, _, _ = two_region_bathtub()
    with pytest.raises(ValueError, match="already declared"):
        m.region("filling")
    with pytest.raises(ValueError, match="undeclared region"):
        m.transition("filling", "nope", when=())
    with pytest.raises(ValueError, match="not a landmark"):
        m.transition("filling", "overflowing", when=(("amount", "==", "NOPE"),))
    with pytest.raises(ValueError, match="guard op"):
        qr.Guard("amount", "!=", "FULL")


def test_model_schema_round_trip():
    m, initial, _ = two_region_bathtub(values=True)
    data = m.to_dict()
    assert data["schema"] == "qrlib.model/v1"
    rebuilt = qr.Model.from_dict(json.loads(json.dumps(data)))

    assert tuple(rebuilt.variables) == tuple(m.variables)
    assert rebuilt.variables["amount"].space.landmark("FULL").value == 1.0
    assert rebuilt.initial_region == "filling"
    assert len(rebuilt.region_transitions) == 1

    # semantics preserved: identical behavior structure
    original = qr.qsim(m, initial)
    replayed = qr.qsim(rebuilt, rebuilt.state(time=TimeTag.POINT, **initial.as_dict()))
    assert len(replayed.behaviors()) == len(original.behaviors())
    assert replayed.graph.export()["nodes"] == original.graph.export()["nodes"]


def test_result_export_carries_schema_and_regions():
    m, initial, _ = two_region_bathtub()
    result = qr.qsim(m, initial)
    data = result.to_dict()
    assert data["schema"] == "qrlib.result/v1"
    regions = {n["region"] for n in data["graph"]["nodes"]}
    assert regions == {"filling", "overflowing"}


def test_unsupported_schema_version_rejected():
    with pytest.raises(ValueError, match="unsupported model schema"):
        qr.Model.from_dict({"schema": "qrlib.model/v999"})
