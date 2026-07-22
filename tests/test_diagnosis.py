"""Model-based diagnosis (GDE/QDOCS lineage) and the At constraint."""

import json
import math

import pytest

import qrlib as qr
from qrlib import Landmark, Qdir, TimeTag, diagnosis
from qrlib.bridge.abstraction import AbstractionConfig, abstract_trajectory
from qrlib.bridge import coverage

CFG = AbstractionConfig(landmark_atol=1e-9, direction_eps=1e-4, debounce=3)
FULL, TOP, OMAX, IF = 1.0, 1.2, 0.48, 1.0
f1 = lambda A: 1.2 * A
f2 = lambda L: 0.4 * L
DT = 0.005


def base_model():
    m = qr.Model("tank")
    m.variable("amount", landmarks=(Landmark("0", 0.0), Landmark("FULL", FULL)))
    m.variable("level", landmarks=(Landmark("0", 0.0), Landmark("TOP", TOP)))
    m.variable("outflow", landmarks=(Landmark("0", 0.0), Landmark("OMAX", OMAX)))
    m.variable("inflow", landmarks=(Landmark("0", 0.0), Landmark("IF*", IF)))
    m.variable("netflow", landmarks=(Landmark("0", 0.0),), unbounded=True)
    m.constrain(qr.MPlus("amount", "level", cvals=(("0", "0"), ("FULL", "TOP"))))
    m.constrain(qr.Add("netflow", "outflow", "inflow"))
    m.constrain(qr.Deriv("amount", "netflow"))
    return m


def components():
    drain = diagnosis.Component(
        "drain",
        {
            "ok": (qr.MPlus("level", "outflow", cvals=(("0", "0"), ("TOP", "OMAX"))),),
            "blocked": (qr.Constant("outflow"), qr.At("outflow", "0")),
        },
    )
    supply = diagnosis.Component(
        "supply",
        {
            "ok": (qr.Constant("inflow"), qr.At("inflow", "IF*")),
            "stuck_off": (qr.Constant("inflow"), qr.At("inflow", "0")),
        },
    )
    return (drain, supply)


def blocked_drain_obs(m):
    rows = [[t * DT, f1(t * DT), 0.0, IF, IF] for t in range(int(0.9 / DT))]
    return abstract_trajectory(rows, m, config=CFG)


def nominal_obs(m):
    rows, A = [], 0.0
    while A <= 0.9:
        rows.append([A, f1(A), f2(f1(A)), IF, IF - f2(f1(A))])
        A += DT * (IF - f2(f1(A)))
    return abstract_trajectory(rows, m, config=CFG)


# --- the At constraint -----------------------------------------------------


def test_at_pins_magnitude_and_serializes():
    m = qr.Model("pinned")
    m.variable("x", landmarks=(Landmark("0", 0.0), Landmark("SET", 2.0)))
    m.variable("y", landmarks=("0",), unbounded=True)
    m.constrain(qr.Constant("x"))
    m.constrain(qr.At("x", "SET"))
    m.constrain(qr.MPlus("x", "y"))

    # engine: an initial state off the pin is rejected
    bad = m.state(time=TimeTag.POINT, x=("0", Qdir.STD), y=("0", Qdir.STD))
    with pytest.raises(ValueError, match="At"):
        qr.qsim(m, bad)
    ok = m.state(time=TimeTag.POINT, x=("SET", Qdir.STD), y=(("0", "+inf"), Qdir.STD))
    result = qr.qsim(m, ok)
    assert result.behaviors()[0].terminal is qr.TerminalClass.QUIESCENT

    # schema round trip preserves the landmark argument
    rebuilt = qr.Model.from_dict(json.loads(json.dumps(m.to_dict())))
    at = next(c for c in rebuilt.constraints if isinstance(c, qr.At))
    assert (at.x, at.landmark) == ("x", "SET")

    # validation: unknown landmark rejected at constrain time
    with pytest.raises(ValueError, match="not a landmark"):
        m.constrain(qr.At("y", "SET"))


def test_at_tensor_equivalence_and_semiquant_narrowing():
    pytest.importorskip("torch")
    m = qr.Model("pinned")
    m.variable("x", landmarks=(Landmark("0", 0.0), Landmark("SET", 2.0)))
    m.variable("y", landmarks=(Landmark("0", 0.0),), unbounded=True)
    m.constrain(qr.Constant("x"))
    m.constrain(qr.At("x", "SET"))
    m.constrain(qr.MPlus("x", "y"))
    ok = m.state(time=TimeTag.POINT, x=("SET", Qdir.STD), y=(("0", "+inf"), Qdir.STD))
    ref = qr.qsim(m, ok)
    ten = qr.qsim(m, ok, config=qr.SimConfig(use_tensor=True))
    assert ref.graph.export() == ten.graph.export()

    # semiquant: At narrows to the landmark's numeric value
    (b,) = ref.behaviors()
    r = qr.semiquant.refine(ref.graph, b)
    assert r.feasible
    assert r.bounds[0]["x"] == qr.semiquant.Interval(2.0, 2.0)


def test_causal_treats_at_as_exogenous_pin():
    from qrlib.analysis import causal

    m = qr.Model("pinned")
    m.variable("x", landmarks=(Landmark("0", 0.0), Landmark("SET", 2.0)))
    m.variable("y", landmarks=("0",), unbounded=True)
    m.constrain(qr.At("x", "SET"))
    m.constrain(qr.MPlus("x", "y"))
    order = causal.causal_order(m)
    assert order.exogenous == ("x",)
    assert ("x", "y") in order.instantaneous_edges


# --- coverage pre-pass: constant observation of an equilibrium -------------


def test_constant_observation_matches_quiescent_node():
    m2 = qr.Model("still")
    m2.variable("x", landmarks=(Landmark("0", 0.0),), upper_unbounded=True)
    m2.constrain(qr.Constant("x"))
    initial = m2.state(time=TimeTag.POINT, x=(("0", "+inf"), Qdir.STD))
    graph = qr.qsim(m2, initial).graph
    rows = [[0.7]] * 40  # a window watching the equilibrium: one interval state
    obs = abstract_trajectory(rows, m2, config=CFG)
    res = coverage.check(obs, graph)
    assert res.covered
    assert res.note == "constant quiescent continuation"


# --- diagnosis scenarios ---------------------------------------------------


def test_blocked_drain_uniquely_diagnosed():
    m = base_model()
    obs = blocked_drain_obs(m)
    result = diagnosis.diagnose(m, components(), obs)
    assert result.fault_detected
    assert "MPlus" in result.normal.reason  # normal physics soundly refuted
    (d,) = result.diagnoses
    assert dict(d.modes) == {"drain": "blocked", "supply": "ok"}
    assert d.faulted == ("drain",)
    assert not d.vacuous
    assert all(e.covered for e in d.evidence)
    # normal + 2 single faults; the pair is superset-pruned
    assert result.candidates_checked == 3


def test_nominal_run_needs_no_fault():
    m = base_model()
    result = diagnosis.diagnose(m, components(), nominal_obs(m))
    assert not result.fault_detected
    assert result.normal.consistent
    assert result.diagnoses == ()
    assert result.candidates_checked == 1  # search stops at all-normal


def test_supply_stuck_off_diagnosed_via_operating_point():
    m = base_model()
    rows = [
        [a, f1(a), f2(f1(a)), 0.0, -f2(f1(a))]
        for a in (0.5 * math.exp(-0.48 * t * DT) for t in range(int(1.0 / DT)))
    ]
    obs = abstract_trajectory(rows, m, config=CFG)
    init = m.state(
        time=TimeTag.POINT,
        amount=(("0", "FULL"), Qdir.DEC),
        level=(("0", "TOP"), Qdir.DEC),
        outflow=(("0", "OMAX"), Qdir.DEC),
        inflow=("0", Qdir.STD),
        netflow=(("-inf", "0"), Qdir.INC),
    )
    result = diagnosis.diagnose(m, components(), obs, initial=init)
    assert result.fault_detected
    assert "At" in result.normal.reason  # the nominal operating point refutes
    (d,) = result.diagnoses
    assert dict(d.modes) == {"drain": "ok", "supply": "stuck_off"}


def frozen_scenario(m):
    rows = [[0.5, 0.6, 0.0, 0.0, 0.0] for _ in range(60)]
    obs = abstract_trajectory(rows, m, config=CFG)
    init = m.state(
        time=TimeTag.POINT,
        amount=(("0", "FULL"), Qdir.STD),
        level=(("0", "TOP"), Qdir.STD),
        outflow=("0", Qdir.STD),
        inflow=("0", Qdir.STD),
        netflow=("0", Qdir.STD),
    )
    return obs, init


def test_double_fault_found_at_cardinality_two():
    m = base_model()
    obs, init = frozen_scenario(m)
    result = diagnosis.diagnose(m, components(), obs, initial=init)
    assert result.fault_detected
    (d,) = result.diagnoses
    assert dict(d.modes) == {"drain": "blocked", "supply": "stuck_off"}
    assert set(d.faulted) == {"drain", "supply"}
    # normal + 2 refuted singles + 1 pair
    assert result.candidates_checked == 4
    # the frozen equilibrium is matched via the constant-continuation rule
    assert d.evidence[0].note == "constant quiescent continuation"


def test_max_faults_budget_reports_honestly():
    m = base_model()
    obs, init = frozen_scenario(m)
    result = diagnosis.diagnose(m, components(), obs, initial=init, max_faults=1)
    assert result.fault_detected
    assert result.diagnoses == ()
    assert "no candidate" in result.note


def test_multiple_observations_must_all_be_covered():
    m = base_model()
    obs = blocked_drain_obs(m)
    prefix = qr.bridge.abstraction.AbstractedBehavior(
        obs.states[:2], obs.spans[:2], obs.var_order, obs.config, obs.regions[:2]
    )
    result = diagnosis.diagnose(m, components(), [obs, prefix])
    (d,) = result.diagnoses
    assert dict(d.modes) == {"drain": "blocked", "supply": "ok"}
    assert len(d.evidence) == 2 and all(e.covered for e in d.evidence)


def test_validation_errors():
    m = base_model()
    with pytest.raises(ValueError, match="normal mode"):
        diagnosis.Component("c", {"broken": ()}, normal="ok")
    comp = diagnosis.Component("c", {"ok": (qr.Constant("nope"),)})
    with pytest.raises(ValueError, match="undeclared variable"):
        diagnosis.diagnose(m, [comp], blocked_drain_obs(m))
    dup = components()[0]
    with pytest.raises(ValueError, match="duplicate component"):
        diagnosis.diagnose(m, [dup, dup], blocked_drain_obs(m))


def test_result_exports_plain_data():
    m = base_model()
    result = diagnosis.diagnose(m, components(), blocked_drain_obs(m))
    data = result.to_dict()
    json.dumps(data)
    assert data["fault_detected"] is True
    assert data["diagnoses"][0]["modes"] == {"drain": "blocked", "supply": "ok"}
    assert data["normal"]["consistent"] is False
