"""Phase-space non-intersection filter (Lee & Kuipers)."""

import json
import random
from collections import Counter
from dataclasses import replace

import pytest

import qrlib as qr
from qrlib import Landmark, Qdir, TerminalClass, TimeTag
from qrlib.bridge import abstraction, coverage

from test_qsim_golden import spring
from test_soundness import CFG, rk4


def damped_spring(*, values=False):
    """x'' = -(x + f(v)), f monotone increasing through 0 (friction)."""
    m = qr.Model("damped-spring")
    lm = (Landmark("0", value=0.0),) if values else ("0",)
    for name in ("x", "v", "a", "fv", "load"):
        m.variable(name, landmarks=lm, unbounded=True)
    m.constrain(qr.Deriv("x", "v"))
    m.constrain(qr.Deriv("v", "a"))
    m.constrain(qr.MPlus("v", "fv", cvals=(("0", "0"),)))
    m.constrain(qr.Add("x", "fv", "load"))  # load = x + fv
    m.constrain(qr.Minus("load", "a"))      # a = -load
    initial = m.state(
        time=TimeTag.POINT,
        x=("0", Qdir.INC),
        v=(("0", "+inf"), Qdir.DEC),
        a=(("-inf", "0"), Qdir.INC),
        fv=(("0", "+inf"), Qdir.DEC),
        load=(("0", "+inf"), Qdir.DEC),
    )
    return m, initial


# the friction-chain variables (a, fv, load) chatter; their directions are
# irrelevant to the (x, v) phase plane
DAMPED_CFG = qr.SimConfig.classic(
    max_states=1500, max_depth=34, ignore_qdir=("a", "fv", "load")
)
NIC_CFG = replace(DAMPED_CFG, phase_pairs=(("x", "v"),))


@pytest.fixture(scope="module")
def plain_result():
    m, init = damped_spring()
    return qr.qsim(m, init, config=DAMPED_CFG)


@pytest.fixture(scope="module")
def nic_result():
    m, init = damped_spring()
    return qr.qsim(m, init, config=NIC_CFG)


@pytest.fixture(scope="module")
def nic_valued_result():
    m, init = damped_spring(values=True)
    return qr.qsim(m, init, config=NIC_CFG)


def crossing_sequence(result, b):
    """(x-description, x-rank-in-final-frame, v-dir) at each v==0 crossing."""
    graph = result.graph
    final = graph.nodes[b.node_ids[-1]].model
    fx = final.spaces[final.index("x")]
    out = []
    for nid in b.node_ids:
        node = graph.nodes[nid]
        frame = node.model
        vq = node.state["v"]
        if vq.mag == frame.spaces[frame.index("v")].rank_of("0") and vq.dir in (
            Qdir.INC,
            Qdir.DEC,
        ):
            xs = frame.spaces[frame.index("x")]
            name = xs.describe(node.state["x"].mag)
            rank = fx.rank_of(name) if name in fx.effective_landmarks else None
            out.append((name, rank, vq.dir))
    return out


# --- validation ------------------------------------------------------------


def test_phase_pair_validation():
    m, init = damped_spring()
    with pytest.raises(ValueError, match="unknown variable"):
        qr.qsim(m, init, config=qr.SimConfig(phase_pairs=(("x", "nope"),)))
    with pytest.raises(ValueError, match="two distinct"):
        qr.qsim(m, init, config=qr.SimConfig(phase_pairs=(("x", "x"),)))
    with pytest.raises(ValueError, match="Deriv"):
        qr.qsim(m, init, config=qr.SimConfig(phase_pairs=(("x", "a"),)))
    with pytest.raises(ValueError, match="direction-ignored"):
        qr.qsim(
            m,
            init,
            config=qr.SimConfig(phase_pairs=(("x", "v"),), ignore_qdir=("v",)),
        )
    with pytest.raises(ValueError, match="envisionment"):
        qr.qsim(
            m,
            init,
            config=qr.SimConfig(phase_pairs=(("x", "v"),), envisionment=True),
        )


# --- the filter must not touch real behaviors ------------------------------


def test_undamped_spring_cycle_graph_unchanged():
    # without landmark discovery the peaks stay in unnamed intervals:
    # crossings compare as unknown, nothing is provable, nothing pruned —
    # the golden cycle graph comes out identical with the filter on
    m, init = spring()
    cfg = qr.SimConfig(discover_landmarks=False)
    plain = qr.qsim(m, init, config=cfg)
    nic = qr.qsim(m, init, config=replace(cfg, phase_pairs=(("x", "v"),)))
    assert nic.graph.export() == plain.graph.export()
    assert nic.stats["phase_filtered"] == 0


def test_undamped_spring_discovery_wobbles_pruned():
    # with discovery on, plain QSIM mints peak landmarks and admits the
    # famous spurious amplitude wobbles (every monotone instance of this
    # model is conservative). Non-intersection prunes the wobbles while
    # the true closed orbit survives.
    m, init = spring()
    plain = qr.qsim(m, init, config=qr.SimConfig.classic())
    nic = qr.qsim(
        m,
        init,
        config=qr.SimConfig.classic(phase_pairs=(("x", "v"),)),
    )
    assert len(plain.behaviors()) == 199
    assert len(nic.behaviors()) == 119
    assert nic.stats["phase_filtered"] == 120
    assert any(b.terminal is TerminalClass.CYCLE for b in nic.behaviors())


# --- damped spring: the classic demonstration ------------------------------


def test_damped_spring_prunes_intersecting_behaviors(plain_result, nic_result):
    plain, nic = plain_result, nic_result

    assert len(plain.behaviors()) == 558
    assert len(nic.behaviors()) == 482
    assert nic.stats["phase_filtered"] == 106

    def families(result):
        return {
            tuple(name for name, _, _ in crossing_sequence(result, b))
            for b in result.behaviors()
            if len(crossing_sequence(result, b)) >= 3
        }

    plain_fams, nic_fams = families(plain), families(nic)
    assert nic_fams < plain_fams  # strictly fewer crossing histories
    # survivors: the closed orbit and the strictly monotone spirals
    assert nic_fams == {
        ("x*0", "x*1", "x*0"),
        ("x*0", "x*1", "x*0", "x*1"),
        ("x*0", "x*1", "x*0", "x*1", "x*0"),
        ("x*0", "x*1", "x*2"),
        ("x*0", "x*1", "x*2", "x*3"),
        ("x*0", "x*1", "x*2", "x*3", "x*4"),
    }
    # the pruned families are self-crossing curves: amplitude wobbles
    # (down-then-up on one side) and reopened closed orbits
    for wobble in (
        ("x*0", "x*1", "x*0", "x*2"),  # returns to x*0, then a new minimum
        ("x*0", "x*1", "x*2", "x*1"),  # new maximum, then the old minimum
        ("x*0", "x*1", "x*2", "x*1", "x*2"),
    ):
        assert wobble in plain_fams and wobble not in nic_fams


def test_surviving_crossings_are_monotone_or_closed(nic_result):
    nic = nic_result
    zero = None
    for b in nic.behaviors():
        seq = crossing_sequence(nic, b)
        groups = {}
        for name, rank, vdir in seq:
            assert rank is not None  # every crossing sits at a landmark
            final = nic.graph.nodes[b.node_ids[-1]].model
            zero = final.spaces[final.index("x")].rank_of("0")
            side = (rank > zero) - (rank < zero)
            groups.setdefault((vdir, side), []).append(rank)
        for ranks in groups.values():
            inc = all(a < b2 for a, b2 in zip(ranks, ranks[1:]))
            dec = all(a > b2 for a, b2 in zip(ranks, ranks[1:]))
            eq = all(a == b2 for a, b2 in zip(ranks, ranks[1:]))
            assert inc or dec or eq, ranks


def test_phase_refuted_states_are_deadends_not_spec_pruned(nic_result):
    nic = nic_result
    census = Counter(b.terminal for b in nic.behaviors())
    assert census[TerminalClass.DEADEND] == 6
    assert TerminalClass.SPEC_PRUNED not in census


# --- soundness: real damped trajectories stay covered ----------------------


@pytest.mark.parametrize("seed", range(3))
def test_damped_spring_trajectories_covered(seed, nic_valued_result):
    rng = random.Random(seed)
    c = rng.uniform(0.15, 0.6)  # underdamped: several oscillations
    m, _ = damped_spring(values=True)
    period = 2 * 3.141592653589793 / (1 - c * c / 4) ** 0.5
    dt = period / 400.0
    ys = rk4(
        lambda y: [y[1], -(y[0] + c * y[1])], [0.0, 1.0], dt,
        int(2.5 * period / dt),
    )
    rows = [[x, v, -(x + c * v), c * v, x + c * v] for x, v in ys]
    observed = abstraction.abstract_trajectory(rows, m, config=CFG)
    res = coverage.check(observed, nic_valued_result.graph)
    assert res.covered, res.diagnosis


# --- export ----------------------------------------------------------------


def test_config_exports_phase_pairs(nic_result):
    data = json.loads(json.dumps(nic_result.to_dict()))
    assert data["config"]["phase_pairs"] == [["x", "v"]]
    assert data["stats"]["phase_filtered"] == 106
