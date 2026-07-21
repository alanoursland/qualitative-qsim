"""The soundness harness — THE invariant of the library.

QSIM guarantees that every real behavior of every concrete instance of a
qualitative model appears (abstracted) in the predicted behavior graph.
This harness instantiates models with randomized monotone functions and
parameters, integrates them numerically (RK4, pure Python), abstracts the
trajectories, and asserts coverage.

Sampling windows are finite: asymptotic approaches to equilibrium are
observed as prefixes (the quiescent point itself is the t->infinity limit);
finite-time events (overflow region exits, oscillation cycles) are matched
all the way through.
"""

import random

import pytest

import qrlib as qr
from qrlib import Landmark, Qdir, TimeTag
from qrlib.bridge import abstraction, coverage
from qrlib.bridge.abstraction import AbstractionConfig

CFG = AbstractionConfig(landmark_atol=1e-9, direction_eps=1e-4, debounce=3)


def rk4(f, y0, dt, steps, stop=None):
    ys = [list(y0)]
    y = list(y0)
    for _ in range(steps):
        k1 = f(y)
        k2 = f([yi + dt / 2 * ki for yi, ki in zip(y, k1)])
        k3 = f([yi + dt / 2 * ki for yi, ki in zip(y, k2)])
        k4 = f([yi + dt * ki for yi, ki in zip(y, k3)])
        y = [
            yi + dt / 6 * (a + 2 * b + 2 * c + d)
            for yi, a, b, c, d in zip(y, k1, k2, k3, k4)
        ]
        if stop is not None and stop(y):
            break
        ys.append(list(y))
    return ys


# --- bathtub ---------------------------------------------------------------


def bathtub_instance(seed, *, overflow):
    """A concrete bathtub: level = f1(amount), outflow = f2(level), both
    randomized monotone power laws with f(0)=0; constant inflow."""
    rng = random.Random(seed)
    a1, p1 = rng.uniform(0.5, 2.0), rng.uniform(0.8, 1.3)
    a2, p2 = rng.uniform(0.5, 2.0), rng.uniform(0.8, 1.3)
    inflow = rng.uniform(0.5, 1.5)
    f1 = lambda A: a1 * A**p1
    f2 = lambda L: a2 * L**p2
    amount_eq = (inflow / (a2 * a1**p2)) ** (1.0 / (p1 * p2))
    full = 0.6 * amount_eq if overflow else 2.0 * amount_eq

    m = qr.Model("bathtub")
    m.variable("amount", landmarks=(Landmark("0", value=0.0), Landmark("FULL", value=full)))
    m.variable("level", landmarks=(Landmark("0", value=0.0), Landmark("TOP", value=f1(full))))
    m.variable("outflow", landmarks=(Landmark("0", value=0.0), Landmark("OMAX", value=f2(f1(full)))))
    m.variable("inflow", landmarks=(Landmark("0", value=0.0), Landmark("IF*", value=inflow)))
    m.variable("netflow", landmarks=(Landmark("0", value=0.0),), unbounded=True)
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

    net = lambda A: inflow - f2(f1(A))
    dt = 0.002 * amount_eq / inflow
    if overflow:
        amounts, times = [0.0], [0.0]
        A = 0.0
        while A < full:
            prev = A
            A = rk4(lambda y: [net(y[0])], [A], dt, 1)[-1][0]
            if A >= full:
                # append the exact boundary crossing at its interpolated time
                frac = (full - prev) / (A - prev)
                amounts.append(full)
                times.append(times[-1] + frac * dt)
                break
            amounts.append(A)
            times.append(times[-1] + dt)
    else:
        amounts = [y[0] for y in rk4(lambda y: [net(y[0])], [0.0], dt, 100000,
                                     stop=lambda y: net(y[0]) < 0.05 * inflow)]
        assert net(amounts[-1]) >= 0.04 * inflow  # window ends mid-approach
        times = [i * dt for i in range(len(amounts))]
    rows = [[A, f1(A), f2(f1(A)), inflow, net(A)] for A in amounts]
    return m, initial, rows, times


@pytest.mark.parametrize("seed", range(4))
def test_bathtub_equilibrium_trajectories_are_covered(seed):
    m, initial, rows, times = bathtub_instance(seed, overflow=False)
    graph = qr.qsim(m, initial).graph
    observed = abstraction.abstract_trajectory(rows, m, times=times, config=CFG)
    res = coverage.check(observed, graph)
    assert res.covered, res.diagnosis
    assert res.matched == res.total == 2  # initial point + rising interval


@pytest.mark.parametrize("seed", range(4))
def test_bathtub_overflow_trajectories_reach_region_exit(seed):
    m, initial, rows, times = bathtub_instance(seed, overflow=True)
    result = qr.qsim(m, initial)
    observed = abstraction.abstract_trajectory(rows, m, times=times, config=CFG)
    res = coverage.check(observed, result.graph)
    assert res.covered, res.diagnosis
    assert res.matched == res.total == 3  # point, interval, boundary point
    final_node = result.graph.nodes[res.witness[-1]]
    assert final_node.terminal is qr.TerminalClass.REGION_EXIT


def test_bathtub_score_is_one_across_instances():
    m0, initial, _, _ = bathtub_instance(0, overflow=False)
    graph = qr.qsim(m0, initial).graph
    observations = []
    for seed in range(3):
        # same qualitative model; different concrete instances of it
        m, _, rows, times = bathtub_instance(seed, overflow=False)
        # quantize against each instance's own landmark values
        observations.append(abstraction.abstract_trajectory(rows, m, times=times, config=CFG))
    fraction, results = coverage.score(observations, graph)
    assert fraction == 1.0


# --- spring ----------------------------------------------------------------


def spring_instance(seed):
    rng = random.Random(seed)
    k = rng.uniform(0.5, 2.0)
    m = qr.Model("spring")
    for name in ("x", "v", "a"):
        m.variable(name, landmarks=(Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Deriv("x", "v"))
    m.constrain(qr.Deriv("v", "a"))
    m.constrain(qr.MMinus("x", "a", cvals=(("0", "0"),)))  # a = -k x
    initial = m.state(
        time=TimeTag.POINT,
        x=("0", Qdir.INC),
        v=(("0", "+inf"), Qdir.STD),
        a=("0", Qdir.DEC),
    )
    period = 2 * 3.141592653589793 / k**0.5
    dt = period / 400.0
    ys = rk4(lambda y: [y[1], -k * y[0]], [0.0, 1.0], dt, int(2.2 * period / dt))
    rows = [[x, v, -k * x] for x, v in ys]
    return m, initial, rows


@pytest.mark.parametrize("seed", range(3))
def test_spring_two_periods_covered_by_cycle_graph(seed):
    m, initial, rows = spring_instance(seed)
    graph = qr.qsim(
        m, initial, config=qr.SimConfig(discover_landmarks=False)
    ).graph
    observed = abstraction.abstract_trajectory(rows, m, config=CFG)
    res = coverage.check(observed, graph)
    assert res.covered, res.diagnosis
    # 2.2 periods: matching necessarily followed the CYCLE closure
    assert res.total > 9
    assert res.matched == res.total


def test_spring_covered_by_discovery_graph_via_frame_translation():
    # the energy-filtered discovery graph puts peaks AT discovered
    # landmarks (x*0...); observations quantized to the root space must
    # still be covered (containment translation)
    from test_qsim_phase2 import energy_filter

    m, initial, rows = spring_instance(0)
    result = qr.qsim(
        m, initial, config=qr.SimConfig(successor_filters=(energy_filter,))
    )
    assert result.status is qr.SimStatus.COMPLETE
    observed = abstraction.abstract_trajectory(rows, m, config=CFG)
    res = coverage.check(observed, result.graph)
    assert res.covered, res.diagnosis


# --- U-tube ----------------------------------------------------------------


def test_utube_trajectory_is_covered():
    c = 0.7
    m = qr.Model("u-tube")
    m.variable("a", landmarks=(Landmark("0", value=0.0),), upper_unbounded=True)
    m.variable("b", landmarks=(Landmark("0", value=0.0),), upper_unbounded=True)
    m.variable("diff", landmarks=(Landmark("0", value=0.0),), unbounded=True)
    m.variable("flow", landmarks=(Landmark("0", value=0.0),), unbounded=True)
    m.variable("mflow", landmarks=(Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Add("diff", "b", "a"))
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
    graph = qr.qsim(m, initial).graph

    d0 = 1.5
    ys = rk4(
        lambda y: [-c * (y[0] - y[1]), c * (y[0] - y[1])],
        [2.0, 0.5],
        0.01,
        100000,
        stop=lambda y: (y[0] - y[1]) < 0.05 * d0,
    )
    rows = [[a, b, a - b, c * (a - b), -c * (a - b)] for a, b in ys]
    observed = abstraction.abstract_trajectory(rows, m, config=CFG)
    res = coverage.check(observed, graph)
    assert res.covered, res.diagnosis


# --- a violating trajectory is refuted -------------------------------------


def test_violating_trajectory_is_refuted_with_diagnosis():
    m, initial, _, _ = bathtub_instance(0, overflow=False)
    graph = qr.qsim(m, initial).graph
    full = m.variables["amount"].space.landmark("FULL").value
    inflow = m.variables["inflow"].space.landmark("IF*").value
    # fabricated: amount draining with positive inflow and falling netflow —
    # inconsistent with the model from this initial state
    rows = [
        [0.4 * full - 0.001 * t, 0.3 - 0.0008 * t, 0.2 - 0.0005 * t, inflow, -0.1 - 0.0001 * t]
        for t in range(50)
    ]
    observed = abstraction.abstract_trajectory(rows, m, config=CFG)
    res = coverage.check(observed, graph)
    assert not res.covered
    assert res.divergence_index == 0
    assert res.diagnosis is not None
