"""Phase-6: Q2-style semi-quantitative refinement — guaranteed bounds,
transition-time bounds, envelope tightening, and numeric refutation."""

from math import inf

import pytest

import qrlib as qr
from qrlib import Landmark, Qdir, TimeTag, semiquant
from qrlib.semiquant import Envelope, Interval

from test_regions import two_region_bathtub


# --- interval arithmetic ---------------------------------------------------


def test_interval_arithmetic():
    a, b = Interval(1.0, 2.0), Interval(-3.0, 4.0)
    assert a.add(b) == Interval(-2.0, 6.0)
    assert a.sub(b) == Interval(-3.0, 5.0)
    assert a.mul(b) == Interval(-6.0, 8.0)
    assert Interval(-2.0, 3.0).mul(Interval(-5.0, 7.0)) == Interval(-15.0, 21.0)
    assert a.div(Interval(2.0, 4.0)) == Interval(0.25, 1.0)
    assert a.div(b) is None  # divisor contains zero
    assert Interval(1.0, 2.0).intersect(Interval(3.0, 4.0)).empty
    # infinities never produce NaN poison; 0 * inf = 0 (endpoints denote
    # unboundedness of real values, never attained infinities)
    assert Interval(0.0, inf).mul(Interval(0.0, 0.0)) == Interval(0.0, 0.0)
    assert Interval(-inf, inf).add(Interval(1.0, 1.0)) == Interval(-inf, inf)


# --- models ----------------------------------------------------------------


def bathtub_with(full=1.0, inflow=1.0, omax=None, omax_bounds=None):
    m = qr.Model("bathtub")
    m.variable("amount", landmarks=(Landmark("0", 0.0), Landmark("FULL", full)))
    m.variable("level", landmarks=("0", "TOP"))
    if omax is not None:
        out_lm = Landmark("OMAX", omax)
    elif omax_bounds is not None:
        out_lm = Landmark("OMAX", lo=omax_bounds[0], hi=omax_bounds[1])
    else:
        out_lm = "OMAX"
    m.variable("outflow", landmarks=(Landmark("0", 0.0), out_lm))
    m.variable("inflow", landmarks=(Landmark("0", 0.0), Landmark("IF*", inflow)))
    m.variable("netflow", landmarks=(Landmark("0", 0.0),), unbounded=True)
    c_al = m.constrain(qr.MPlus("amount", "level", cvals=(("0", "0"), ("FULL", "TOP"))))
    c_lo = m.constrain(qr.MPlus("level", "outflow", cvals=(("0", "0"), ("TOP", "OMAX"))))
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
    return m, initial, (c_al, c_lo)


# --- the classic Q2 result: transition-time bounds -------------------------


def test_overflow_time_bounds_and_correct_refutation():
    # OMAX = 0.48 < inflow = 1.0: the drain can never balance, so BOTH
    # equilibrium branches are numerically impossible and only the
    # overflow crossing survives — with the classic time bounds
    # [FULL/IF, FULL/(IF - OMAX)].
    m, initial, _ = two_region_bathtub(values=True)
    result = qr.qsim(m, initial)
    refined = semiquant.refine_all(result)
    assert len(refined) == 3

    feasible = [(b, r) for b, r in refined if r.feasible]
    assert len(feasible) == 1
    b, r = feasible[0]
    assert b.regions[-1] == "overflowing"
    crossing_time = r.times[2]  # the amount==FULL boundary instant
    assert crossing_time.lo == pytest.approx(1.0)
    assert crossing_time.hi == pytest.approx(1.0 / 0.52)

    for _, rr in refined:
        if not rr.feasible:
            assert rr.reason is not None and rr.refuted_at is not None

    # soundness: the true crossing time of a concrete instance lies inside
    dt, A, t = 0.0005, 0.0, 0.0
    while A < 1.0:
        A += dt * (1.0 - 0.4 * 1.2 * A)
        t += dt
    assert crossing_time.lo <= t <= crossing_time.hi


# --- envelope tightening ---------------------------------------------------


def band(k_lo, k_hi):
    return Envelope(
        lambda x: k_lo * x,
        lambda x: k_hi * x,
        inv_lower=lambda y: y / k_lo,
        inv_upper=lambda y: y / k_hi,
    )


def test_envelopes_pin_the_discovered_equilibrium():
    # level = f1(amount), outflow = f2(level), both within +-10% of identity;
    # at equilibrium outflow = inflow = 1, so the discovered landmark
    # amount*0 must lie in [1/1.1^2, 1/0.9^2].
    m, initial, (c_al, c_lo) = bathtub_with(full=2.0)
    result = qr.qsim(m, initial)
    env = {c_al: band(0.9, 1.1), c_lo: band(0.9, 1.1)}

    refined = semiquant.refine_all(result, envelopes=env)
    (below_full,) = [
        (b, r)
        for b, r in refined
        if r.feasible and b.terminal is qr.TerminalClass.QUIESCENT
        and b.states[-1]["amount"].mag % 2 == 0
        and result.graph.nodes[b.node_ids[-1]].model.spaces[0].describe(
            b.states[-1]["amount"].mag
        )
        == "amount*0"
    ]
    b, r = below_full
    final = r.bounds[-1]
    assert final["amount"].lo == pytest.approx(1 / 1.21)
    assert final["amount"].hi == pytest.approx(1 / 0.81)
    assert final["level"].lo == pytest.approx(1 / 1.1)
    assert final["level"].hi == pytest.approx(1 / 0.9)
    # asymptotic approach: arrival time is correctly unbounded
    assert r.times[-1].hi == inf
    assert r.times[-1].lo == 0.0


# --- pruning in both directions --------------------------------------------


def test_strong_drain_prunes_overflow():
    # OMAX in [1.5, 1.6] > inflow: overflow and at-FULL equilibrium are
    # impossible (outflow would exceed inflow); only the below-FULL
    # equilibrium survives.
    m, initial, _ = bathtub_with(omax_bounds=(1.5, 1.6))
    result = qr.qsim(m, initial)
    assert len(result.behaviors()) == 3
    feasible = semiquant.feasible_behaviors(result)
    assert len(feasible) == 1
    assert feasible[0].terminal is qr.TerminalClass.QUIESCENT
    assert feasible[0].states[-1]["amount"].mag % 2 == 0  # the amount*0 one


def test_weak_drain_prunes_equilibria():
    # OMAX in [0.4, 0.48] < inflow: outflow can never reach inflow, so
    # both equilibria are impossible; only overflow (region exit) survives.
    m, initial, _ = bathtub_with(omax_bounds=(0.4, 0.48))
    result = qr.qsim(m, initial)
    feasible = semiquant.feasible_behaviors(result)
    assert len(feasible) == 1
    assert feasible[0].terminal is qr.TerminalClass.DOMAIN_EXIT


# --- soundness of the no-annotation case -----------------------------------


def test_unannotated_model_is_never_refuted():
    from test_qsim_golden import bathtub, spring, utube

    for maker in (bathtub, utube):
        m, initial = maker()
        result = qr.qsim(m, initial)
        for b, r in semiquant.refine_all(result):
            assert r.feasible, (m.name, r.reason)

    m, initial = spring()
    result = qr.qsim(m, initial, config=qr.SimConfig(discover_landmarks=False))
    for b, r in semiquant.refine_all(result):
        assert r.feasible  # cycles refine to (unbounded) feasible bounds


# --- export ----------------------------------------------------------------


def test_result_exports_plain_data():
    import json

    m, initial, _ = two_region_bathtub(values=True)
    result = qr.qsim(m, initial)
    (b, r) = semiquant.refine_all(result)[1]
    data = r.to_dict()
    json.dumps(data)  # plain data, serializable
    assert data["feasible"] is True
    assert len(data["states"]) == len(b.states)
    s2 = data["states"][2]
    assert s2["vars"]["amount"] == [1.0, 1.0]
    assert s2["time"][0] == pytest.approx(1.0)
