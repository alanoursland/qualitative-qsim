"""Batched interval propagation, in parity with the reference semiquant."""

import itertools
import random

import pytest

torch = pytest.importorskip("torch")

import qrlib as qr
from qrlib.quantity import Qdir, QVal
from qrlib.semiquant import Interval, _mag_range
from qrlib.tensor import interval as iv

FLOAT = torch.float64


def _pair(interval):
    return (
        torch.tensor([interval.lo], dtype=FLOAT),
        torch.tensor([interval.hi], dtype=FLOAT),
    )


def _rand_interval(rng):
    lo = rng.choice([-float("inf"), -3.0, -1.0, 0.0, 2.0]) + rng.random()
    hi = lo + rng.random() * 5
    if rng.random() < 0.15:
        hi = float("inf")
    if rng.random() < 0.15:
        lo = -float("inf")
    return Interval(min(lo, hi), max(lo, hi))


# --- primitive parity vs the reference Interval ----------------------------


def test_interval_primitives_match_reference():
    rng = random.Random(0)
    for _ in range(3000):
        a, b = _rand_interval(rng), _rand_interval(rng)
        ta, tb = _pair(a), _pair(b)
        for ref, ten in (
            (a.add(b), iv.iadd(ta, tb)),
            (a.sub(b), iv.isub(ta, tb)),
            (a.mul(b), iv.imul(ta, tb)),
            (a.neg(), iv.ineg(ta)),
            (a.intersect(b), iv.intersect(ta, tb)),
        ):
            assert float(ten[0]) == ref.lo
            assert float(ten[1]) == ref.hi
        rd, td = a.div(b), iv.idiv(ta, tb)
        if rd is None:  # divisor straddles zero -> no refinement
            assert float(td[0]) == -float("inf") and float(td[1]) == float("inf")
        else:
            assert float(td[0]) == rd.lo and float(td[1]) == rd.hi


def test_mul_zero_times_infinity_is_zero():
    # the semiquant convention: 0 * inf = 0 (infinite endpoint unattained)
    zero = (torch.tensor([0.0], dtype=FLOAT), torch.tensor([0.0], dtype=FLOAT))
    unb = (torch.tensor([0.0], dtype=FLOAT), torch.tensor([float("inf")], dtype=FLOAT))
    lo, hi = iv.imul(zero, unb)
    assert float(lo) == 0.0 and float(hi) == 0.0


# --- batched narrowing == reference fixpoint -------------------------------


def _numeric_bathtub():
    m = qr.Model("bt")
    m.variable("amount", landmarks=(qr.Landmark("0", value=0.0), qr.Landmark("FULL", value=10.0)))
    m.variable("level", landmarks=(qr.Landmark("0", value=0.0), qr.Landmark("TOP", value=5.0)))
    m.variable("outflow", landmarks=(qr.Landmark("0", value=0.0), qr.Landmark("OMAX", value=3.0)))
    m.variable("inflow", landmarks=(qr.Landmark("0", value=0.0), qr.Landmark("IF", value=2.0)))
    m.variable("netflow", landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Add("netflow", "outflow", "inflow"))
    return m.compile()


def _ref_narrow_add(frame, region, b):
    """Reference within-state ADD narrowing to a fixpoint (semiquant math)."""
    for _ in range(30):
        changed = False
        for cc in frame.constraints_of(region):
            if cc.kind != "add":
                continue
            vs = cc.vars

            def nar(i, ivl):
                nonlocal changed
                new = b[i].intersect(ivl)
                if new != b[i]:
                    b[i] = new
                    changed = True

            nar(vs[2], b[vs[0]].add(b[vs[1]]))
            nar(vs[0], b[vs[2]].sub(b[vs[1]]))
            nar(vs[1], b[vs[2]].sub(b[vs[0]]))
        if not changed:
            break
    return b


def test_batched_narrowing_matches_reference():
    frame = _numeric_bathtub()
    ranks = [range(sp.num_ranks) for sp in frame.spaces]
    states = []
    for combo in itertools.islice(itertools.product(*ranks), 300):
        states.append(
            qr.QState.from_dict(
                {n: QVal(combo[i], Qdir.STD) for i, n in enumerate(frame.var_order)},
                qr.TimeTag.POINT,
            )
        )
    lo, hi, feas = iv.narrow_batch(frame, *iv.bounds_from_states(frame, states))
    for b, st in enumerate(states):
        ref = _ref_narrow_add(
            frame,
            "default",
            [_mag_range(frame.spaces[i], st[frame.var_order[i]].mag)
             for i in range(len(frame.var_order))],
        )
        if any(r.empty for r in ref):
            assert not bool(feas[b])
        else:
            assert bool(feas[b])
            for i, r in enumerate(ref):
                assert abs(float(lo[b, i]) - r.lo) < 1e-9
                assert abs(float(hi[b, i]) - r.hi) < 1e-9


# --- feasibility screen: sound refutation ----------------------------------


def test_mult_and_at_screen():
    m = qr.Model("mult")
    m.variable("x", landmarks=(qr.Landmark("0", value=0.0), qr.Landmark("X", value=2.0)))
    m.variable("y", landmarks=(qr.Landmark("0", value=0.0), qr.Landmark("Y", value=3.0)))
    m.variable("z", landmarks=(qr.Landmark("0", value=0.0), qr.Landmark("Z", value=6.0)))
    m.constrain(qr.Mult("x", "y", "z"))
    f = m.compile()

    def st(x, y, z):
        return qr.QState.from_dict(
            {"x": QVal(f.spaces[0].rank_of(x), Qdir.STD),
             "y": QVal(f.spaces[1].rank_of(y), Qdir.STD),
             "z": QVal(f.spaces[2].rank_of(z), Qdir.STD)},
            qr.TimeTag.POINT,
        )

    # 2 * 3 = 6 is feasible; 2 * 3 = 0 is not
    mask = iv.feasible_mask(f, [st("X", "Y", "Z"), st("X", "Y", "0")])
    assert mask.tolist() == [True, False]

    ma = qr.Model("at")
    ma.variable("x", landmarks=(qr.Landmark("0", value=0.0), qr.Landmark("X", value=2.0)))
    ma.constrain(qr.At("x", "X"))
    fa = ma.compile()
    at_ok = qr.QState.from_dict({"x": QVal(fa.spaces[0].rank_of("X"), Qdir.STD)}, qr.TimeTag.POINT)
    at_no = qr.QState.from_dict({"x": QVal(fa.spaces[0].rank_of("0"), Qdir.STD)}, qr.TimeTag.POINT)
    assert iv.feasible_mask(fa, [at_ok, at_no]).tolist() == [True, False]


def test_screen_refutes_a_numerically_impossible_state():
    # the point of semi-quantitative reasoning: numbers can refute a
    # qualitatively-consistent state. With OMAX(3) > IF(2), an at-FULL
    # equilibrium (netflow 0, outflow OMAX, inflow IF) needs 3 = 2 — the
    # Add is qualitatively fine (0 + (+) = (+)) but numerically impossible.
    from qrlib.engines import filters

    frame = _numeric_bathtub()
    impossible = qr.QState.from_dict(
        {
            "amount": QVal(frame.spaces[0].rank_of("FULL"), Qdir.STD),
            "level": QVal(frame.spaces[1].rank_of("TOP"), Qdir.STD),
            "outflow": QVal(frame.spaces[2].rank_of("OMAX"), Qdir.STD),
            "inflow": QVal(frame.spaces[3].rank_of("IF"), Qdir.STD),
            "netflow": QVal(frame.spaces[4].rank_of("0"), Qdir.STD),
        },
        qr.TimeTag.POINT,
    )
    feasible_numbers = qr.QState.from_dict(
        {  # outflow in (0, OMAX) can equal inflow: numerically consistent
            "amount": QVal(frame.spaces[0].rank_between("0", "FULL"), Qdir.STD),
            "level": QVal(frame.spaces[1].rank_between("0", "TOP"), Qdir.STD),
            "outflow": QVal(frame.spaces[2].rank_between("0", "OMAX"), Qdir.STD),
            "inflow": QVal(frame.spaces[3].rank_of("IF"), Qdir.STD),
            "netflow": QVal(frame.spaces[4].rank_of("0"), Qdir.STD),
        },
        qr.TimeTag.POINT,
    )
    # both are qualitatively consistent (the sign algebra accepts them)
    assert filters.check_state(frame, impossible) is None
    assert filters.check_state(frame, feasible_numbers) is None
    # but the numeric screen refutes only the impossible one
    mask = iv.feasible_mask(frame, [impossible, feasible_numbers])
    assert mask.tolist() == [False, True]


def test_validation():
    frame = _numeric_bathtub()
    with pytest.raises(ValueError, match="must be"):
        iv.narrow_batch(frame, torch.zeros(3, 2), torch.zeros(3, 2))
