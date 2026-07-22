"""The declarative EnergyFilter (open-questions.md #7)."""

import pytest

import qrlib as qr
from qrlib import EnergyFilter, Trend
from qrlib.quantity import Landmark, Qdir, QuantitySpace, QVal
from qrlib.state import QState, TimeTag

from test_qsim_phase2 import energy_filter, spring


# --- the headline: tames the frictionless spring ---------------------------


def test_reproduces_the_handwritten_spring_filter():
    m, initial = spring()
    hand = qr.qsim(m, initial, config=qr.SimConfig(successor_filters=(energy_filter,)))
    for ef in (EnergyFilter(), EnergyFilter(("x", "v", "a"))):
        decl = qr.qsim(m, initial, config=qr.SimConfig(successor_filters=(ef,)))
        # byte-for-byte the same behavior graph as the bespoke filter
        assert decl.graph.export() == hand.graph.export()
        assert decl.stats["nodes"] == 17
        assert decl.stats["user_filtered"] == 8
        (b,) = decl.behaviors()
        assert b.terminal is qr.TerminalClass.CYCLE


def test_without_the_filter_the_spring_is_intractable():
    m, initial = spring()
    plain = qr.qsim(m, initial, max_states=300)
    assert plain.status is qr.SimStatus.TRUNCATED
    tamed = qr.qsim(m, initial, config=qr.SimConfig(successor_filters=(EnergyFilter(),)))
    assert tamed.status is qr.SimStatus.COMPLETE


# --- conserved vs dissipative semantics, at the candidate level ------------


class _Frame:
    """A one-variable stand-in exposing just what the filter reads."""

    def __init__(self, space):
        self.var_order = ("q",)
        self.spaces = (space,)

    def index(self, name):
        return 0


def _setup():
    # -inf < 0 < P < +inf, with P a discovered positive extremum
    space = QuantitySpace(
        (Landmark("0"), Landmark("P")), lower_unbounded=True, upper_unbounded=True
    )
    frame = _Frame(space)
    parent = QState.from_dict({"q": QVal(space.rank_of("P"), Qdir.STD)}, TimeTag.POINT)

    def cand(mag, d):
        return QState.from_dict({"q": QVal(mag, d)}, TimeTag.POINT)

    return space, frame, parent, cand


def test_conserved_pins_turning_points_to_discovered_extrema():
    space, frame, parent, cand = _setup()
    inside = space.rank_between("0", "P")  # steady at a smaller amplitude
    cons = EnergyFilter(("q",), trend=Trend.CONSERVED)
    # conservation forbids coming to rest at a fresh amplitude...
    assert cons(parent, cand(inside, Qdir.STD), frame) is False
    # ...and forbids accelerating outward at the peak
    assert cons(parent, cand(space.rank_of("P"), Qdir.INC), frame) is False
    # but permits swinging *through* the peak (a real mid-swing continuation
    # would be moving inward; INC is what's refused)
    assert cons(parent, cand(space.rank_of("P"), Qdir.DEC), frame) is True


def test_nonincreasing_allows_shrinking_but_not_growth():
    space, frame, parent, cand = _setup()
    inside = space.rank_between("0", "P")
    beyond = space.rank_between("P", "+inf")
    diss = EnergyFilter(("q",), trend="nonincreasing")  # string coerces to Trend
    assert diss.trend is Trend.NONINCREASING
    # resting at a smaller amplitude is fine when energy may dissipate...
    assert diss(parent, cand(inside, Qdir.STD), frame) is True
    # ...but a larger turning point, or outward motion at the peak, is not
    assert diss(parent, cand(beyond, Qdir.STD), frame) is False
    assert diss(parent, cand(space.rank_of("P"), Qdir.INC), frame) is False


def test_negative_side_is_symmetric():
    space = QuantitySpace(
        (Landmark("N"), Landmark("0")), lower_unbounded=True, upper_unbounded=True
    )
    frame = _Frame(space)
    parent = QState.from_dict({"q": QVal(space.rank_of("N"), Qdir.STD)}, TimeTag.POINT)
    cons = EnergyFilter(("q",))
    # at the negative extremum N, moving further negative (DEC) is refused
    bad = QState.from_dict({"q": QVal(space.rank_of("N"), Qdir.DEC)}, TimeTag.POINT)
    ok = QState.from_dict({"q": QVal(space.rank_of("N"), Qdir.INC)}, TimeTag.POINT)
    assert cons(parent, bad, frame) is False
    assert cons(parent, ok, frame) is True


def test_variables_outside_scope_are_untouched():
    # a real two-variable frame with a discovered extremum on each; a
    # filter scoped to just "q" must ignore "r" overshooting its peak
    m = qr.Model("two")
    m.variable("q", landmarks=(Landmark("0"), Landmark("P")), unbounded=True)
    m.variable("r", landmarks=(Landmark("0"), Landmark("R")), unbounded=True)
    frame = m.compile()

    def state(qmag, qdir, rmag, rdir):
        return QState.from_dict(
            {"q": QVal(qmag, qdir), "r": QVal(rmag, rdir)}, TimeTag.POINT
        )

    parent = state(frame.spaces[0].rank_of("P"), Qdir.STD,
                   frame.spaces[1].rank_of("R"), Qdir.STD)
    # r accelerating outward past its peak (would be vetoed if in scope);
    # q is a legal mid-swing (DEC through its peak)
    cand = state(frame.spaces[0].rank_of("P"), Qdir.DEC,
                 frame.spaces[1].rank_of("R"), Qdir.INC)
    assert EnergyFilter(("q",))(parent, cand, frame) is True   # r ignored
    assert EnergyFilter(("q", "r"))(parent, cand, frame) is False  # r vetoes


# --- soundness: real trajectories stay covered -----------------------------


def test_energy_filtered_graph_still_covers_real_springs():
    from qrlib.bridge import abstraction, coverage
    from test_soundness import CFG, spring_instance

    m, initial, rows = spring_instance(0)
    result = qr.qsim(
        m, initial, config=qr.SimConfig(successor_filters=(EnergyFilter(),))
    )
    assert result.status is qr.SimStatus.COMPLETE
    observed = abstraction.abstract_trajectory(rows, m, config=CFG)
    assert coverage.check(observed, result.graph).covered


# --- export ----------------------------------------------------------------


def test_describe_and_to_dict():
    ef = EnergyFilter(("x", "v"), reference="0", trend="nonincreasing")
    assert ef.describe() == "energy(nonincreasing, x, v, ref='0')"
    assert ef.to_dict() == {
        "kind": "energy",
        "trend": "nonincreasing",
        "variables": ["x", "v"],
        "reference": "0",
    }
    assert EnergyFilter().describe() == "energy(conserved, all variables, ref='0')"
