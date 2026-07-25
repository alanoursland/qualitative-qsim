"""Unit tests for the numeric bridge: abstraction mechanics, coverage
matching, landmark harvest/proposal."""

import math

import pytest

import qrlib as qr
from qrlib import Landmark, Qdir, TimeTag
from qrlib.bridge import abstraction, coverage, harvest
from qrlib.bridge.abstraction import AbstractionConfig, CrossingEvent


def one_var_model():
    m = qr.Model("m")
    m.variable("x", landmarks=(Landmark("0", value=0.0),), unbounded=True)
    return m.compile()


def test_transversal_crossing_synthesizes_point():
    cm = one_var_model()
    xs = [[1.0 - 0.05 * i + 0.013] for i in range(41)]  # no sample at 0 exactly
    b = abstraction.abstract_trajectory(xs, cm)
    tags = [s.time for s in b.states]
    assert tags == [TimeTag.INTERVAL, TimeTag.POINT, TimeTag.INTERVAL]
    mid = b.states[1]["x"]
    assert mid.mag == cm.spaces[0].rank_of("0")
    assert mid.dir is Qdir.DEC
    assert b.spans[1][0] == b.spans[1][1]  # zero-width synthesized point
    assert b.time_bounds[1][0] < b.time_bounds[1][1]  # crossing is bracketed


def test_refined_crossing_preserves_exact_event_time_and_sample_spans():
    cm = one_var_model()
    b = abstraction.abstract_trajectory(
        [[-1.0], [1.0]],
        cm,
        times=[0.0, 1.0],
        crossings=[CrossingEvent(0.25, "x", "0", (0.0,))],
    )
    assert [state.time for state in b.states] == [
        TimeTag.INTERVAL,
        TimeTag.POINT,
        TimeTag.INTERVAL,
    ]
    assert b.states[1]["x"] == qr.QVal(cm.spaces[0].rank_of("0"), Qdir.INC)
    assert b.spans[1] == (1, 1)
    assert b.time_bounds[1] == (0.25, 0.25)


def test_ordered_refined_crossings_resolve_coarse_undersampling():
    m = qr.Model("two-crossings")
    m.variable(
        "x",
        landmarks=(Landmark("0", value=0.0), Landmark("1", value=1.0)),
        unbounded=True,
    )
    b = abstraction.abstract_trajectory(
        [[-1.0], [2.0]],
        m,
        times=[0.0, 1.0],
        crossings=[
            CrossingEvent(0.25, "x", "0", {"x": 0.0}),
            CrossingEvent(0.75, "x", "1", {"x": 1.0}),
        ],
    )
    assert [state["x"].mag for state in b.states] == [1, 2, 3, 4, 5]
    assert b.time_bounds[1] == (0.25, 0.25)
    assert b.time_bounds[3] == (0.75, 0.75)


def test_simultaneous_refined_crossings_share_full_solver_state():
    m = qr.Model("simultaneous")
    m.variable("x", landmarks=(Landmark("0", value=0.0),), unbounded=True)
    m.variable("y", landmarks=(Landmark("0", value=0.0),), unbounded=True)
    events = [
        CrossingEvent(0.5, "x", "0", {"x": 0.0, "y": 0.0}),
        CrossingEvent(0.5, "y", "0", {"x": 0.0, "y": 0.0}),
    ]
    b = abstraction.abstract_trajectory(
        [[-1.0, -2.0], [1.0, 2.0]],
        m,
        times=[0.0, 1.0],
        crossings=events,
    )
    assert b.time_bounds[1] == (0.5, 0.5)
    assert all(b.states[1][name].mag == 2 for name in ("x", "y"))


@pytest.mark.parametrize(
    "event, match",
    [
        (CrossingEvent(0.0, "x", "0", (0.0,)), "strictly inside"),
        (CrossingEvent(0.5, "missing", "0", (0.0,)), "unknown variable"),
        (CrossingEvent(0.5, "x", "missing", (0.0,)), "unknown landmark"),
        (CrossingEvent(0.5, "x", "0", (0.1,)), "does not match"),
    ],
)
def test_refined_crossing_validation(event, match):
    with pytest.raises(ValueError, match=match):
        abstraction.abstract_trajectory(
            [[-1.0], [1.0]],
            one_var_model(),
            times=[0.0, 1.0],
            crossings=[event],
        )


def test_abstraction_requires_strictly_increasing_times():
    with pytest.raises(ValueError, match="strictly increasing"):
        abstraction.abstract_trajectory(
            [[-1.0], [0.0], [1.0]],
            one_var_model(),
            times=[0.0, 0.0, 1.0],
        )


def test_extremum_synthesizes_steady_point():
    cm = one_var_model()
    xs = [[0.5 + t * 0.1 * (2 - t * 0.1)] for t in range(1, 20)]
    b = abstraction.abstract_trajectory(xs, cm)
    dirs = [s["x"].dir for s in b.states]
    assert dirs == [Qdir.INC, Qdir.STD, Qdir.DEC]
    assert b.states[1].time is TimeTag.POINT
    assert b.states[1]["x"].mag % 2 == 1  # steady at an unnamed value


def test_landmark_endpoint_extremum_uses_instantaneous_direction():
    landmarks = (
        Landmark("NEG", value=-1.0),
        Landmark("0", value=0.0),
        Landmark("POS", value=1.0),
    )
    m = qr.Model("oscillator")
    for name in ("x", "v", "a"):
        m.variable(name, landmarks=landmarks)
    m.constrain(qr.Deriv("x", "v"))
    m.constrain(qr.Deriv("v", "a"))
    m.constrain(
        qr.Minus(
            "x",
            "a",
            cvals=(("NEG", "POS"), ("0", "0"), ("POS", "NEG")),
        )
    )
    initial = m.state(
        x=("POS", Qdir.STD),
        v=("0", Qdir.DEC),
        a=("NEG", Qdir.STD),
    )
    predicted = qr.qsim(
        m,
        initial,
        config=qr.SimConfig(
            backend="reference",
            max_states=100,
            max_depth=30,
            successor_filters=(qr.EnergyFilter(("x", "v")),),
        ),
    )
    times = [i * 2.0 * math.pi / 16 for i in range(17)]
    samples = [
        [math.cos(t), -math.sin(t), -math.cos(t)]
        for t in times
    ]
    observed = abstraction.abstract_trajectory(
        samples,
        m,
        times=times,
        config=AbstractionConfig(
            debounce=1,
            direction_eps=1e-6,
            eps_relative=False,
        ),
    )
    first = observed.states[0]
    assert first["x"].dir is Qdir.STD
    assert first["v"].dir is Qdir.DEC
    assert first["a"].dir is Qdir.STD
    result = coverage.check(observed, predicted.graph)
    assert result.covered, result.diagnosis


def test_undersampled_jump_raises():
    m = qr.Model("m")
    m.variable(
        "x",
        landmarks=(Landmark("0", value=0.0), Landmark("A", value=1.0)),
        unbounded=True,
    )
    cm = m.compile()
    with pytest.raises(ValueError, match="undersampled"):
        # jumps from (-inf,0) to (A,+inf), crossing two landmarks
        abstraction.abstract_trajectory([[-0.5], [-0.2], [1.5], [1.8]], cm)


def test_out_of_space_value_reports_variable_and_sample():
    m = qr.Model("m")
    m.variable("x", landmarks=(Landmark("0", value=0.0),), upper_unbounded=True)
    with pytest.raises(ValueError, match="variable 'x'"):
        abstraction.abstract_trajectory([[0.5], [-0.5]], m.compile())


def test_debounce_removes_chatter_runs():
    cm = one_var_model()
    base = [[0.5 + 0.01 * i] for i in range(30)]
    # one glitch sample with a locally negative slope
    base[15][0] = base[14][0] - 0.001
    b = abstraction.abstract_trajectory(
        base, cm, config=AbstractionConfig(debounce=4)
    )
    assert [s["x"].dir for s in b.states] == [Qdir.INC]


def test_relative_thresholds_project_monotone_pairs_consistently():
    import math

    from qrlib.bridge import coverage
    from qrlib.engines.filters import check_state

    m = qr.Model("monotone-asymptote")
    landmarks = (
        Landmark("0", value=0.0),
        Landmark("one", value=1.0),
    )
    m.variable("x", landmarks=landmarks)
    m.variable("y", landmarks=landmarks)
    m.constrain(qr.MPlus("x", "y", cvals=(("0", "0"), ("one", "one"))))
    count = 4000
    times = [9.0 * index / (count - 1) for index in range(count)]
    x = [1.0 - math.exp(-time) for time in times]
    rows = [[value, value ** 0.25] for value in x]

    observed = abstraction.abstract_trajectory(rows, m, times=times)
    compiled = m.compile()
    assert all(
        check_state(compiled, state) is None for state in observed.states
    )
    assert all(
        state["x"].dir is state["y"].dir for state in observed.states
    )

    graph = qr.qsim(
        m,
        m.state(x=("0", Qdir.INC), y=("0", Qdir.INC)),
        config=qr.SimConfig(max_states=200),
    ).graph
    assert coverage.check(observed, graph).covered


def test_consistency_projection_does_not_invent_motion_without_evidence():
    from qrlib.engines.filters import check_state

    m = qr.Model("invalid-monotone-pair")
    landmarks = (
        Landmark("0", value=0.0),
        Landmark("one", value=1.0),
    )
    m.variable("x", landmarks=landmarks)
    m.variable("y", landmarks=landmarks)
    m.constrain(qr.MPlus("x", "y", cvals=(("0", "0"), ("one", "one"))))
    rows = [[index / 100.0, 0.5] for index in range(100)]

    observed = abstraction.abstract_trajectory(rows, m)

    assert any(
        check_state(m.compile(), state) is not None
        for state in observed.states
    )
    assert all(state["y"].dir is Qdir.STD for state in observed.states)


def test_array_like_interop():
    cm = one_var_model()

    class FakeTensor:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

    b = abstraction.abstract_trajectory(
        FakeTensor([[0.1 + 0.1 * i] for i in range(10)]), cm
    )
    assert [s["x"].dir for s in b.states] == [Qdir.INC]


# --- coverage matching -----------------------------------------------------


def test_coverage_translates_discovered_landmarks():
    # graph node at a discovered landmark inside (0, +inf) must be covered
    # by an observation quantized to the coarser root interval
    m = qr.Model("m")
    m.variable("x", landmarks=(Landmark("0", value=0.0),), unbounded=True)
    m.variable("v", landmarks=(Landmark("0", value=0.0),), unbounded=True)
    m.variable("a", landmarks=(Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Deriv("x", "v"))
    m.constrain(qr.Deriv("v", "a"))
    m.constrain(qr.Minus("x", "a"))
    initial = m.state(
        time=TimeTag.POINT,
        x=("0", Qdir.INC),
        v=(("0", "+inf"), Qdir.STD),
        a=("0", Qdir.DEC),
    )
    result = qr.qsim(
        m, initial, config=qr.SimConfig.classic()
    )  # discovery on: n2 has x at x*0
    peak = result.graph.nodes[2]
    assert "x*0" in peak.model.spaces[peak.model.index("x")].names

    root_space = m.variables["x"].space
    obs_peak = qr.QState.from_dict(
        {
            "x": qr.QVal(root_space.rank_between("0", "+inf"), Qdir.STD),
            "v": qr.QVal(root_space.rank_of("0"), Qdir.DEC),
            "a": qr.QVal(root_space.rank_between("-inf", "0"), Qdir.STD),
        },
        TimeTag.POINT,
    )
    res = coverage.check([obs_peak], result.graph)
    assert res.covered
    assert 2 in res.witness


def test_coverage_failure_carries_diagnosis():
    m = qr.Model("m")
    m.variable("x", landmarks=(Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Constant("x"))
    initial = m.state(time=TimeTag.POINT, x=(("0", "+inf"), Qdir.STD))
    result = qr.qsim(m, initial)
    bad = qr.QState.from_dict(
        {"x": qr.QVal(m.variables["x"].space.rank_between("0", "+inf"), Qdir.INC)},
        TimeTag.POINT,
    )
    res = coverage.check([bad], result.graph)
    assert not res.covered
    assert res.divergence_index == 0
    assert "x" in res.diagnosis
    assert res.mismatches == ("x",)
    assert res.candidate_node is not None

    portable = coverage.check([bad], result.graph, ascii=True)
    assert portable.diagnosis is not None
    portable.diagnosis.encode("cp1252")
    result.graph.describe_state(bad, ascii=True).encode("cp1252")


# --- harvest / proposal ----------------------------------------------------


def test_harvest_inserts_by_value_and_reports_conflicts():
    m = qr.Model("m")
    m.variable(
        "x",
        landmarks=(Landmark("0", value=0.0), Landmark("TOP", value=10.0)),
        upper_unbounded=True,
    )
    grown = harvest.harvest_into_model(
        m, [harvest.LandmarkRecord("x", "EQ", 4.0)]
    )
    space = grown.variables["x"].space
    assert space.names == ("0", "EQ", "TOP")
    assert space.landmark("EQ").value == 4.0
    # original untouched
    assert m.variables["x"].space.names == ("0", "TOP")

    with pytest.raises(ValueError, match="duplicates"):
        harvest.harvest_into_model(m, [harvest.LandmarkRecord("x", "Z", 10.0)])
    with pytest.raises(ValueError, match="below the"):
        harvest.harvest_into_model(m, [harvest.LandmarkRecord("x", "Z", -1.0)])
    with pytest.raises(ValueError, match="numeric value"):
        harvest.harvest_into_model(m, [harvest.LandmarkRecord("x", "Z")])
    with pytest.raises(ValueError, match="unknown variable"):
        harvest.harvest_into_model(m, [harvest.LandmarkRecord("y", "Z", 1.0)])


def test_propose_landmarks_finds_steady_plateau():
    m = qr.Model("m")
    m.variable("x", landmarks=(Landmark("0", value=0.0),), upper_unbounded=True)
    # rise to a plateau at 3.0, dwell, nothing near an existing landmark
    xs = [[min(3.0, 0.1 * i)] for i in range(120)]
    (rec,) = harvest.propose_landmarks(
        xs, m, config=AbstractionConfig(direction_eps=1e-3), min_dwell=10
    )
    assert rec.variable == "x"
    assert rec.name == "x^0"
    assert abs(rec.value - 3.0) < 0.05
    # round-trip: proposals harvest cleanly
    grown = harvest.harvest_into_model(m, [rec])
    assert "x^0" in grown.variables["x"].space.names
