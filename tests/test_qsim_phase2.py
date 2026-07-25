"""Phase-2 machinery: landmark discovery, chatter mitigation, the
successor-filter hook (energy arguments), and envisionment mode."""

from dataclasses import asdict, replace
import warnings

import pytest

import qrlib as qr
from qrlib import Qdir, SimStatus, TerminalClass, TimeTag
from qrlib.analysis import queries


def spring():
    m = qr.Model("spring")
    m.variable("x", landmarks=("0",), unbounded=True)
    m.variable("v", landmarks=("0",), unbounded=True)
    m.variable("a", landmarks=("0",), unbounded=True)
    m.constrain(qr.Deriv("x", "v"))
    m.constrain(qr.Deriv("v", "a"))
    m.constrain(qr.Minus("x", "a"))
    initial = m.state(
        time=TimeTag.POINT,
        x=("0", Qdir.INC),
        v=(("0", "+inf"), Qdir.STD),
        a=("0", Qdir.DEC),
    )
    return m, initial


def damped_spring():
    # a = -(x + v) via s = x + v, a = -s. The directions of s and a are
    # weakly constrained -> classic chatter.
    m = qr.Model("damped-spring")
    for name in ("x", "v", "s", "a"):
        m.variable(name, landmarks=("0",), unbounded=True)
    m.constrain(qr.Deriv("x", "v"))
    m.constrain(qr.Deriv("v", "a"))
    m.constrain(qr.Add("x", "v", "s"))
    m.constrain(qr.Minus("s", "a"))
    initial = m.state(
        time=TimeTag.POINT,
        x=("0", Qdir.INC),
        v=(("0", "+inf"), Qdir.DEC),
        s=(("0", "+inf"), Qdir.DEC),
        a=(("-inf", "0"), Qdir.INC),
    )
    return m, initial


def energy_filter(parent, cand, frame):
    """Energy argument for the undamped spring: no variable may pass beyond,
    or peak short of, an already-discovered extremum landmark on that side."""
    for vi, name in enumerate(frame.var_order):
        space = frame.spaces[vi]
        zero = space.rank_of("0")
        qv = cand[name]
        named = [space.rank_of(n) for n in space.names if n != "0"]
        pos = [r for r in named if r > zero]
        neg = [r for r in named if r < zero]
        if qv.dir is Qdir.STD and qv.mag % 2 == 1:  # steady at unnamed value
            if (qv.mag > zero and pos) or (qv.mag < zero and neg):
                return False
        if qv.mag % 2 == 0 and qv.mag != zero:  # at a discovered extremum
            if qv.mag in pos and qv.dir is Qdir.INC:
                return False
            if qv.mag in neg and qv.dir is Qdir.DEC:
                return False
    return True


# --- landmark discovery ----------------------------------------------------


def test_practical_profile_is_the_default_and_tames_damped_spring():
    assert qr.SimConfig() == qr.SimConfig.practical()
    assert qr.SimConfig().max_states == 512
    assert qr.SimConfig().profile == "practical"
    assert qr.SimConfig.classic().profile == "classic"
    assert qr.SimConfig.classic() == qr.SimConfig(
        discover_landmarks=True,
        dynamic_chatter=False,
    )

    spring_model, spring_initial = spring()
    classic = qr.qsim(
        spring_model,
        spring_initial,
        config=qr.SimConfig.classic(max_states=100),
    )
    explicit_previous_defaults = qr.qsim(
        spring_model,
        spring_initial,
        config=qr.SimConfig(
            discover_landmarks=True,
            dynamic_chatter=False,
            max_states=100,
        ),
    )
    assert classic.graph.export() == explicit_previous_defaults.graph.export()
    assert classic.stats == explicit_previous_defaults.stats

    m, initial = damped_spring()
    result = qr.qsim(m, initial)

    assert result.status is SimStatus.COMPLETE
    assert result.stats["nodes"] < result.config.max_states
    assert result.stats["nodes"] == 78
    assert result.stats["landmarks_minted"] == 0
    assert result.stats["chatter_merged"] > 0
    assert result.to_dict()["config"]["profile"] == "practical"


def test_spring_discovery_shows_intractable_branching():
    # Authentic QSIM: each successive extremum can be below/at/above the
    # previous one's landmark -> spurious amplitude branching, unbounded
    # landmark creation, truncation.
    m, initial = spring()
    result = qr.qsim(
        m, initial, config=qr.SimConfig.classic(max_states=300)
    )
    assert result.status is SimStatus.TRUNCATED
    assert result.stats["landmarks_minted"] > 10
    assert len(result.behaviors()) > 3
    diagnostic = result.stats["truncation"]
    assert diagnostic["limits_hit"] == ["max_states"]
    assert diagnostic["likely_cause"] == "landmark_growth"
    assert "SimConfig.practical()" in diagnostic["suggestions"][0]

    def x_beyond_first_peak(node):
        frame = node.model
        space = frame.spaces[frame.index("x")]
        if "x*0" not in space.names:
            return False
        return node.state["x"].mag > space.rank_of("x*0") > space.rank_of("0")

    assert any(x_beyond_first_peak(n) for n in result.graph.nodes.values())


def test_spring_energy_filter_tames_branching_to_the_true_cycle():
    m, initial = spring()
    result = qr.qsim(
        m,
        initial,
        config=qr.SimConfig.classic(
            successor_filters=(energy_filter,)
        ),
    )
    assert result.status is SimStatus.COMPLETE
    assert result.stats["user_filtered"] > 0  # the filter did real work

    (b,) = result.behaviors()
    assert b.terminal is TerminalClass.CYCLE
    # one calibration period discovers the four extremum landmarks
    # (x*0, x*1, v*0, v*1 — a's extremes ride along via MINUS), then the
    # cycle closes exactly one period later.
    assert result.stats["nodes"] == 17
    assert len(b.states) == 17
    final_frame = result.graph.nodes[b.node_ids[-1]].model
    x_space = final_frame.spaces[final_frame.index("x")]
    v_space = final_frame.spaces[final_frame.index("v")]
    assert {"x*0", "x*1"} <= set(x_space.names)
    assert {"v*0", "v*1"} <= set(v_space.names)


def test_discovery_mints_corresponding_values_for_minus():
    m, initial = spring()
    result = qr.qsim(
        m,
        initial,
        config=qr.SimConfig.classic(
            successor_filters=(energy_filter,)
        ),
    )
    (b,) = result.behaviors()
    frame = result.graph.nodes[b.node_ids[-1]].model
    minus = next(cc for cc in frame.constraints if cc.kind == "minus")
    x_space = frame.spaces[frame.index("x")]
    a_space = frame.spaces[frame.index("a")]
    # the peak x*0 and its acceleration partner a*0 were recorded together
    assert (x_space.rank_of("x*0"), a_space.rank_of("a*0")) in minus.cvals


def test_max_landmarks_per_variable_caps_discovery():
    m, initial = spring()
    result = qr.qsim(
        m,
        initial,
        config=qr.SimConfig.classic(
            max_landmarks_per_variable=1,
            max_states=200,
        ),
    )
    for node in result.graph.nodes.values():
        for vi in range(len(node.model.var_order)):
            space = node.model.spaces[vi]
            assert sum("*" in n for n in space.names) <= 1
    with pytest.raises(ValueError, match="max_landmarks_per_variable"):
        qr.SimConfig(max_landmarks_per_variable=-1)


def test_legacy_max_landmarks_keyword_maps_to_per_variable_cap():
    with pytest.warns(DeprecationWarning, match="max_landmarks_per_variable"):
        legacy = qr.SimConfig(max_landmarks=2)
    assert legacy.max_landmarks_per_variable == 2

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="conflicts"):
            qr.SimConfig(
                max_landmarks=2,
                max_landmarks_per_variable=3,
            )

    m, initial = damped_spring()
    with pytest.warns(DeprecationWarning):
        old = qr.qsim(
            m,
            initial,
            config=qr.SimConfig.classic(
                max_states=100,
                max_landmarks=2,
            ),
        )
    new = qr.qsim(
        m,
        initial,
        config=qr.SimConfig.classic(
            max_states=100,
            max_landmarks_per_variable=2,
        ),
    )
    assert old.graph.export() == new.graph.export()
    assert old.stats == new.stats
    assert "max_landmarks" not in old.to_dict()["config"]

    with pytest.warns(DeprecationWarning, match="max_landmarks_per_variable"):
        assert legacy.max_landmarks == 2
    assert "max_landmarks" not in asdict(legacy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        copied = replace(legacy, max_states=20)
    assert copied.max_landmarks_per_variable == 2
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]


# --- chatter mitigation ----------------------------------------------------


def test_ignore_qdir_collapses_chatter():
    m, initial = damped_spring()
    cfg = qr.SimConfig(
        discover_landmarks=False,
        dynamic_chatter=False,
        max_states=400,
    )
    plain = qr.qsim(m, initial, config=cfg)
    ignored = qr.qsim(
        m,
        initial,
        config=qr.SimConfig(
            discover_landmarks=False,
            dynamic_chatter=False,
            max_states=400,
            ignore_qdir=("s", "a"),
        ),
    )
    # chatter blows the plain run past the state budget; abstraction
    # collapses it to a small complete graph
    assert plain.status is SimStatus.TRUNCATED
    assert ignored.status is SimStatus.COMPLETE
    assert ignored.stats["nodes"] < plain.stats["nodes"] / 10
    census = queries.terminal_census(ignored.graph)
    assert TerminalClass.QUIESCENT in census  # decay to equilibrium survives

    # ignored variables are stored with untracked direction
    some_state = ignored.graph.nodes[1].state
    assert some_state["a"].dir is Qdir.IGN
    assert some_state["x"].dir is not Qdir.IGN


def test_ignore_qdir_unknown_variable_rejected():
    m, initial = spring()
    try:
        qr.qsim(m, initial, config=qr.SimConfig(ignore_qdir=("nope",)))
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected ValueError")


# --- envisionment ----------------------------------------------------------


def test_spring_envisionment_merges_the_cycle():
    m, initial = spring()
    result = qr.qsim(
        m, initial, config=qr.SimConfig(discover_landmarks=False, envisionment=True)
    )
    assert result.status is SimStatus.COMPLETE
    assert result.stats["nodes"] == 8  # the 8 distinct states of the cycle
    assert result.stats["merged"] == 1  # the closing back-edge
    (b,) = result.behaviors()
    assert b.terminal is TerminalClass.CYCLE
    assert b.cycle_target == result.graph.root
    assert len(b.states) == 9  # 8 states + the re-entry


def test_bathtub_envisionment_matches_tree():
    # no shared states in the bathtub tree: envisionment adds nothing
    m = qr.Model("bathtub")
    m.variable("amount", landmarks=("0", "FULL"))
    m.variable("level", landmarks=("0", "TOP"))
    m.variable("outflow", landmarks=("0", "OMAX"))
    m.variable("inflow", landmarks=("0", "IF*"))
    m.variable("netflow", landmarks=("0",), unbounded=True)
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
    tree = qr.qsim(m, initial)
    env = qr.qsim(m, initial, config=qr.SimConfig(envisionment=True))
    assert env.stats["nodes"] == tree.stats["nodes"] == 5
    assert len(env.behaviors()) == len(tree.behaviors()) == 3


def test_envisionment_cycles_are_structural_and_behavior_reporting_is_compact():
    from qrlib.analysis import queries

    m, initial = damped_spring()
    result = qr.qsim(
        m,
        initial,
        config=qr.SimConfig(
            max_states=150,
            discover_landmarks=False,
            envisionment=True,
        ),
    )
    loops = queries.cycles(result.graph)
    behaviors = result.behaviors()

    assert loops
    assert any(b.terminal is TerminalClass.CYCLE for b in behaviors)
    assert len(behaviors) <= len(result.graph.nodes) + len(loops)
    assert tuple(result.iter_behaviors(limit=5)) == behaviors[:5]
    assert result.behaviors(limit=0) == ()
    with pytest.raises(ValueError, match="nonnegative"):
        result.behaviors(limit=-1)
