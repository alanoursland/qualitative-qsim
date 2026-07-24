"""Dynamic chatter abstraction (Clancy & Kuipers)."""

import json
import random
from collections import Counter

import pytest

import qrlib as qr
from qrlib import TerminalClass
from qrlib import decompose as dc
from qrlib import guide
from qrlib.bridge import abstraction, coverage
from qrlib.engines.chatter import chatter_variables
from qrlib.guide import F, dir_is, mag

from test_decsim import CASCADE_PARTS, assert_covers, cascade
from test_phase import damped_spring
from test_qsim_golden import bathtub, spring, utube
from test_soundness import CFG, rk4

AUTO_CFG = qr.SimConfig(max_states=1500, max_depth=34, dynamic_chatter=True)


@pytest.fixture(scope="module")
def auto_valued_result():
    m, init = damped_spring(values=True)
    return qr.qsim(m, init, config=AUTO_CFG)


# --- structural detection --------------------------------------------------


def test_structural_candidates():
    # damped spring: a/load are dir-free (Add is weak); fv is rigid to the
    # state variable v through M+, so it is NOT a candidate
    assert chatter_variables(damped_spring()[0]) == ("a", "load")
    # cascade: netA is anchored through the rigid chain
    # netA -Minus- outA -M+- amtA (a state variable); netB is Add-only
    assert chatter_variables(cascade()[0]) == ("netB",)
    # bathtub: Constant(inflow) makes its Add rigid; nothing chatters
    assert chatter_variables(bathtub()[0]) == ()
    assert chatter_variables(spring()[0]) == ()
    # utube: the flow class has no anchor — structurally *permitted* to
    # chatter (see below: it never actually does)
    assert chatter_variables(utube()[0]) == ("diff", "flow", "mflow")


def test_track_qdir_excludes_and_validates():
    m, init = damped_spring()
    r = qr.qsim(
        m, init,
        config=qr.SimConfig(
            max_states=200, dynamic_chatter=True, track_qdir=("load",)
        ),
    )
    assert r.stats["chatter_candidates"] == {"default": ["a"]}
    with pytest.raises(ValueError, match="track_qdir names unknown"):
        qr.qsim(m, init, config=qr.SimConfig(dynamic_chatter=True, track_qdir=("nope",)))


# --- permission is not manifestation ---------------------------------------


def test_candidates_without_wiggles_change_nothing():
    # the utube's flow class is a candidate, but filtering always forces
    # its directions: no merge ever fires and the graph is untouched —
    # exactly what static (up-front) abstraction would have lost
    m, init = utube()
    plain = qr.qsim(m, init)
    auto = qr.qsim(m, init, config=qr.SimConfig(dynamic_chatter=True))
    assert auto.stats["chatter_candidates"] == {
        "default": ["diff", "flow", "mflow"]
    }
    assert auto.stats["chatter_merged"] == 0
    assert auto.graph.export() == plain.graph.export()


def test_no_candidates_change_nothing():
    m, init = bathtub()
    plain = qr.qsim(m, init)
    auto = qr.qsim(m, init, config=qr.SimConfig(dynamic_chatter=True))
    assert auto.stats["chatter_candidates"] == {"default": []}
    assert auto.graph.export() == plain.graph.export()


# --- chatter collapses where it manifests ----------------------------------


def test_damped_spring_tamed_automatically():
    m, init = damped_spring()
    auto = qr.qsim(m, init, config=AUTO_CFG)
    assert auto.stats["chatter_candidates"] == {"default": ["a", "load"]}
    assert auto.stats["chatter_merged"] == 1481
    assert len(auto.behaviors()) == 602
    # comparable to the hand-tuned ignore_qdir run (558 behaviors at the
    # same budget) — and no variable list had to be authored


def test_cascade_completes_automatically():
    # without abstraction the cascade blows the budget on netB direction
    # wiggles; dynamic chatter finds and collapses them by itself
    m, init = cascade()
    plain = qr.qsim(m, init)
    assert plain.status is qr.SimStatus.TRUNCATED
    auto = qr.qsim(m, init, config=qr.SimConfig(dynamic_chatter=True))
    assert auto.status is qr.SimStatus.COMPLETE
    assert auto.stats["nodes"] == 19
    census = Counter(b.terminal for b in auto.behaviors())
    assert census == {TerminalClass.QUIESCENT: 4, TerminalClass.DOMAIN_EXIT: 1}


def test_composes_with_phase_filter():
    m, init = damped_spring()
    from dataclasses import replace

    nic = qr.qsim(m, init, config=replace(AUTO_CFG, phase_pairs=(("x", "v"),)))
    assert nic.stats["phase_filtered"] == 100
    assert len(nic.behaviors()) == 500


# --- interplay: guide and decsim -------------------------------------------


def test_guide_dir_atoms_are_auto_tracked():
    m, init = damped_spring()
    spec = F(mag("v", "==", "0") & dir_is("load", "std"))
    g = guide.guided(
        m, init, spec,
        config=qr.SimConfig(max_states=300, max_depth=20, dynamic_chatter=True),
    )
    assert g.result.config.track_qdir == ("load",)
    assert g.result.stats["chatter_candidates"] == {"default": ["a"]}


def test_decsim_with_dynamic_chatter():
    # the cascade decomposition needs no hand-written ignore list either:
    # tankA's net flow is rigidly anchored (no candidates), tankB's netB
    # is abstracted dynamically, and the guided interface variable outA
    # is force-tracked so its upstream word keeps direction conjuncts
    m, init = cascade()
    cfg = qr.SimConfig(dynamic_chatter=True)
    r = dc.decsim(m, init, CASCADE_PARTS, config=cfg)
    assert r.status is qr.SimStatus.COMPLETE
    tank_a, tank_b = r.runs
    assert tank_a.result.stats["chatter_candidates"] == {"default": []}
    assert tank_b.result.stats["chatter_candidates"] == {"default": ["netB"]}
    assert tank_b.result.config.track_qdir == ("outA",)
    assert len(r.joint_behaviors()) == 5
    assert_covers(r, qr.qsim(m, init, config=cfg), m)


# --- soundness: real trajectories stay covered -----------------------------


@pytest.mark.parametrize("seed", range(3))
def test_damped_spring_trajectories_covered(seed, auto_valued_result):
    rng = random.Random(seed)
    c = rng.uniform(0.15, 0.6)
    m, _ = damped_spring(values=True)
    period = 2 * 3.141592653589793 / (1 - c * c / 4) ** 0.5
    dt = period / 400.0
    ys = rk4(
        lambda y: [y[1], -(y[0] + c * y[1])], [0.0, 1.0], dt,
        int(2.5 * period / dt),
    )
    rows = [[x, v, -(x + c * v), c * v, x + c * v] for x, v in ys]
    observed = abstraction.abstract_trajectory(rows, m, config=CFG)
    res = coverage.check(observed, auto_valued_result.graph)
    assert res.covered, res.diagnosis


# --- export ----------------------------------------------------------------


def test_stats_export_is_plain_data():
    m, init = cascade()
    auto = qr.qsim(m, init, config=qr.SimConfig(dynamic_chatter=True))
    data = json.loads(json.dumps(auto.to_dict()))
    assert data["stats"]["chatter_candidates"] == {"default": ["netB"]}
    assert data["stats"]["chatter_merged"] > 0
    assert data["config"]["dynamic_chatter"] is True
