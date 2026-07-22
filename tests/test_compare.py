"""Comparative statics: how an equilibrium depends on its parameters."""

import json

import pytest

import qrlib as qr
from qrlib import Qdir, TimeTag
from qrlib.analysis import compare
from qrlib.analysis.compare import Change

from test_qsim_golden import bathtub


# --- the headline: parameter -> equilibrium shift --------------------------


def test_bathtub_inflow_raises_the_equilibrium():
    m, _ = bathtub()
    up = compare.compare(m, {"inflow": +1})
    # raising the inflow raises the equilibrium amount, level, and outflow;
    # the net flow is zero at equilibrium in both systems (unchanged)
    assert up["amount"] is Change.INCREASE
    assert up["level"] is Change.INCREASE
    assert up["outflow"] is Change.INCREASE
    assert up["netflow"] is Change.UNCHANGED
    assert up.indeterminate == ()

    down = compare.compare(m, {"inflow": -1})
    assert down["amount"] is Change.DECREASE
    assert down["outflow"] is Change.DECREASE


def test_propagation_through_add_and_monotone_chain():
    # gap = drive - resp, resp integrates gap; at equilibrium gap = 0, so
    # resp tracks drive one-for-one
    m = qr.Model("servo")
    for v in ("drive", "resp", "gap"):
        m.variable(v, landmarks=("0",), unbounded=True)
    m.constrain(qr.Add("gap", "resp", "drive"))  # gap + resp = drive
    m.constrain(qr.Deriv("resp", "gap"))
    m.constrain(qr.Constant("drive"))
    r = compare.compare(m, {"drive": +1})
    assert r["resp"] is Change.INCREASE
    assert r["gap"] is Change.UNCHANGED


# --- indeterminacy is reported, never guessed ------------------------------


def test_opposing_additive_changes_are_indeterminate():
    m = qr.Model("sum")
    for v in ("a", "b", "total", "out"):
        m.variable(v, landmarks=("0",), unbounded=True)
    m.constrain(qr.Add("a", "b", "total"))
    m.constrain(qr.MPlus("total", "out"))
    m.constrain(qr.Constant("a"))
    m.constrain(qr.Constant("b"))
    opp = compare.compare(m, {"a": +1, "b": -1})
    assert opp["total"] is Change.INDETERMINATE
    assert opp["out"] is Change.INDETERMINATE  # indeterminacy propagates
    assert set(opp.indeterminate) == {"total", "out"}

    same = compare.compare(m, {"a": +1, "b": +1})
    assert same["total"] is Change.INCREASE and same["out"] is Change.INCREASE


# --- MULT needs the base operating point -----------------------------------


def mult_model():
    m = qr.Model("product")
    for v in ("force", "velocity", "power"):
        m.variable(v, landmarks=("0",), unbounded=True)
    m.constrain(qr.Mult("force", "velocity", "power"))
    m.constrain(qr.Constant("force"))
    m.constrain(qr.Constant("velocity"))
    return m


def test_mult_sign_depends_on_operating_point():
    m = mult_model()

    def base(vel_sign):
        vel = (("0", "+inf"), Qdir.STD) if vel_sign > 0 else (("-inf", "0"), Qdir.STD)
        pw = (("0", "+inf"), Qdir.STD) if vel_sign > 0 else (("-inf", "0"), Qdir.STD)
        return m.state(force=(("0", "+inf"), Qdir.STD), velocity=vel, power=pw)

    # raising force raises power when velocity > 0, lowers it when velocity < 0
    assert compare.compare(m, {"force": +1}, base=base(+1))["power"] is Change.INCREASE
    assert compare.compare(m, {"force": +1}, base=base(-1))["power"] is Change.DECREASE
    # without a base the product's sign is undecidable
    assert compare.compare(m, {"force": +1})["power"] is Change.INDETERMINATE


# --- validation + export ---------------------------------------------------


def test_validation():
    m, _ = bathtub()
    with pytest.raises(ValueError, match="unknown variable"):
        compare.compare(m, {"nope": +1})
    with pytest.raises(ValueError, match="must be -1, 0"):
        compare.compare(m, {"inflow": 2})


def test_contradiction_is_reported():
    # pin a monotone pair to move oppositely: unsatisfiable
    m = qr.Model("clash")
    m.variable("x", landmarks=("0",), unbounded=True)
    m.variable("y", landmarks=("0",), unbounded=True)
    m.constrain(qr.MPlus("x", "y"))  # y moves with x
    with pytest.raises(ValueError, match="contradictory change"):
        compare.compare(m, {"x": +1, "y": -1})


def test_narrate_and_export():
    m, _ = bathtub()
    r = compare.compare(m, {"inflow": +1})
    text = compare.narrate_comparison(r)
    assert "inflow up" in text
    assert "amount increases" in text

    data = r.to_dict()
    json.dumps(data)
    assert data["perturbation"] == {"inflow": 1}
    assert data["changes"]["amount"] == 1
    assert data["region"] == "default"
