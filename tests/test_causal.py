"""Causal ordering derived from model structure."""

import json

import qrlib as qr
from qrlib.analysis import causal

from test_qsim_golden import bathtub, spring, utube


def test_bathtub_causal_chain_and_feedback():
    m, _ = bathtub()
    order = causal.causal_order(m)
    assert not order.is_singular

    # inflow is the sole exogenous driver; amount is the state
    assert order.exogenous == ("inflow",)
    assert order.state_variables == ("amount",)

    # instantaneous chain amount -> level -> outflow -> netflow
    assert ("amount", "level") in order.instantaneous_edges
    assert ("level", "outflow") in order.instantaneous_edges
    assert ("outflow", "netflow") in order.instantaneous_edges
    assert ("inflow", "netflow") in order.instantaneous_edges
    assert order.levels["amount"] == 0
    assert order.levels["level"] == 1
    assert order.levels["outflow"] == 2
    assert order.levels["netflow"] == 3

    # integration closes the loop over time
    assert order.integration_edges == (("netflow", "amount"),)
    (loop,) = order.loops
    assert set(loop) == {"amount", "level", "outflow", "netflow"}

    # links: netflow is caused by outflow + inflow via ADD
    nf = order.link_of("netflow")
    assert nf.relation == "+"
    assert set(nf.inputs) == {"outflow", "inflow"}
    assert nf.integral is False
    amt = order.link_of("amount")
    assert amt.integral is True and amt.inputs == ("netflow",)


def test_spring_is_one_feedback_loop():
    m, _ = spring()
    order = causal.causal_order(m)
    assert order.exogenous == ()  # autonomous
    assert set(order.state_variables) == {"x", "v"}
    (loop,) = order.loops
    assert set(loop) == {"x", "v", "a"}
    # both integrations present
    assert set(order.integration_edges) == {("v", "x"), ("a", "v")}


def test_utube_autonomous_single_loop():
    m, _ = utube()
    order = causal.causal_order(m)
    assert order.exogenous == ()
    assert set(order.state_variables) == {"a", "b"}
    (loop,) = order.loops
    assert set(loop) == {"a", "b", "diff", "flow", "mflow"}
    assert not order.is_singular


def test_pure_algebraic_chain_is_acyclic():
    m = qr.Model("chain")
    for n in ("x", "y", "z"):
        m.variable(n, landmarks=("0",), unbounded=True)
    m.constrain(qr.Constant("x"))
    m.constrain(qr.MPlus("x", "y"))
    m.constrain(qr.MPlus("y", "z"))
    order = causal.causal_order(m)
    assert order.exogenous == ("x",)
    assert order.state_variables == ()
    assert order.loops == ()
    assert order.levels == {"x": 0, "y": 1, "z": 2}
    assert ("x", "y") in order.instantaneous_edges
    assert ("y", "z") in order.instantaneous_edges


def test_mult_direction_from_matching():
    # z = x * y, x constant, y constant -> z determined by x,y
    m = qr.Model("prod")
    for n in ("x", "y", "z"):
        m.variable(n, landmarks=("0",), unbounded=True)
    m.constrain(qr.Constant("x"))
    m.constrain(qr.Constant("y"))
    m.constrain(qr.Mult("x", "y", "z"))
    order = causal.causal_order(m)
    assert set(order.exogenous) == {"x", "y"}
    z = order.link_of("z")
    assert z.relation == "×" and set(z.inputs) == {"x", "y"}
    assert order.levels["z"] == 1


def test_under_determined_is_singular():
    m = qr.Model("under")
    for n in ("x", "y", "z"):
        m.variable(n, landmarks=("0",), unbounded=True)
    m.constrain(qr.Constant("x"))
    m.constrain(qr.MPlus("x", "y"))  # z has no determining equation
    order = causal.causal_order(m)
    assert order.is_singular
    assert order.singular == ("z",)


def test_over_determined_reports_redundant_equation():
    m = qr.Model("over")
    m.variable("x", landmarks=("0",), unbounded=True)
    m.constrain(qr.Constant("x"))
    m.constrain(qr.Constant("x"))  # second equation can't be matched
    order = causal.causal_order(m)
    assert order.link_of("x") is not None
    assert order.singular == ()  # no free variable
    assert order.is_singular  # but over-determined
    assert order.redundant == ("constant(x)",)


def test_compiled_model_causal_order_is_per_region():
    from test_regions import two_region_bathtub

    m, _, _ = two_region_bathtub()
    compiled = m.compile()

    # the whole constraint set unions both regions (amount is both
    # integrated in 'filling' and held constant in 'overflowing'), so it is
    # over-determined — causal ordering must be done per region
    whole = causal.causal_order(compiled)
    assert whole.is_singular
    assert whole.redundant  # a redundant equation surfaces

    # 'filling': amount is an integrated state driven by netflow
    fill = causal.causal_order(compiled, region="filling")
    assert not fill.is_singular
    assert fill.state_variables == ("amount",)
    assert fill.exogenous == ("inflow",)

    # 'overflowing' holds amount constant, so it becomes exogenous
    over = causal.causal_order(compiled, region="overflowing")
    assert not over.is_singular
    assert set(over.exogenous) == {"amount", "inflow"}
    assert over.state_variables == ()


def test_narrate_and_to_dict():
    m, _ = bathtub()
    order = causal.causal_order(m)
    text = causal.narrate_causes(order)
    assert "exogenous inputs: inflow" in text
    assert "state variables (given by integration): amount" in text
    assert "amount → level" in text
    assert "d(amount)/dt = netflow" in text
    assert "feedback loop" in text

    data = order.to_dict()
    json.dumps(data)
    assert data["exogenous"] == ["inflow"]
    assert data["levels"]["netflow"] == 3
    assert len(data["loops"]) == 1
