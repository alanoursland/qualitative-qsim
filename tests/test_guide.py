"""TeQSIM-style guided simulation and temporal-logic verdicts."""

import json

import pytest

import qrlib as qr
from qrlib import Qdir, TerminalClass, guide
from qrlib.guide import F, G, U, X, Verdict, dir_is, mag

from test_qsim_golden import bathtub, spring


# --- formula mechanics -----------------------------------------------------


def test_progression_basics():
    m, init = bathtub()
    frame = m.compile()
    at0 = mag("amount", "==", "0")
    assert guide.progress(at0, init, frame) is guide.TRUE
    assert guide.progress(mag("amount", "==", "FULL"), init, frame) is guide.FALSE
    # X defers; G persists; F discharges on truth
    assert guide.progress(X(at0), init, frame) == at0
    assert guide.progress(G(at0), init, frame) == G(at0)
    assert guide.progress(F(mag("amount", "==", "FULL")), init, frame) == F(
        mag("amount", "==", "FULL")
    )
    # implication sugar and simplification
    f = at0 >> dir_is("amount", "inc")
    assert guide.progress(f, init, frame) is guide.TRUE
    assert (~guide.TRUE) is guide.FALSE
    assert (guide.TRUE & at0) == at0


def test_spec_validation():
    m, _ = bathtub()
    frame = m.compile()
    with pytest.raises(ValueError, match="unknown variable"):
        guide._validate_spec(mag("nope", "==", "0"), frame, ())
    with pytest.raises(ValueError, match="not a landmark"):
        guide._validate_spec(mag("amount", "==", "NOPE"), frame, ())
    with pytest.raises(ValueError, match="ignore_qdir"):
        guide._validate_spec(dir_is("amount", "inc"), frame, ("amount",))
    with pytest.raises(ValueError, match="magnitude op"):
        mag("amount", "!=", "0")
    with pytest.raises(ValueError, match="direction"):
        dir_is("amount", "up")


# --- guidance (pruning during generation) ----------------------------------


def test_focusing_eventually_full():
    # F cannot prune prefixes, but verdicts exclude the below-FULL branch
    m, init = bathtub()
    r = guide.guided(m, init, F(mag("amount", "==", "FULL")))
    assert len(r.satisfied) == 2  # at-FULL equilibrium + overflow
    assert len(r.violated) == 1  # equilibrium at amount*0: F(FULL) fails forever
    assert not r.universal
    assert r.result.stats["spec_filtered"] == 0


def test_safety_spec_prunes_during_generation():
    m, init = bathtub()
    r = guide.guided(m, init, G(mag("amount", "<", "FULL")))
    # both at-FULL point states are bad prefixes: pruned at attach time
    assert r.result.stats["spec_filtered"] == 2
    assert r.result.stats["nodes"] == 3  # vs 5 unguided
    (b,) = r.satisfied
    assert b.terminal is TerminalClass.QUIESCENT
    assert r.violated == () and r.universal


def test_until_prunes_the_settling_branch():
    m, init = bathtub()
    r = guide.guided(m, init, U(dir_is("amount", "inc"), mag("amount", "==", "FULL")))
    assert len(r.satisfied) == 2
    assert r.result.stats["spec_filtered"] == 1  # amount*0 equilibrium prefix dies
    assert r.universal


def test_exogenous_input_via_temporal_spec():
    # the TeQSIM headline: drop Constant(inflow) from the model entirely and
    # recover it as a temporal constraint — the guided graph is identical
    # to the Constant-model golden, export for export
    m, init = bathtub()
    m.constraints = [c for c in m.constraints if not isinstance(c, qr.Constant)]
    spec = G(mag("inflow", "==", "IF*") & dir_is("inflow", "std"))
    r = guide.guided(m, init, spec)
    golden = qr.qsim(*bathtub())
    assert r.result.graph.export() == golden.graph.export()
    assert len(r.satisfied) == len(golden.behaviors()) == 3


def test_root_violating_spec_yields_empty_graph():
    m, init = bathtub()
    r = guide.guided(m, init, G(mag("amount", ">", "0")))  # initial amount == 0
    assert r.satisfied == ()
    root = r.result.graph.nodes[r.result.graph.root]
    assert root.terminal is TerminalClass.SPEC_PRUNED
    assert len(r.result.graph.nodes) == 1


# --- exact verdicts --------------------------------------------------------


def test_lasso_verdicts_on_the_spring_cycle():
    m, init = spring()
    result = qr.qsim(m, init, config=qr.SimConfig(discover_landmarks=False))
    (cycle,) = result.behaviors()
    graph = result.graph
    # v hits 0 twice per period: recurrence holds on the loop
    assert guide.verdict(G(F(mag("v", "==", "0"))), graph, cycle) is Verdict.SATISFIED
    # x is not eventually-always positive on a cycle crossing zero
    assert guide.verdict(F(G(mag("x", ">", "0"))), graph, cycle) is Verdict.VIOLATED
    # position matters: at the start x==0 and rising
    assert (
        guide.verdict(mag("x", "==", "0") & dir_is("x", "inc"), graph, cycle)
        is Verdict.SATISFIED
    )


def test_quiescent_constant_suffix_semantics():
    m, init = bathtub()
    result = qr.qsim(m, init)
    graph = result.graph
    below_full = next(
        b
        for b in result.behaviors()
        if b.terminal is TerminalClass.QUIESCENT
        and graph.nodes[b.node_ids[-1]].model.spaces[0].describe(
            b.states[-1]["amount"].mag
        )
        == "amount*0"
    )
    # the equilibrium persists: G(dir std) eventually-forever holds
    assert (
        guide.verdict(F(G(dir_is("amount", "std"))), graph, below_full)
        is Verdict.SATISFIED
    )
    # ...and FULL is never reached on the constant suffix
    assert (
        guide.verdict(F(mag("amount", "==", "FULL")), graph, below_full)
        is Verdict.VIOLATED
    )


def test_truncation_is_undetermined():
    m, init = bathtub()
    r = guide.guided(m, init, F(mag("amount", "==", "FULL")), max_depth=1)
    assert len(r.undetermined) == 1
    assert not r.universal


def test_classify_is_pure_model_checking():
    # classify() works on an unguided result: temporal-logic queries over
    # the behavior graph, no re-simulation
    m, init = bathtub()
    result = qr.qsim(m, init)
    r = guide.classify(result, G(mag("amount", "<", "FULL")))
    assert len(r.satisfied) == 1 and len(r.violated) == 2
    assert not r.universal
    # a property that genuinely holds universally: amount never negative
    r2 = guide.classify(result, G(mag("amount", ">=", "0")))
    assert r2.universal  # sound universal proof (empty violated/undetermined)


def test_guided_result_exports_plain_data():
    m, init = bathtub()
    r = guide.guided(m, init, G(mag("amount", "<", "FULL")))
    data = r.to_dict()
    json.dumps(data)
    assert data["universal"] is True
    assert data["spec"] == "G(amount<FULL)"
    assert data["result"]["config"]["guide"] == "G(amount<FULL)"
    assert data["satisfied"] == 1
