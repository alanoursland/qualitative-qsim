"""Cross-surface qualification for research-preview modules."""

import pytest

import qrlib as qr
from qrlib import Qdir, SimStatus, TimeTag
from qrlib import decompose as dc
from qrlib import guide, semiquant
from qrlib.analysis import causal, compare, monotonicity
from qrlib.analysis.compare import Change
from qrlib.bridge import abstraction, coverage
from qrlib.guide import G, mag

from test_decsim import assert_covers
from test_frontends import cascade_device
from test_phase import damped_spring
from test_qsim_golden import bathtub
from test_semiquant import bathtub_with
from test_soundness import CFG, rk4


def test_device_frontend_decomposition_preserves_monolithic_behaviors():
    """A composed device must retain its behaviors after DecSIM partitioning.

    This crosses the device front end, model compilation, reference QSIM,
    guided component simulation, and DecSIM's behavior join. It complements
    the handwritten-model checks in the individual frontend and decomposition
    suites.
    """
    model = cascade_device().build()
    model.constrain(qr.Constant("A.in"))
    initial = model.state(
        time=TimeTag.POINT,
        **{
            "A.in": ("0", Qdir.STD),
            "A.out": (("0", "QMAX"), Qdir.DEC),
            "B.out": ("0", Qdir.INC),
            "A.amt": (("0", "CAP"), Qdir.DEC),
            "A.net": (("-inf", "0"), Qdir.INC),
            "B.amt": ("0", Qdir.INC),
            "B.net": (("0", "+inf"), Qdir.DEC),
        },
    )
    config = qr.SimConfig(ignore_qdir=("A.net", "B.net"))

    monolithic = qr.qsim(model, initial, config=config)
    decomposed = dc.decsim(
        model,
        initial,
        {
            "tankA": ("A.in", "A.out", "A.amt", "A.net"),
            "tankB": ("B.out", "B.amt", "B.net"),
        },
        config=config,
    )

    assert monolithic.status is SimStatus.COMPLETE
    assert decomposed.status is SimStatus.COMPLETE
    assert len(monolithic.behaviors()) == len(decomposed.joint_behaviors()) == 5
    assert decomposed.stats["component_nodes"] == {
        "tankA": 3,
        "tankB": 15,
    }
    assert_covers(decomposed, monolithic, model)


def test_temporal_guidance_and_q2_refinement_select_the_same_behavior():
    """TeQSIM focusing and Q2 feasibility agree on a numeric bathtub.

    The qualitative model admits three behaviors. A strong drain makes only
    the below-FULL equilibrium numerically feasible, while the temporal guide
    independently asks that amount always remain below FULL. Both mechanisms
    must select the same nonempty behavior rather than merely pruning.
    """
    model, initial, _ = bathtub_with(omax_bounds=(1.5, 1.6))
    config = qr.SimConfig.classic()
    unguided = qr.qsim(model, initial, config=config)
    focused = guide.guided(
        model,
        initial,
        G(mag("amount", "<", "FULL")),
        config=config,
    )

    assert len(unguided.behaviors()) == 3
    feasible = semiquant.feasible_behaviors(unguided)
    assert len(feasible) == 1
    assert len(focused.result.behaviors()) == 1
    assert focused.universal
    assert focused.result.behaviors()[0].states == feasible[0].states

    refinements = semiquant.refine_all(focused.result)
    assert len(refinements) == 1
    assert refinements[0][1].feasible


def test_induced_model_is_valid_for_causal_and_monotonicity_analysis():
    """An induced QDE remains meaningful to two independent analyses."""
    field = lambda state: [  # noqa: E731
        state[1],
        -state[0] - 0.3 * state[1],
    ]
    rows = rk4(field, [1.0, 0.0], 0.05, 300)
    induction = qr.induce.induce(rows, ["x", "v"])

    assert induction.best is not None
    assert set(induction.best.influences) == {
        ("x", "v", 1),
        ("v", "x", -1),
        ("v", "v", -1),
    }

    model = induction.best.model
    order = causal.causal_order(model)
    certificate = monotonicity.check_signed_graph(model)

    assert not order.is_singular
    assert order.exogenous == ()
    assert set(order.state_variables) == {"x", "v"}
    assert order.loops
    assert certificate.is_consistent
    assert certificate.relations
    for relation in certificate.relations:
        assert (
            certificate.polarities[relation.left]
            * certificate.polarities[relation.right]
            == relation.sign
        )


def test_causal_roles_bound_comparative_statics_parameters():
    """Comparative statics accepts a causal input, not an integrated state."""
    model, _ = bathtub()
    order = causal.causal_order(model)

    assert order.exogenous == ("inflow",)
    assert order.state_variables == ("amount",)

    changes = compare.compare(model, {"inflow": 1})
    assert changes["amount"] is Change.INCREASE
    assert changes["level"] is Change.INCREASE
    assert changes["outflow"] is Change.INCREASE
    assert changes["netflow"] is Change.UNCHANGED

    with pytest.raises(ValueError):
        compare.compare(model, {"amount": 1})


def test_energy_phase_and_chatter_filters_compose_without_losing_reality():
    """All three global pruning mechanisms act and retain a real trajectory."""
    counters = {"calls": 0, "rejections": 0}

    class CountingEnergyFilter(qr.EnergyFilter):
        def __call__(self, parent, candidate, frame):
            counters["calls"] += 1
            admitted = super().__call__(parent, candidate, frame)
            counters["rejections"] += not admitted
            return admitted

    model, initial = damped_spring(values=True)
    energy = CountingEnergyFilter(
        ("x", "v"),
        trend=qr.Trend.NONINCREASING,
    )
    result = qr.qsim(
        model,
        initial,
        config=qr.SimConfig.classic(
            max_states=1500,
            max_depth=34,
            dynamic_chatter=True,
            phase_pairs=(("x", "v"),),
            successor_filters=(energy,),
        ),
    )

    assert counters["calls"] > counters["rejections"] > 0
    assert result.stats["phase_filtered"] > 0
    assert result.stats["chatter_merged"] > 0

    damping = 0.3
    rows = rk4(
        lambda state: [
            state[1],
            -(state[0] + damping * state[1]),
        ],
        [0.0, 1.0],
        0.015,
        1000,
    )
    observed_rows = [
        [
            x,
            v,
            -(x + damping * v),
            damping * v,
            x + damping * v,
        ]
        for x, v in rows
    ]
    observed = abstraction.abstract_trajectory(
        observed_rows,
        model,
        config=CFG,
    )
    checked = coverage.check(observed, result.graph)
    assert checked.covered, checked.diagnosis
