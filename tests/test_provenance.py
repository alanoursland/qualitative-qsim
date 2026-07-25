"""Stable model identity and replay-oriented result provenance."""

import json

import qrlib as qr
from qrlib import Qdir


def ordered_model(*, reverse: bool = False) -> qr.Model:
    m = qr.Model("ordered")
    variables = ("y", "x") if reverse else ("x", "y")
    for name in variables:
        m.variable(name, landmarks=("0", "ONE"), upper_unbounded=True)

    cvals = (("ONE", "ONE"), ("0", "0"))
    if reverse:
        cvals = tuple(reversed(cvals))
    monotone = qr.MPlus("x", "y", cvals=cvals)
    constant = qr.Constant("x")
    constraints = (constant, monotone) if reverse else (monotone, constant)
    for constraint in constraints:
        m.constrain(constraint)

    regions = (("high", None), ("low", (monotone,)))
    if reverse:
        regions = tuple(reversed(regions))
    for name, subset in regions:
        m.region(name, constraints=subset)
    m.initial_region = "low"

    transitions = (
        ("low", "high", (("x", "==", "ONE"), ("y", "==", "ONE"))),
        ("high", "low", (("x", "==", "0"),)),
    )
    if reverse:
        transitions = tuple(reversed(transitions))
    for source, target, guards in transitions:
        if reverse:
            guards = tuple(reversed(guards))
        m.transition(source, target, when=guards)
    return m


def constant_model() -> tuple[qr.Model, qr.QState]:
    m = qr.Model("constant")
    m.variable("x", landmarks=("0", "HIGH"), upper_unbounded=True)
    m.constrain(qr.Constant("x"))
    return m, m.state(x=("0", Qdir.STD))


def test_model_hash_is_canonical_across_authoring_order():
    forward = ordered_model()
    reverse = ordered_model(reverse=True)

    assert forward.to_dict() != reverse.to_dict()
    assert forward.content_hash() == reverse.content_hash()
    assert forward.content_hash().startswith("sha256:")
    assert len(forward.content_hash()) == len("sha256:") + 64


def test_model_hash_survives_json_and_compile_round_trips():
    model = ordered_model()
    rebuilt = qr.Model.from_dict(json.loads(json.dumps(model.to_dict())))

    assert rebuilt.content_hash() == model.content_hash()
    assert model.compile().model_hash == model.content_hash()
    assert rebuilt.compile().model_hash == model.content_hash()


def test_model_hash_changes_with_semantic_content():
    model = ordered_model()
    renamed = ordered_model()
    renamed.name = "renamed"
    changed = ordered_model()
    changed.variables["x"] = qr.Variable(
        "x",
        qr.QuantitySpace(
            (
                qr.Landmark("0", value=0.0),
                qr.Landmark("ONE", value=1.0),
            ),
            upper_unbounded=True,
        ),
    )

    assert renamed.content_hash() != model.content_hash()
    assert changed.content_hash() != model.content_hash()


def test_result_carries_pre_discovery_model_hash_for_model_and_compiled_input():
    model, initial = constant_model()
    expected = model.content_hash()

    direct = qr.qsim(model, initial)
    compiled = qr.qsim(model.compile(), initial)

    assert direct.model_hash == expected
    assert compiled.model_hash == expected
    assert direct.to_dict()["model_hash"] == expected
    assert direct.to_dict()["schema"] == "qrlib.result/v3"


def test_landmark_discovery_preserves_the_input_model_hash():
    from test_qsim_golden import bathtub

    model, initial = bathtub()
    expected = model.content_hash()
    result = qr.qsim(model, initial, config=qr.SimConfig.classic())

    assert result.stats["landmarks_minted"] > 0
    assert result.model_hash == expected
    assert {node.model.model_hash for node in result.graph.nodes.values()} == {
        expected
    }


def test_result_exports_replayable_and_opaque_filter_provenance():
    model, initial = constant_model()

    def user_filter(parent, candidate, frame):
        return True

    config = qr.SimConfig(
        successor_filters=(qr.EnergyFilter(("x",)), user_filter)
    )
    data = qr.qsim(model, initial, config=config).to_dict()
    filters = data["config"]["successor_filters"]

    assert filters[0] == {
        "replayable": True,
        "kind": "energy",
        "trend": "conserved",
        "variables": ["x"],
        "reference": "0",
    }
    assert filters[1] == {
        "kind": "opaque",
        "replayable": False,
        "module": __name__,
        "qualname": (
            "test_result_exports_replayable_and_opaque_filter_provenance"
            ".<locals>.user_filter"
        ),
    }
    json.dumps(data)


def test_different_opaque_filters_are_not_collapsed_to_the_same_count():
    model, initial = constant_model()

    def first(parent, candidate, frame):
        return True

    def second(parent, candidate, frame):
        return True

    one = qr.qsim(
        model,
        initial,
        config=qr.SimConfig(successor_filters=(first,)),
    ).to_dict()
    two = qr.qsim(
        model,
        initial,
        config=qr.SimConfig(successor_filters=(second,)),
    ).to_dict()

    assert one["config"]["successor_filters"] != two["config"]["successor_filters"]
    assert one["model_hash"] == two["model_hash"]


def test_opaque_filter_export_does_not_copy_callable_state():
    model, initial = constant_model()

    class NonCopyableFilter:
        def __call__(self, parent, candidate, frame):
            return True

        def __deepcopy__(self, memo):
            raise AssertionError("provenance export must not copy callables")

    data = qr.qsim(
        model,
        initial,
        config=qr.SimConfig(successor_filters=(NonCopyableFilter(),)),
    ).to_dict()

    descriptor = data["config"]["successor_filters"][0]
    assert descriptor["kind"] == "opaque"
    assert descriptor["replayable"] is False
    assert descriptor["qualname"].endswith(
        "test_opaque_filter_export_does_not_copy_callable_state"
        ".<locals>.NonCopyableFilter"
    )
