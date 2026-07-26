"""Signed-graph / orthant consistency certificates."""

import itertools
import json
import random

import pytest

import qrlib as qr
from qrlib.analysis import monotonicity


def _model(name="signs", variables=("a", "b", "c", "isolated")):
    model = qr.Model(name)
    for variable in variables:
        model.variable(variable, landmarks=("0",), unbounded=True)
    return model


def test_consistent_cycle_assigns_canonical_polarities():
    model = _model()
    model.constrain(qr.MPlus("a", "b"))
    model.constrain(qr.MMinus("b", "c"))
    model.constrain(qr.MMinus("c", "a"))

    certificate = monotonicity.check_signed_graph(model)

    assert certificate.is_consistent
    assert certificate.polarities == {"a": 1, "b": 1, "c": -1, "isolated": 1}
    assert certificate.components == (("a", "b", "c"), ("isolated",))
    assert certificate.conflict_cycle == ()
    for relation in certificate.relations:
        assert (
            certificate.polarities[relation.left]
            * certificate.polarities[relation.right]
            == relation.sign
        )
    json.dumps(certificate.to_dict())


def test_negative_cycle_returns_concrete_conflict_witness():
    model = _model(variables=("a", "b", "c"))
    model.constrain(qr.MPlus("a", "b"))
    model.constrain(qr.MPlus("b", "c"))
    model.constrain(qr.MMinus("c", "a"))

    certificate = monotonicity.check_signed_graph(model)

    assert not certificate.is_consistent
    assert len(certificate.conflict_cycle) == 3
    product = 1
    for relation in certificate.conflict_cycle:
        product *= relation.sign
    assert product == -1
    assert {relation.constraint_index for relation in certificate.conflict_cycle} == {
        0,
        1,
        2,
    }
    assert certificate.to_dict()["consistent"] is False


def test_duplicate_opposite_relationships_form_two_edge_conflict():
    model = _model(variables=("x", "y"))
    model.constrain(qr.MPlus("x", "y"))
    model.constrain(qr.MMinus("x", "y"))

    certificate = monotonicity.check_signed_graph(model)

    assert not certificate.is_consistent
    assert len(certificate.conflict_cycle) == 2
    assert {edge.kind for edge in certificate.conflict_cycle} == {"M+", "M-"}


def test_minus_participates_but_context_dependent_algebra_does_not():
    model = _model(variables=("x", "y", "z"))
    model.constrain(qr.Minus("x", "y"))
    model.constrain(qr.MPlus("x", "y"))
    model.constrain(qr.Add("x", "y", "z"))

    certificate = monotonicity.check_signed_graph(model)

    assert not certificate.is_consistent
    assert [edge.kind for edge in certificate.relations] == ["Minus", "M+"]


def test_negative_self_relation_is_inconsistent():
    model = _model(variables=("x",))
    model.constrain(qr.MMinus("x", "x"))

    certificate = monotonicity.check_signed_graph(model)

    assert not certificate.is_consistent
    assert certificate.conflict_cycle == certificate.relations


def test_region_checks_do_not_conflate_mutually_exclusive_signs():
    model = _model(variables=("x", "y"))
    positive = model.constrain(qr.MPlus("x", "y"))
    negative = model.constrain(qr.MMinus("x", "y"))
    model.region("heating", constraints=(positive,))
    model.region("cooling", constraints=(negative,))

    whole = monotonicity.check_signed_graph(model)
    heating = monotonicity.check_signed_graph(model, region="heating")
    cooling = monotonicity.check_signed_graph(model.compile(), region="cooling")

    assert not whole.is_consistent
    assert heating.is_consistent
    assert heating.polarities == {"x": 1, "y": 1}
    assert cooling.is_consistent
    assert cooling.polarities == {"x": 1, "y": -1}
    assert cooling.region == "cooling"
    assert cooling.relations[0].constraint_index == 1


def test_unknown_region_is_rejected_for_authored_and_compiled_models():
    model = _model(variables=("x",))
    model.region("only")

    for candidate in (model, model.compile()):
        try:
            monotonicity.check_signed_graph(candidate, region="missing")
        except KeyError as error:
            assert error.args == ("missing",)
        else:
            raise AssertionError("missing region should raise KeyError")


def test_implicit_default_region_matches_compiled_model():
    model = _model(variables=("x", "y"))
    model.constrain(qr.MMinus("x", "y"))

    authored = monotonicity.check_signed_graph(model, region="default")
    compiled = monotonicity.check_signed_graph(model.compile(), region="default")

    assert authored.to_dict() == compiled.to_dict()


def test_polarity_lookup_rejects_unknown_variable():
    certificate = monotonicity.check_signed_graph(_model(variables=("x",)))
    assert certificate.polarity_of("x") == 1
    try:
        certificate.polarity_of("missing")
    except KeyError as error:
        assert "unknown variable" in str(error)
    else:
        raise AssertionError("unknown variable should raise KeyError")


def _random_signed_graph(seed):
    rng = random.Random(seed)
    variables = [f"v{i}" for i in range(4)]
    constraints = []
    for _ in range(5):
        left, right = rng.sample(variables, 2)
        constraints.append(
            (rng.choice(("M+", "M-", "Minus")), left, right)
        )
    return variables, constraints


def _brute_force_consistent(variables, constraints):
    expected_sign = {"M+": 1, "M-": -1, "Minus": -1}
    for values in itertools.product((1, -1), repeat=len(variables)):
        polarities = dict(zip(variables, values))
        if all(
            polarities[left] * polarities[right] == expected_sign[kind]
            for kind, left, right in constraints
        ):
            return True
    return False


@pytest.mark.parametrize("seed", range(120))
def test_random_signed_graphs_match_exhaustive_oracle(seed):
    """A signed graph is consistent exactly when a polarity assignment exists."""
    variables, constraints = _random_signed_graph(seed)
    model = qr.Model(f"random-signed-{seed}")
    for variable in variables:
        model.variable(variable, landmarks=("0",), unbounded=True)
    for kind, left, right in constraints:
        model.constrain(f"{kind}({left}, {right})")

    certificate = monotonicity.check_signed_graph(model)
    expected = _brute_force_consistent(variables, constraints)
    assert certificate.is_consistent is expected

    if certificate.is_consistent:
        for relation in certificate.relations:
            assert (
                certificate.polarities[relation.left]
                * certificate.polarities[relation.right]
                == relation.sign
            )
    else:
        assert certificate.conflict_cycle
        product = 1
        degrees = {}
        for relation in certificate.conflict_cycle:
            product *= relation.sign
            degrees[relation.left] = degrees.get(relation.left, 0) + 1
            degrees[relation.right] = degrees.get(relation.right, 0) + 1
        assert product == -1
        assert all(degree % 2 == 0 for degree in degrees.values())


def test_random_signed_graph_sample_exercises_both_verdicts():
    """Keep the randomized oracle from becoming a one-sided rubber stamp."""
    verdicts = {
        _brute_force_consistent(*_random_signed_graph(seed))
        for seed in range(120)
    }
    assert verdicts == {False, True}
