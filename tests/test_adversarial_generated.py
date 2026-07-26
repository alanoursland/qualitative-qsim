"""Generated, metamorphic, boundary, and mutation-sensitivity checks."""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")

import qrlib as qr
from qrlib.engines import filters
from qrlib.tensor import abstraction as tensor_abstraction

from adversarial import (
    consistent_point_states,
    constraint_kinds,
    random_model,
)
from test_qsim_golden import bathtub


SEEDS = tuple(range(30))


def _semantic_stats(result):
    return {
        key: value
        for key, value in result.stats.items()
        if key != "backend"
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_models_survive_roundtrip_and_backend_changes(seed):
    """Backend selection and JSON transport cannot change model semantics."""
    model = random_model(seed)
    rebuilt = qr.Model.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )

    assert rebuilt.content_hash() == model.content_hash()
    assert rebuilt.to_dict() == model.to_dict()

    config = {
        "max_states": 80,
        "max_depth": 8,
        "discover_landmarks": False,
    }
    for initial in consistent_point_states(model):
        reference = qr.qsim(
            model,
            initial,
            config=qr.SimConfig(backend="reference", **config),
        )
        tensor = qr.qsim(
            model,
            initial,
            config=qr.SimConfig(backend="tensor", **config),
        )
        automatic = qr.qsim(
            rebuilt,
            initial,
            config=qr.SimConfig(backend="auto", **config),
        )

        assert reference.status is tensor.status is automatic.status
        assert reference.graph.export() == tensor.graph.export()
        assert reference.graph.export() == automatic.graph.export()
        assert _semantic_stats(reference) == _semantic_stats(tensor)
        assert _semantic_stats(reference) == _semantic_stats(automatic)


def test_generated_sample_exercises_every_intended_branch():
    """The deterministic corpus must not silently collapse to easy models."""
    models = [random_model(seed) for seed in SEEDS]
    kinds = set().union(*(constraint_kinds(model) for model in models))

    assert {
        "Add",
        "At",
        "Constant",
        "Deriv",
        "MMinus",
        "MPlus",
        "Minus",
    } <= kinds
    assert any(model.regions for model in models)
    assert any(not model.regions for model in models)
    assert {len(model.variables) for model in models} == {2, 3}
    assert any(
        constraint.corresponding_values
        for model in models
        for constraint in model.constraints
    )


def test_landmark_boundaries_match_tensor_quantization():
    """Points immediately around landmarks take the same monotone buckets."""
    model = qr.Model("boundary-quantization")
    model.variable(
        "x",
        landmarks=(
            qr.Landmark("low", value=-1.0),
            qr.Landmark("zero", value=0.0),
            qr.Landmark("high", value=1.0),
        ),
        unbounded=True,
    )
    frame = model.compile()
    space = frame.spaces[frame.index("x")]
    tolerance = 1e-9
    probes = sorted(
        {
            -2.0,
            -1.0 - 2 * tolerance,
            math.nextafter(-1.0, -math.inf),
            -1.0,
            math.nextafter(-1.0, math.inf),
            -0.5,
            -2 * tolerance,
            math.nextafter(0.0, -math.inf),
            0.0,
            math.nextafter(0.0, math.inf),
            2 * tolerance,
            0.5,
            math.nextafter(1.0, -math.inf),
            1.0,
            math.nextafter(1.0, math.inf),
            1.0 + 2 * tolerance,
            2.0,
        }
    )

    reference = [
        space.rank_of_value(value, atol=tolerance)
        for value in probes
    ]
    values = torch.tensor(probes, dtype=torch.float64).reshape(1, -1, 1)
    tensor = tensor_abstraction.quantize_batch(
        values,
        frame,
        tolerance,
    )[0, :, 0].tolist()

    assert tensor == reference
    assert reference == sorted(reference)
    assert any(rank % 2 == 0 for rank in reference)
    assert any(rank % 2 == 1 for rank in reference)


def test_backend_differential_oracle_is_mutation_sensitive(monkeypatch):
    """A seeded permissive-predicate fault must create a visible divergence."""
    model, initial = bathtub()
    config = {
        "max_states": 40,
        "max_depth": 8,
        "discover_landmarks": False,
    }

    # Build the tensor tables from the correct predicate before injecting the
    # reference fault. This is a controlled mutation, not production behavior.
    tensor = qr.qsim(
        model,
        initial,
        config=qr.SimConfig(backend="tensor", **config),
    )
    calls = 0

    def accept_everything(constraint, values):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(filters, "check", accept_everything)
    mutated = qr.qsim(
        model,
        initial,
        config=qr.SimConfig(backend="reference", **config),
    )

    assert calls > 0
    assert tensor.status is qr.SimStatus.COMPLETE
    assert mutated.status is qr.SimStatus.TRUNCATED
    assert tensor.graph.export() != mutated.graph.export()

