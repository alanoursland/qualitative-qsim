"""Workload-aware QSIM reference/tensor backend dispatch."""

import importlib
import sys

import pytest

import qrlib as qr
from qrlib import QVal, Qdir

qsim_module = importlib.import_module("qrlib.engines.qsim")


def binary_domains(count):
    domain = [QVal(1, Qdir.DEC), QVal(1, Qdir.INC)]
    return [domain[:] for _ in range(count)]


def constrained_frame(count=2):
    model = qr.Model(f"chain-{count}")
    for index in range(count):
        model.variable(f"x{index}", landmarks=("0",), unbounded=True)
    for index in range(count - 1):
        model.constrain(qr.MPlus(f"x{index}", f"x{index + 1}"))
    return model.compile()


def constant_simulation(config=None):
    model = qr.Model("constant")
    model.variable("x", landmarks=("0", "HIGH"), upper_unbounded=True)
    model.constrain(qr.Constant("x"))
    initial = model.state(x=("0", Qdir.STD))
    return qr.qsim(model, initial, config=config)


def test_backend_config_validation_and_legacy_override():
    assert qr.SimConfig().backend_mode == "auto"
    assert qr.SimConfig(backend="reference").backend_mode == "reference"
    assert qr.SimConfig(backend="tensor").backend_mode == "tensor"
    assert qr.SimConfig(use_tensor=False).backend_mode == "reference"
    assert qr.SimConfig(use_tensor=True).backend_mode == "tensor"

    with pytest.raises(ValueError, match="backend must be"):
        qr.SimConfig(backend="gpu")
    with pytest.raises(ValueError, match="conflicts"):
        qr.SimConfig(backend="reference", use_tensor=True)
    with pytest.raises(TypeError, match="bool or None"):
        qr.SimConfig(use_tensor="yes")


def test_auto_policy_threshold_and_constraint_requirement():
    cfg = qr.SimConfig()

    assert qsim_module._select_backend(
        cfg, binary_domains(10), (0,)
    ) == ("reference", "auto_below_threshold")
    assert qsim_module._select_backend(
        cfg, binary_domains(11), (0,)
    ) == ("tensor", "auto_at_or_above_threshold")
    assert qsim_module._select_backend(
        cfg, binary_domains(12), ()
    ) == ("reference", "auto_no_active_constraints")


def test_explicit_modes_ignore_workload_shape():
    small = binary_domains(2)
    assert qsim_module._select_backend(
        qr.SimConfig(backend="reference"), small, (0,)
    ) == ("reference", "explicit_reference")
    assert qsim_module._select_backend(
        qr.SimConfig(backend="tensor"), small, (0,)
    ) == ("tensor", "explicit_tensor")


def test_qsim_reports_actual_backend_calls_and_reasons():
    auto = constant_simulation()
    reference = constant_simulation(qr.SimConfig(backend="reference"))
    tensor = constant_simulation(qr.SimConfig(backend="tensor"))

    auto_stats = auto.stats["backend"]
    assert auto_stats["requested"] == "auto"
    assert auto_stats["reference_filter_calls"] > 0
    assert auto_stats["tensor_filter_calls"] == 0
    assert auto_stats["selection_reasons"]["auto_below_threshold"] > 0

    ref_stats = reference.stats["backend"]
    assert ref_stats["requested"] == "reference"
    assert ref_stats["selection_reasons"]["explicit_reference"] > 0

    tensor_stats = tensor.stats["backend"]
    assert tensor_stats["requested"] == "tensor"
    assert tensor_stats["tensor_filter_calls"] > 0
    assert tensor_stats["selection_reasons"]["explicit_tensor"] > 0

    assert auto.graph.export() == reference.graph.export() == tensor.graph.export()


def test_auto_executes_tensor_and_matches_reference_above_threshold():
    frame = constrained_frame(11)
    domains = binary_domains(11)
    active_idx = frame.regions[0].constraint_idx
    auto_cfg = qr.SimConfig()
    ref_cfg = qr.SimConfig(backend="reference")
    auto_stats = {"backend": qsim_module._new_backend_stats(auto_cfg)}
    ref_stats = {"backend": qsim_module._new_backend_stats(ref_cfg)}

    auto = qsim_module._filtered_combos(
        frame, domains, active_idx, auto_cfg, auto_stats
    )
    reference = list(
        qsim_module._filtered_combos(
            frame, domains, active_idx, ref_cfg, ref_stats
        )
    )

    assert auto == reference
    assert auto_stats["backend"]["tensor_filter_calls"] == 1
    assert auto_stats["backend"]["reference_filter_calls"] == 0
    assert auto_stats["backend"]["selection_reasons"] == {
        "auto_at_or_above_threshold": 1
    }


def test_auto_falls_back_when_tensor_module_is_unavailable(monkeypatch):
    frame = constrained_frame(11)
    cfg = qr.SimConfig()
    stats = {"backend": qsim_module._new_backend_stats(cfg)}
    monkeypatch.setitem(sys.modules, "qrlib.tensor.engine", None)

    result = list(
        qsim_module._filtered_combos(
            frame,
            binary_domains(11),
            frame.regions[0].constraint_idx,
            cfg,
            stats,
        )
    )

    assert result
    assert stats["backend"]["reference_filter_calls"] == 1
    assert stats["backend"]["tensor_filter_calls"] == 0
    assert stats["backend"]["fallback_reasons"] == {"tensor_unavailable": 1}


def test_explicit_tensor_reports_unavailability(monkeypatch):
    frame = constrained_frame(2)
    cfg = qr.SimConfig(backend="tensor")
    stats = {"backend": qsim_module._new_backend_stats(cfg)}
    monkeypatch.setitem(sys.modules, "qrlib.tensor.engine", None)

    with pytest.raises(RuntimeError, match="tensor backend requested"):
        qsim_module._filtered_combos(
            frame,
            binary_domains(2),
            frame.regions[0].constraint_idx,
            cfg,
            stats,
        )


def test_tensor_oversized_product_fallback_is_reported(monkeypatch):
    from qrlib.tensor import engine as tensor_engine

    frame = constrained_frame(2)
    domains = [
        [QVal(1, direction) for direction in (Qdir.DEC, Qdir.STD, Qdir.INC)]
        for _ in range(2)
    ]
    cfg = qr.SimConfig(backend="tensor")
    stats = {"backend": qsim_module._new_backend_stats(cfg)}
    monkeypatch.setattr(tensor_engine, "ASSEMBLE_CAP", 4)

    result = qsim_module._filtered_combos(
        frame,
        domains,
        frame.regions[0].constraint_idx,
        cfg,
        stats,
    )

    assert result
    assert stats["backend"]["tensor_filter_calls"] == 1
    assert stats["backend"]["fallback_reasons"] == {"oversized_product": 1}
