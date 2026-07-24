"""Integrity checks for the production-shaped scale qualification."""

import json
from pathlib import Path

import torch

from benchmarks.bench_scale_profiles import (
    ABSTRACTION_PROFILES,
    AbstractionProfile,
    _trajectory,
)


ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "benchmarks" / "scale-profile-2026-07-23.json"


def test_profile_contract_covers_documented_limits_and_tail_stress():
    by_name = {profile.name: profile for profile in ABSTRACTION_PROFILES}
    assert len(by_name) == len(ABSTRACTION_PROFILES)
    assert by_name["million_timesteps"].batch * by_name[
        "million_timesteps"
    ].timesteps == 1_000_000
    assert by_name["wide_v64"].variables == 64
    assert by_name["high_run_density"].pattern == "alternating"
    assert all(profile.variables <= 64 for profile in ABSTRACTION_PROFILES)


def test_synthetic_profile_generation_is_deterministic_and_shaped():
    profile = AbstractionProfile(
        "test", 2, 7, 3, "smooth", 1.0, "unit test"
    )
    first = _trajectory(profile, torch.device("cpu"))
    second = _trajectory(profile, torch.device("cpu"))
    assert first.shape == (2, 7, 3)
    assert first.dtype == torch.float64
    assert torch.equal(first, second)


def test_committed_qualification_record_matches_profile_schema():
    record = json.loads(RESULT.read_text(encoding="utf-8"))
    assert record["schema"] == "qrlib.scale-profile/v1"
    assert "no external-user telemetry" in record["qualification"]["claim_scope"]
    expected = {profile.name for profile in ABSTRACTION_PROFILES}
    observed = {row["profile"] for row in record["abstraction"]}
    assert observed == expected
    assert all(row["latency_median_s"] > 0 for row in record["abstraction"])
    assert all(0 < row["run_density"] <= 1 for row in record["abstraction"])
    assert [row["models"] for row in record["ensembles"]] == [1, 8, 32]
